import tensorflow as tf
import numpy as np
from utils import Make_Video_batch, make_checkpoint_folder
import sys


def gauss_cross_entropy(mu1, var1, mu2, var2):
    """
    Computes the element-wise cross entropy
    Given q(z) ~ N(z| mu1, var1)
    returns E_q[ log N(z| mu2, var2) ]
    args:
        mu1:  mean of expectation (batch, tmax, 2) tf variable
        var1: var  of expectation (batch, tmax, 2) tf variable
        mu2:  mean of integrand (batch, tmax, 2) tf variable
        var2: var of integrand (batch, tmax, 2) tf variable

    returns:
        cross_entropy: (batch, tmax, 2) tf variable
    """

    term0 = 1.8378770664093453 # log(2*pi)
    term1 = 2*tf.log(var2)
    term2 = (mu1**2 + var1 - 2*mu1*mu2 + mu2**2) / var2

    cross_entropy = -0.5*( term0 + term1 + term2)

    return cross_entropy

def build_MLP_inference_graph(vid_batch, layers=[250], tftype=tf.float32):
    """
    Takes a placeholder for batches of videos to be fed in, returns 
    a mean and var of 2d latent space that are tf variables.

    args:
        vid_batch: tf placeholder (batch, tmax, width, height)
        layers: list of widths of fully connected layers
        tftype: data type to use in graph

    returns:
        means:  tf variable, (batch, tmax, 2) x,y points
        vars:  tf variable, (batch, tmax, 2) x,y uncertainties
    """

    batch, tmax, px, py = vid_batch.get_shape()

    # first layer, flatten images to vectors
    h0 = tf.reshape(vid_batch, (batch*tmax, px*py))

    # loop over layers in given list
    for l in layers:
        i_dims = int(h0.get_shape()[-1])
        W = tf.Variable(tf.truncated_normal([i_dims, l],
                stddev=1.0 / np.sqrt(float(i_dims))), name="encW")
        B = tf.Variable(tf.zeros([1, l]), name="encB")
        h0 = tf.matmul(h0, W) + B
        h0 = tf.nn.tanh(h0)

    # final layer just outputs x,y mean and log(var) of q network
    i_dims = int(h0.get_shape()[-1])
    W = tf.Variable(tf.truncated_normal([i_dims, 4],
            stddev=1.0 / np.sqrt(float(i_dims))), name="encW")
    B = tf.Variable(tf.zeros([1, 4]), name="encB")
    h0 = tf.matmul(h0, W) + B
    # h0 = tf.nn.relu(h0)

    h0 = tf.reshape(h0, (batch, tmax, 4))
    # print(h0.get_shape())

    q_means = h0[:, :, :2]
    q_vars  = tf.abs(tf.exp(h0[:, :, 2:]))

    # print(h0.get_shape())
    # print(q_means.get_shape())
    # print(q_vars.get_shape())
    # import sys; sys.exit()

    tf.print(q_means)
    return q_means, q_vars

def build_MLP_decoder_graph(latent_samples, px, py, layers=[250]):
    """
    args:
        latent_samples: (batch, tmax, 2), tf variable
        px: image width (int)
        py: image height (int)
        layers: list of num. of nodes (list ints)

    returns:
        pred_batch_vid_logits: (batch, tmax, px, py) tf variable
    """

    batch, tmax, _ = latent_samples.get_shape()

    # flatten all frames into one matrix
    h0 = tf.reshape(latent_samples, (batch*tmax, 2))
    # print(h0.get_shape())

    # loop over layers in given list
    for l in layers:
        i_dims = int(h0.get_shape()[-1])
        W = tf.Variable(tf.truncated_normal([i_dims, l],
                stddev=1.0 / np.sqrt(float(i_dims))), name="decW")
        B = tf.Variable(tf.zeros([1, l]), name="decB")
        h0 = tf.matmul(h0, W) + B
        h0 = tf.nn.tanh(h0)


    # final layer just outputs full video batch
    l = px*py
    i_dims = int(h0.get_shape()[-1])
    W = tf.Variable(tf.truncated_normal([i_dims, l],
            stddev=1.0 / np.sqrt(float(i_dims))), name="decW")
    B = tf.Variable(tf.zeros([1, l]), name="decB")
    h0 = tf.matmul(h0, W) + B

    pred_vid_batch_logits = tf.reshape(h0, (batch, tmax, px, py))

    return pred_vid_batch_logits

def build_gp_lhood_and_post_graph(latent_mean, latent_var, lt=5):
    """
    args:
        latent_mean: (batch, tmax, 2) tf variable
        latent_var: (batch, tmax, 2) tf variable

    returns:
        gp_lhood: shape=(batch), tf variable
        post_mu: shape: (batch, tmax, 2) tf variable
        post_var: shape: (batch, tmax, 2) tf variable
    """

    batch, tmax, _ = latent_mean.get_shape()
    batch = int(batch)
    tmax = int(tmax)

    # rename and reshape latents! (batch, tmax, 1)
    lm_x = tf.reshape(latent_mean[:,:,0], (batch, tmax, 1))
    lv_x = latent_var[:,:,0] # (batch, tmax)

    lm_y = tf.reshape(latent_mean[:,:,1], (batch, tmax, 1))
    lv_y = latent_var[:,:,1] # (batch, tmax)

    # all kernel matrices can be computed once and stored as constants
    k_mat = np.arange(tmax)
    k_mat = np.exp(( (k_mat.reshape(-1,1) - k_mat.reshape(1,-1))**2)*(-0.5/lt**2))
    k_jit = k_mat + 0.0001*np.eye(tmax)

    k_mat = k_mat.reshape((1, tmax, tmax))

    k_mat = np.tile(k_mat, [batch, 1, 1])
    K = tf.constant(k_mat, dtype=tf.float32) # shape: (batch, tmax, tmax)
    eK = tf.reshape(K, (batch, tmax, 1, tmax)) # need this later for post_var

    k_jit = np.tile(k_jit.reshape(1, tmax, tmax), [batch, 1, 1])
    Kj = tf.constant(k_jit, dtype=tf.float32) # shape: (batch, tmax, tmax)

    KX = Kj + tf.matrix_diag(lv_x) # (batch, tmax, tmax)
    KY = Kj + tf.matrix_diag(lv_y)

    chol_X = tf.cholesky(KX) # (batch, tmax, tmax)
    chol_Y = tf.cholesky(KY)

    # print(tf.matrix_diag(lv_x[:,:,0]).get_shape())
    # import sys; sys.exit()

    # Compute prior_KL = marginal likelihood first
    logdet_Kx = 2*tf.reduce_sum(tf.log(tf.matrix_diag_part(chol_X)), 1) # (batch,)
    logdet_Ky = 2*tf.reduce_sum(tf.log(tf.matrix_diag_part(chol_Y)), 1)

    # print(lm_x.get_shape())
    # print(lm_y.get_shape())
    # sys.exit()

    iK_x = tf.cholesky_solve(chol_X, lm_x) # (batch, tmax, 1)
    iK_y = tf.cholesky_solve(chol_Y, lm_y)

    x_iK_x = tf.matmul(lm_x, iK_x, transpose_a=True)[:, 0, 0] # (batch, 1, 1) -> (batch,)
    y_iK_y = tf.matmul(lm_y, iK_y, transpose_a=True)[:, 0, 0]

    gp_lhood = -0.5*( logdet_Kx + logdet_Ky + x_iK_x + y_iK_y ) # (batch,)


    # Now compute posterior mean
    post_mu_x = tf.matmul(K, iK_x) # (batch, tmax, 1)
    post_mu_y = tf.matmul(K, iK_y)
    post_mu = tf.concat([post_mu_x, post_mu_y], 2) # (batch, tmax, 2)


    # posterior variance, we only want diagonal elements
    # so do batch * tmax matmuls each matmul is (1,tmax)*(tmax,1))
    iK_kx = tf.cholesky_solve(chol_X, K) # (batch, tmax, tmax)
    iK_ky = tf.cholesky_solve(chol_Y, K)
    iK_kx = tf.transpose(iK_kx, (0,2,1))  # (batch, tmax, tmax)
    iK_ky = tf.transpose(iK_ky, (0,2,1))
    iK_kx = tf.reshape(iK_kx, (batch, tmax, tmax, 1))
    iK_ky = tf.reshape(iK_ky, (batch, tmax, tmax, 1))

    # eK (batch, tmax, 1, tmax), iK_kx (batch, tmax, tmax, 1)
    post_var_x = 1.0 - tf.matmul(eK, iK_kx) # (batch, tmax, 1, 1)
    post_var_y = 1.0 - tf.matmul(eK, iK_ky)
    post_var_x = tf.reshape(post_var_x, (batch, tmax, 1))
    post_var_y = tf.reshape(post_var_y, (batch, tmax, 1))
    post_var = tf.concat([post_var_x, post_var_y], 2) # (batch, tmax, 2)


    return gp_lhood, post_mu, post_var

def build_elbo_graph(vid_batch, beta, lt=5):
    """
    Takes placeholder inputs and build complete elbo graph
    args:
        vid_batch: (batch, tmax, px, py) tf placeholder
        beta: shape=(1) or (), tf placeholder 
    
    returns:
        prior_kl: tf variable (batch)
        recon_err: tf variabel (batch)
        elbo: tf variable (batch)
    """

    batch, tmax, px, py = [int(nn) for nn in vid_batch.get_shape()]

    # first encode images
    qnet_mean, qnet_var = build_MLP_inference_graph(vid_batch)

    # prior KL = gp_lhood + cross_entropy(gp_post, qnet)
    gp_lhood, post_mean, post_var = build_gp_lhood_and_post_graph(qnet_mean, qnet_var, lt=lt)
    ce_term = gauss_cross_entropy(post_mean, post_var, qnet_mean, qnet_var) # (batch, tmax, 2)
    ce_term = tf.reduce_sum(ce_term, (1,2)) # (batch)
    elbo_prior_kl = gp_lhood - ce_term # (batch)

    # recon error for repam trick
    epsilon = tf.random.normal(shape=(batch, tmax, 2))
    latent_samples = post_mean + epsilon * tf.sqrt(post_var)
    pred_vid_batch_logits = build_MLP_decoder_graph(latent_samples, px, py)
    recon_err = tf.nn.sigmoid_cross_entropy_with_logits(labels=vid_batch, 
                                                        logits=pred_vid_batch_logits)
    elbo_recon = tf.reduce_sum(-recon_err, (1,2,3))

    # Finally put them together to get elbo!
    elbo = elbo_recon + beta*elbo_prior_kl

    return elbo_prior_kl, elbo_recon, elbo, qnet_mean, qnet_var, post_mean, post_var, pred_vid_batch_logits

def build_np_elbo_graph(vid_batch, beta, context=0.5):
    """
    Takes placeholder inputs and build complete elbo graph
    args:
        vid_batch: (batch, tmax, px, py) tf placeholder
        beta: shape=(1) or (), tf placeholder 
    
    returns:
        prior_kl: tf variable (batch)
        recon_err: tf variabel (batch)
        elbo: tf variable (batch)
    """

    qnet_mu, qnet_var = build_MLP_inference_graph(vid_batch)

    mask = 




if __name__=="__main__0":

    # Data settings
    batch = 45
    tmax = 30
    px = 32
    py = 32
    r = 3
    lt = 5
    
    # alias the data generator
    Make_data = lambda seed: Make_Video_batch(tmax=tmax, 
                                              px=px, 
                                              py=py,
                                              lt=lt, 
                                              batch=batch, 
                                              r=r, 
                                              seed=seed)

    Test_Data = Make_data(0)

    # make sure everything is created in the same graph!
    graph = tf.Graph()
    with graph.as_default():

        # placeholders to start the graph
        beta = tf.placeholder(dtype=tf.float32, shape=())
        vid_batch = tf.placeholder(shape=(batch, tmax, px, py), dtype=tf.float32)
        
        # just make the fucking graphs
        q_mu, q_var = build_MLP_inference_graph(vid_batch)
        gp_lhood, p_mu, p_var = build_gp_lhood_and_post_graph(q_mu, q_var)

        # initializer ops for the graph
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            print("Session started")
            
            sess.run(init_op)
            lhood = sess.run(p_var, {vid_batch:Test_Data})

            print("gp likelihood")
            print(lhood)

            print("negative gp likelihood ratio")
            print(np.mean(lhood<0))


if __name__=="__main__":

    # Data settings
    batch = 45
    tmax = 8
    px = 32
    py = 32
    r = 3
    lt = 5

    # alias the data generator
    Make_data = lambda seed: Make_Video_batch(tmax=tmax, 
                                              px=px, 
                                              py=py,
                                              lt=lt, 
                                              batch=batch, 
                                              r=r, 
                                              seed=seed)

    # make a test batch
    Test_Data = Make_data(0)

    # make sure everything is created in the same graph!
    graph = tf.Graph()
    with graph.as_default():

        # placeholders to start the graph
        beta = tf.placeholder(dtype=tf.float32, shape=())
        vid_batch = tf.placeholder(shape=(batch, tmax, px, py), dtype=tf.float32)

        # make the graph and get aaallll the intermediate values
        p_kl, recon, elbo, q_m, q_v, p_m, p_v, _ = build_elbo_graph(vid_batch, beta, lt=lt)

        # the actual loss functions!
        av_p_kl  = tf.reduce_mean(p_kl)
        av_recon = tf.reduce_mean(recon)
        av_elbo  = tf.reduce_mean(elbo)


        # add optimizer ops to graph (minimizing neg elbo!), print out trainable vars
        global_step = tf.Variable(0, name='global_step',trainable=False)
        optimizer  = tf.train.AdamOptimizer()
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        optim_step = optimizer.minimize(loss=-av_elbo, 
                                        var_list=train_vars,
                                        global_step=global_step)
        print("\n\nTrainable variables:")
        for v in train_vars:
            print(v)
        
        # initializer ops for the graph
        init_op = tf.global_variables_initializer()

        # Make a folder to save everything
        chkpnt_dir = make_checkpoint_folder()
        saver = tf.train.Saver()
        print("\nCheckpoint Directory:\n"+str(chkpnt_dir)+"\n")
        
        beta_0 = 100

        # Now let's start doing some computation!
        with tf.Session() as sess:

            # attempt a restore
            try:
                saver.restore(sess, tf.train.latest_checkpoint(chkpnt_dir))
                print("\n\nRestored Model")
            except:
                sess.run(init_op)
                print("\n\nInitialised Model")

            # start training that elbo!
            for t in range(1, 200):
                
                # prior KL annealing factor
                beta_t = 1 #+ (beta_0-1) * np.exp(t/2000)

                # Train, do an optim step
                Data = Make_data(t)
                _ = sess.run(optim_step, {vid_batch:Data, beta:beta_t})

                # Test, don't do an optim step
                test_elbo, g_s = sess.run([av_elbo, global_step], {vid_batch:Test_Data, beta:1.0})
                print(str(g_s)+": elbo "+str(test_elbo))

                # Checkpoint
                if g_s%100==1:
                    saver.save(sess, chkpnt_dir+"model", global_step=g_s)
                    print("\n\nModel Saved: "+ chkpnt_dir +"\n\n")