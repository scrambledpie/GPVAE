import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import math_ops as tfmath_ops
from utils import Make_Video_batch, make_checkpoint_folder
from utils import build_video_batch_graph, plot_latents, MSE_rotation
from utils import pandas_res_saver
import sys
import time
import pickle
import os
from utils_circles_grid import Make_circles, Make_squares, plot_circle
from utils_circles_grid import plot_heatmap, plot_square


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
    term1 = tf.log(var2)
    term2 = (var1 + mu1**2 - 2*mu1*mu2 + mu2**2) / var2

    cross_entropy = -0.5*( term0 + term1 + term2 )

    return cross_entropy


def gauss_entropy(var1):
    """
    Computes the element-wise entropy
    Given q(z) ~ N(z| mu1, var1)
    returns E_q[ log N(z| mu1, var1) ] = -0.5 ( log(var1) + 1 + log(2*pi) )
    args:
        var1: var  of expectation (batch, tmax, 2) tf variable

    returns:
        cross_entropy: (batch, tmax, 2) tf variable
    """

    term0 = tf.log(var1) + 2.8378770664093453 # 1 + log(2*pi)

    cross_entropy = -0.5*( term0 )

    return cross_entropy


def build_MLP_inference_graph(vid_batch, layers=[500], tftype=tf.float32):
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

    h0 = tf.reshape(h0, (batch, tmax, 4))

    q_means = h0[:, :, :2]
    q_vars  = tf.exp(h0[:, :, 2:])

    return q_means, q_vars


def build_MLP_decoder_graph(latent_samples, px, py, layers=[500]):
    """
    Constructs a TF graph that goes from latent points in 2D time series
    to a bernoulli probabilty for each pixel in output video time series.
    Args:
        latent_samples: (batch, tmax, 2), tf variable
        px: image width (int)
        py: image height (int)
        layers: list of num. of nodes (list of ints)

    Returns:
        pred_batch_vid_logits: (batch, tmax, px, py) tf variable
    """

    batch, tmax, _ = latent_samples.get_shape()

    # flatten all latents into one matrix (decoded in i.i.d fashion)
    h0 = tf.reshape(latent_samples, (batch*tmax, 2))

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


def build_1d_gp(X, Y, varY, X_test, lt=5):
    """
    Takes input-output dataset and returns post mean, var, marginal lhood.
    This is standard GP regression (in this application X is time, Y is 
    recognition network means with noise as recognition netowrk variance).

    Args:
        X: inputs tensor (batch, npoints)
        Y: outputs tensor (batch, npoints)
        varY: noise of outputs tensor (batch, npoints)
        X_test: (batch, ns) input points to compute post mean + var

    Returns:
        p_m: (batch, ns) post mean at X_test
        p_v: (batch, ns) post var at X_test
        logZ: (batch) marginal lhood of each dataset in batch
    """

    # Prepare all constants
    batch, _ = X.get_shape()
    n = tf.shape(X)[1]
    _, ns = X_test.get_shape()

    # inverse square length scale
    ilt = tf.constant( -0.5*(1/(lt*lt)) )
    
    # lhood term 1/3
    lhood_pi_term = tf.cast(n, dtype=tf.float32) * np.log(2*np.pi)
    
    # data cov matrix K = exp( -1/2 * (X-X)**2/l**2) + noise
    K = tf.reshape(X, (batch, n, 1)) - tf.reshape(X, (batch, 1, n)) # (batch, n n)
    K = tf.exp( (K**2) * ilt)  + tf.matrix_diag(varY) 
    chol_K = tf.linalg.cholesky(K) # (batch, n, n)

    # lhood term 2/3
    lhood_logdet_term = 2*tf.reduce_sum(tf.log(tf.matrix_diag_part(chol_K)), 1) # (batch)

    # lhood term 3/3
    Y = tf.reshape(Y, (batch, n, 1))
    iKY = tf.cholesky_solve( chol_K, Y) # (batch, n, 1)
    lh_quad_term = tf.matmul(tf.reshape(Y, (batch, 1, n)), iKY) # (batch, 1, 1)
    lh_quad_term = tf.reshape(lh_quad_term, [batch])

    # log P(Y|X) = -1/2 * ( n log(2 pi) + Y inv(K+noise) Y + log det(K+noise))
    gp_lhood = -0.5*( lhood_pi_term + lh_quad_term + lhood_logdet_term )

    # Compute posterior mean and variances
    Ks = tf.reshape(X, (batch, n, 1)) - tf.reshape(X_test, (batch, 1, ns)) #broadcasts to (batch, n, ns)
    Ks = tf.exp( (Ks**2) * ilt) # (batch, n, ns)
    Ks_t = tf.transpose(Ks, (0, 2, 1)) # (batch, ns, n)

    # posterior mean
    p_m = tf.matmul(Ks_t, iKY)
    p_m = tf.reshape(p_m, (batch, ns))

    # posterior variance
    iK_Ks = tf.cholesky_solve(chol_K, Ks) # (batch, n, ns)
    Ks_iK_Ks = tf.reduce_sum(Ks * iK_Ks, axis=1) # (batch, ns)
    p_v = 1 - Ks_iK_Ks # (batch, ns)
    p_v = tf.reshape(p_v, (batch, ns))

    return p_m, p_v, gp_lhood


def build_sin_and_np_elbo_graphs(vid_batch, beta, lt=5, context_ratio=0.5):
    """
    Builds both standard (sin) eblo and neural process (np) elbo. 
    Returns pretty much everything!
    Args:
        vid_batch: tf variable (batch, tmax, px, py) binay arrays or images
        beta: scalar, tf variable, annealing term for prior KL
        lt: length scale of GP
        context_ratio: float in [0,1], for np elbo, random target-context split ratio

    Returns:
        sin_elbo: "standard" elbo
        sin_elbo_recon: recon struction term
        sin_elbo_prior_kl: prior KL term
        np_elbo: neural process elbo
        np_elbo_recon: ...
        np_prior_kl: ...
        full_p_mu: approx posterior mean
        full_p_var: approx post var
        qnet_mu: recognition network mean
        qnet_var: recog. net var
        pred_vid: reconstructed video
        globals(): aaaalll variables in local scope
    """

    batch, tmax, px, py = [int(s) for s in vid_batch.get_shape()]

    # Choose a random split of target-context for each batch
    con_tf = tf.random.normal(shape=(),
                              mean=context_ratio*float(tmax),
                              stddev=np.sqrt(context_ratio*(1-context_ratio)*float(tmax)))
    con_tf = tf.math.maximum(con_tf, 2)
    con_tf = tf.math.minimum(con_tf, int(tmax)-2)
    con_tf = tf.cast(tf.round(con_tf), tf.int32)

    dt = vid_batch.dtype
    
    # recognition network terms
    qnet_mu, qnet_var = build_MLP_inference_graph(vid_batch)

    ##################################################################
    ####################### CONTEXT LIKELIHOOD #######################
    # make random indices
    ran_ind = tf.range(tmax, dtype=tf.int32)
    ran_ind = [tf.random.shuffle(ran_ind) for i in range(batch)] # (batch, tmax)
    ran_ind = [tf.reshape(r_i, (1,tmax)) for r_i in ran_ind] # len batch list( (tmax), ..., (tmax) )
    ran_ind = tf.concat(ran_ind, 0) # ()

    con_ind = ran_ind[:, :con_tf]
    tar_ind = ran_ind[:, con_tf:]

    T = tf.range(tmax, dtype=dt)
    batch_T = tf.concat([tf.reshape(T, (1,tmax)) for i in range(batch)], 0) # (batch, tmax)

    # time stamps of context points
    con_T = [tf.gather(T, con_ind[i,:]) for i in range(batch)]
    con_T = [tf.reshape(ct, (1,con_tf)) for ct in con_T]
    con_T = tf.concat(con_T, 0)

    # encoded means of contet points
    con_lm = [tf.gather(qnet_mu[i,:,:], con_ind[i,:], axis=0) for i in range(batch)]
    con_lm = [tf.reshape(cm, (1,con_tf,2)) for cm in con_lm]
    con_lm = tf.concat(con_lm, 0)

    # encoded variances of context points
    con_lv = [tf.gather(qnet_var[i,:,:], con_ind[i,:], axis=0) for i in range(batch)]
    con_lv = [tf.reshape(cv, (1,con_tf,2)) for cv in con_lv]
    con_lv = tf.concat(con_lv, 0)

    # conext Lhoods
    _,_, con_lhoodx = build_1d_gp(con_T, con_lm[:,:,0], con_lv[:,:,0], batch_T)
    _,_, con_lhoody = build_1d_gp(con_T, con_lm[:,:,1], con_lv[:,:,1], batch_T)
    con_lhood = con_lhoodx + con_lhoody


    ####################################################################################
    #################### PriorKL 1/3: FULL APPROX POST AND LIKELIHOOD ##################

    # posterior and lhood for full dataset
    p_mx, p_vx, full_lhoodx = build_1d_gp(batch_T, qnet_mu[:,:,0], qnet_var[:,:,0], batch_T)
    p_my, p_vy, full_lhoody = build_1d_gp(batch_T, qnet_mu[:,:,1], qnet_var[:,:,1], batch_T)

    full_p_mu = tf.stack([p_mx, p_my], axis=2)
    full_p_var = tf.stack([p_vx, p_vy], axis=2)

    full_lhood = full_lhoodx + full_lhoody

    ####################################################################################
    ########################### PriorKL 2/3: CROSS ENTROPY TERMS #######################

    # cross entropy term
    sin_elbo_ce = gauss_cross_entropy(full_p_mu, full_p_var, qnet_mu, qnet_var) #(batch, tmax, 2)
    sin_elbo_ce = tf.reduce_sum(sin_elbo_ce, 2) # (batch, tmax)


    np_elbo_ce = [tf.gather(sin_elbo_ce[i,:], tar_ind[i,:]) for i in range(batch)] # (batch, con_tf)
    np_elbo_ce = [tf.reduce_sum(np_i) for np_i in np_elbo_ce] # list of scalars, len=batch

    np_elbo_ce = tf.stack(np_elbo_ce) # (batch)
    sin_elbo_ce = tf.reduce_sum(sin_elbo_ce, 1) # (batch)


    ####################################################################################
    ################################ Prior KL 3/3 ######################################

    sin_elbo_prior_kl = full_lhood - sin_elbo_ce
    np_prior_kl       = full_lhood - np_elbo_ce - con_lhood


    ####################################################################################
    ########################### RECONSTRUCTION TERMS ###################################

    epsilon = tf.random.normal(shape=(batch, tmax, 2))
    latent_samples = full_p_mu + epsilon * tf.sqrt(full_p_var)
    pred_vid_batch_logits = build_MLP_decoder_graph(latent_samples, px, py)
    pred_vid = tf.nn.sigmoid(pred_vid_batch_logits)
    recon_err = tf.nn.sigmoid_cross_entropy_with_logits(labels=vid_batch, 
                                                        logits=pred_vid_batch_logits)
    sin_elbo_recon = tf.reduce_sum(-recon_err, (2,3)) # (batch, tmax)

    np_elbo_recon = [tf.gather(sin_elbo_recon[i,:], tar_ind[i,:]) for i in range(batch)] # (batch, con_tf)
    np_elbo_recon = [tf.reduce_sum(np_i) for np_i in np_elbo_recon]

    # finally the reconstruction error for each objective!
    np_elbo_recon = tf.stack(np_elbo_recon)  # (batch)
    sin_elbo_recon = tf.reduce_sum(sin_elbo_recon, 1) # (batch)



    #####################################################################################
    ####################### PUT IT ALL TOGETHER  ########################################

    sin_elbo = sin_elbo_recon + beta * sin_elbo_prior_kl
    
    np_elbo  = np_elbo_recon + beta * np_prior_kl

    return sin_elbo, sin_elbo_recon, sin_elbo_prior_kl, \
           np_elbo,   np_elbo_recon,       np_prior_kl, \
           full_p_mu, full_p_var, \
           qnet_mu, qnet_var, pred_vid, globals()
    

def run_experiment(args):

    # Make a folder to save everything
    extra = args.elbo + "_" + str(args.beta0)

    if args.modellt<0.01:
        extra += "_lt0_"
    

    chkpnt_dir = make_checkpoint_folder(args.base_dir, args.expid, extra)
    # chkpnt_dir = "/home/michael/GPVAE_checkpoints/84:__on__9_10_2019__at__19:15:31/"
    pic_folder = chkpnt_dir + "pics/"
    res_file = chkpnt_dir + "res/ELBO_pandas"
    print("\nCheckpoint Directory:\n"+str(chkpnt_dir)+"\n")


    # Data synthesis settings
    batch = 35
    tmax = 30
    px = 32
    py = 32
    r = 3
    vid_lt = 5
    model_lt = args.modellt

    # Load/ceate batches of reproducible videos
    if os.path.isfile(args.base_dir + "/Test_Batches.pkl"):
        with open(args.base_dir + "/Test_Batches.pkl", "rb") as f:
            Test_Batches = pickle.load(f)
    else:
        make_batch = lambda s: Make_Video_batch(tmax=tmax, px=px, py=py, lt=vid_lt, batch=batch, seed=s, r=r)
        Test_Batches = [make_batch(s) for s in range(10)]
        with open(args.base_dir + "/Test_Batches.pkl", "wb") as f:
            pickle.dump(Test_Batches, f)

    
    # Initialise a plots
    # this plot displays a  batch of videos + latents + reconstructions
    fig, ax = plt.subplots(4,4, figsize=(8, 8), constrained_layout=True)
    plt.ion()

    truth_c, V_c = Make_circles(); batch_V_c = np.tile(V_c, (batch,1,1,1))
    truth_sq, V_sq = Make_squares(); batch_V_sq = np.tile(V_sq, (batch,1,1,1))


    # make sure everything is created in the same graph!
    graph = tf.Graph()
    with graph.as_default():

        # Make all the graphs
        beta = tf.compat.v1.placeholder(dtype=tf.float32, shape=())
        vid_batch = build_video_batch_graph(batch=batch, tmax=tmax, px=px, py=py, r=r, lt=vid_lt)
        s_elbo, s_rec, s_pkl, np_elbo, np_rec, np_pkl, \
            p_m,p_v,q_m,q_v,pred_vid, _ = build_sin_and_np_elbo_graphs(vid_batch, beta, lt=model_lt)

        # The actual loss functions
        if args.elbo=="SIN":
            loss  = -tf.reduce_mean(s_elbo)
            e_elb = tf.reduce_mean(s_elbo)
            e_pkl = tf.reduce_mean(s_pkl)
            e_rec = tf.reduce_mean(s_rec)
        elif args.elbo=="NP":
            loss  = -tf.reduce_mean(np_elbo)
            e_elb = tf.reduce_mean(np_elbo)
            e_pkl = tf.reduce_mean(np_pkl)
            e_rec = tf.reduce_mean(np_rec)

        av_s_elbo = tf.reduce_mean(s_elbo)
        av_s_rec  = tf.reduce_mean(s_rec)
        av_s_pkl  = tf.reduce_mean(s_pkl)


        # Add optimizer ops to graph (minimizing neg elbo!), print out trainable vars
        global_step = tf.Variable(0, name='global_step',trainable=False)
        optimizer  = tf.compat.v1.train.AdamOptimizer()
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        optim_step = optimizer.minimize(loss=loss, 
                                        var_list=train_vars,
                                        global_step=global_step)
        print("\n\nTrainable variables:")
        for v in train_vars:
            print(v)

        
        # Initializer ops for the graph and saver
        init_op = tf.global_variables_initializer()
        saver = tf.compat.v1.train.Saver()


        # Results to be tracked and Pandas saver
        res_vars = [global_step,
                    loss,
                    av_s_elbo,
                    av_s_rec,
                    av_s_pkl,
                    e_elb,
                    e_rec,
                    e_pkl,
                    tf.math.reduce_min(q_v),
                    tf.math.reduce_max(q_v),
                    tf.math.reduce_min(p_v),
                    tf.math.reduce_max(p_v)]
        res_names= ["Step",
                    "Loss",
                    "Test ELBO",
                    "Test Reconstruction",
                    "Test Prior KL",
                    "Train ELBO",
                    "Train Reconstruction",
                    "Train Prior KL",
                    "min qs_var",
                    "max qs_var",
                    "min q_var",
                    "max q_var",
                    "MSE",
                    "Beta",
                    "Time"]
        res_saver = pandas_res_saver(res_file, res_names)

        # Now let's start doing some computation!
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.ram)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            # Attempt to restore weights
            try:
                saver.restore(sess, tf.train.latest_checkpoint(chkpnt_dir))
                print("\n\nRestored Model Weights")
            except:
                sess.run(init_op)
                print("\n\nInitialised Model Weights")

            # Start training that elbo!
            for t in range(args.steps):

                # Annealing factor for prior KL
                beta_t = 1 + (args.beta0-1) * np.exp(t/2000)

                # Train: do an optim step
                _, g_s = sess.run([optim_step, global_step], {beta:beta_t})
                
                # Print out diagnostics/tracking
                if g_s%10==0:
                    TD = Test_Batches[0][1]
                    test_elbo, e_rec_i, e_pkl_i = sess.run([e_elb, e_rec, e_pkl], {vid_batch:TD, beta:1.0})
                    test_qv, test_pv, test_pm, test_qm = sess.run([q_v, p_v, p_m, q_m], {vid_batch:TD, beta:1.0})

                    print(str(g_s)+": elbo "+str(test_elbo) ) #+"\t "+"\t "+str(e_rec_i)+"  "+str(e_pkl_i)+\
                    # ",\t\t qvar range:\t",str(test_pv.max()),"\t",str(test_qv.min()) ,\
                    # ",\t\t qmean range:\t",str(np.abs(test_pm).max()),"\t",str(np.abs(test_qm).max())  )


                # Save elbo, recon, priorKL....
                if g_s%100==0:
                    TT, TD = Test_Batches[0]
                    p_m_i, p_v_i = sess.run([p_m, p_v], {vid_batch:TD, beta:1})
                    _, _, MSE, _ = MSE_rotation(p_m_i, TT, p_v_i)
                    new_res = sess.run(res_vars, {vid_batch:TD, beta:1})
                    new_res += [MSE, beta_t, time.time()]
                    res_saver(new_res)


                # show plot and occasionally save
                if g_s%20==0:
                    # [[ax_ij.clear() for ax_ij in ax_i] for ax_i in ax]
                    TT, TD = Test_Batches[0]
                    reconpath, reconvar, reconvid = sess.run([p_m, p_v, pred_vid], {vid_batch:TD, beta:1})
                    rp, _, MSE, rv = MSE_rotation(reconpath, TT, reconvar)
                    _ = plot_latents(TD, TT, reconvid, rp, rv, ax=ax, nplots=4)
                    # plt.tight_layout()
                    print("Chek 7")
                    plt.draw()
                    fig.suptitle(str(g_s)+' ELBO: ' + str(test_elbo))

                    if True: #g_s%500==0:
                        plt.savefig(pic_folder + str(g_s).zfill(6)+".png")
                        # plt.close(fig)
                    
                    q_m_c = sess.run(q_m,{vid_batch: batch_V_c})
                    q_m_sq = sess.run(q_m, {vid_batch: batch_V_sq})

                    # import pdb; pdb.set_trace()

                    plot_circle(ax[3][0], ax[3][1], q_m_c)
                    plot_square(ax[3][2], ax[3][3], q_m_sq)

                    plt.show()
                    plt.pause(0.01)

                # Save NN weights
                if g_s%1000==0:
                    saver.save(sess, chkpnt_dir+"model", global_step=g_s)
                    print("\n\nModel Saved: "+ chkpnt_dir +"\n\n")



if __name__=="__main__":

    default_base_dir = os.getcwd()

    parser = argparse.ArgumentParser(description='Train GPP-VAE')
    parser.add_argument('--steps', type=int, default=50000, help='Number of steps of Adam')
    parser.add_argument('--beta0', type=float, default=1, help='initial beta annealing value')
    parser.add_argument('--elbo', type=str, choices=['SIN', 'NP'], default='SIN',
                         help='Structured Inf Nets ELBO or Neural Processes ELBO')
    parser.add_argument('--modellt', type=float, default=5, help='time scale of model to fit to data')
    parser.add_argument('--base_dir', type=str, default=default_base_dir, help='folder within a new dir is made for each run')
    parser.add_argument('--expid', type=str, default="debug", help='give this experiment a name')
    parser.add_argument('--ram', type=float, default=0.5, help='fraction of GPU ram to use')
    parser.add_argument('--seed', type=int, default=None, help='seed for rng')

    args = parser.parse_args()

    run_experiment(args)

