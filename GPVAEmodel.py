import tensorflow as tf
import numpy as np
from utils import Make_Video_batch



def build_MLP_inference_graph(vid_batch, layers=[250,250], tftype=tf.float32):
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

    # first layer, flatten images to vectors
    # vid_batch = tf.placeholder(tftype, shape=(batch, tmax, px, py))

    batch, tmax, px, py = vid_batch.get_shape()

    h0 = tf.reshape(vid_batch, (batch*tmax, px*py))
    # print(h0.get_shape())

    # loop over layers in given list
    for l in layers:
        i_dims = int(h0.get_shape()[-1])
        W = tf.Variable(tf.truncated_normal([i_dims, l],
                stddev=1.0 / np.sqrt(float(i_dims))))
        B = tf.Variable(tf.zeros([1, l]))
        h0 = tf.matmul(h0, W) + B

    # final layer just outputs x,y mean and log(var) of q network
    l = 4
    i_dims = int(h0.get_shape()[-1])
    W = tf.Variable(tf.truncated_normal([i_dims, l],
            stddev=1.0 / np.sqrt(float(i_dims))))
    B = tf.Variable(tf.zeros([1, l]))
    h0 = tf.matmul(h0, W) + B

    h0 = tf.reshape(h0, (batch, tmax, 4))
    # print(h0.get_shape())

    means = h0[:, :, :2]
    vars  = tf.exp(h0[:, :, 2:])

    return means, vars

def build_MLP_decoder_graph(latent_samples, px, py, layers=[250, 250]):
    """
    args:
        latent_samples: batch*tmax*2 (tf variable)
        px: image width (int)
        py: image height (int)
        layers: list of num. of nodes (list ints)

    returns:
        pred_batch_vid: batch*tmax*px*py (tf variable)
    """

    batch = latent_samples.get_shape()[0]
    tmax  = latent_samples.get_shape()[1]

    # flatten all frames into one matrix
    h0 = tf.reshape(latent_samples, (batch*tmax, 2))
    # print(h0.get_shape())

    # loop over layers in given list
    for l in layers:
        i_dims = int(h0.get_shape()[-1])
        W = tf.Variable(tf.truncated_normal([i_dims, l],
                stddev=1.0 / np.sqrt(float(i_dims))))
        B = tf.Variable(tf.zeros([1, l]))
        h0 = tf.matmul(h0, W) + B
        # print(h0.get_shape())

    # final layer just outputs full video batch
    l = px*py
    i_dims = int(h0.get_shape()[-1])
    W = tf.Variable(tf.truncated_normal([i_dims, l],
            stddev=1.0 / np.sqrt(float(i_dims))))
    B = tf.Variable(tf.zeros([1, l]))
    h0 = tf.matmul(h0, W) + B

    pred_vid_batch_logits = tf.reshape(h0, (batch, tmax, px, py))

    return pred_vid_batch_logits

def build_gp_lhood_and_post_graph(latent_mean, latent_var, lt=5):
    """
    args:
        latent_mean: (batch, tmax, 2) tf variable
        latent_var: (batch, tmax, 2) tf variable

    returns:
        KLprior: shape=(batch), tf variable
        post_mu: shape: (batch, tmax, 2) tf variable
        post_var: shape: (batch, tmax, 2) tf variable
    """

    batch, tmax, _ = latent_mean.get_shape()
    batch = int(batch)
    tmax = int(tmax)

    # rename and reshape latents! (batch, tmax, 1)
    lm_x = tf.reshape(latent_mean[:,:,0], (batch, tmax, 1))
    lv_x = tf.reshape(latent_var[:,:,0], (batch, tmax, 1))

    lm_y = tf.reshape(latent_mean[:,:,1], (batch, tmax, 1))
    lv_y = tf.reshape(latent_var[:,:,1], (batch, tmax, 1))

    # all kernel matrices can be computed once and stored as constants
    k_mat = np.arange(tmax)
    k_mat = np.exp((k_mat.reshape(-1,1) - k_mat.reshape(1,-1))**2*(-0.5/lt**2))
    k_mat = k_mat.reshape((1, tmax, tmax))
    k_mat = np.tile(k_mat, [batch, 1, 1])
    K = tf.constant(k_mat, dtype=tf.float32) # shape: (batch, tmax, tmax)
    eK = tf.reshape(K, (batch, tmax, 1, tmax)) # need this later for post_var

    # print("here")
    # print(K.get_shape())

    KX = K + tf.matrix_diag(lv_x[:,:,0]) # (batch, tmax, tmax)
    KY = K + tf.matrix_diag(lv_y[:,:,0])

    chol_X = tf.cholesky(KX) # (batch, tmax, tmax)
    chol_Y = tf.cholesky(KY)

    # print(tf.matrix_diag(lv_x[:,:,0]).get_shape())
    # import sys; sys.exit()

    # Compute prior_KL = marginal likelihood first
    logdet_Kx = tf.reduce_sum(tf.log(tf.matrix_diag_part(chol_X)), 1) # (batch,)
    logdet_Ky = tf.reduce_sum(tf.log(tf.matrix_diag_part(chol_Y)), 1)

    iK_x = tf.cholesky_solve(chol_X, lm_x) # (batch, tmax, 1)
    iK_y = tf.cholesky_solve(chol_Y, lm_y)

    x_iK_x = tf.matmul(lm_x, iK_x, transpose_a=True)[:, 0, 0] # (batch, 1, 1) -> (batch,)
    y_iK_y = tf.matmul(lm_y, iK_y, transpose_a=True)[:, 0, 0]

    prior_kl = -0.5*( logdet_Kx + logdet_Ky + x_iK_x + y_iK_y ) # (batch,)


    # Now compute posterior mean
    post_mu_x = tf.matmul(K, iK_x) # (batch, tmax, 1)
    post_mu_y = tf.matmul(K, iK_y)

    post_mu = tf.concat([post_mu_x, post_mu_y], 2)


    # posterior variance, we only want diagonal elements
    iK_kx = tf.cholesky_solve(chol_X, K) # (batch, tmax, tmax)
    iK_ky = tf.cholesky_solve(chol_Y, K)
    
    iK_kx = tf.reshape(iK_kx, (batch, tmax, tmax, 1)) # batch, tmax, tmax, 1
    iK_ky = tf.reshape(iK_ky, (batch, tmax, tmax, 1))

    post_var_x = 1.0 - tf.matmul(eK, iK_kx) # (batch, tmax, 1, 1)
    post_var_y = 1.0 - tf.matmul(eK, iK_ky)

    # print(post_var_x.get_shape())
    # print(post_var_y.get_shape())
    # import sys; sys.exit()

    post_var_x = tf.reshape(post_var_x, (batch, tmax, 1))
    post_var_y = tf.reshape(post_var_y, (batch, tmax, 1))

    post_var = tf.concat([post_var_x, post_var_y], 2)

    # print(post_var.get_shape())
    # import sys; sys.exit()

    return prior_kl, post_mu, post_var

def build_ELBO_graph(batch, tmax, px, py, beta):
    """
    Makes a tf graph of elbo
    args:
        batch: batch size (int)
        tmax: video length (int)
        px: pixel width (int)
        py: pixel height (int)
        beta: tf variable, shape=(1) or ()
    
    returns:
        peior_KL: tf variable (batch)
        recon_err: tf variabel (batch)
        elbo: tf variable (batch)
    """

    vid_batch = tf.placeholder(shape=(batch, tmax, px, py), dtype=tf.float32)

    q_mean, q_var = build_MLP_inference_graph(vid_batch)

    prior_kl, post_mean, post_var = build_gp_lhood_and_post_graph(q_mean, q_var)

    epsilon = tf.random.normal(shape=(batch, tmax, 2))

    latent_samples = post_mean + epsilon * tf.sqrt(post_var)

    pred_vid_batch_logits = build_MLP_decoder_graph(latent_samples, px, py)

    recon_err = tf.nn.sigmoid_cross_entropy_with_logits(labels=vid_batch, 
                                                        logits=pred_vid_batch_logits)
    recon_err = tf.reduce_sum(recon_err, (1,2,3))

    elbo = recon_err + beta*prior_kl

    return vid_batch, prior_kl, recon_err, elbo, q_mean, q_var, post_mean, post_var


if __name__=="__main__":

    batch = 45
    tmax = 30
    px = 32
    py = 32

    beta = tf.placeholder(dtype=tf.float32, shape=())

    vb, p_kl, recon, elbo, q_m, q_v, p_m, p_v = build_ELBO_graph(batch, tmax, px, py, beta)

    print(p_kl.get_shape())
    print(recon.get_shape())
    print(elbo.get_shape())

    Data = Make_Video_batch(tmax=tmax, px=px, py=py, lt=5, batch=batch)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        print("elbo is")
        print(sess.run(recon, {vb:Data, beta:1.0}))