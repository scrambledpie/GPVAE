import tensorflow as tf
from tensorflow.python.ops import math_ops as tfmath_ops
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime as dt

def Make_Video_batch(tmax=50,
    px=32,
    py=32,
    lt=5,
    batch=1,
    seed=1,
    r=3
    ):
    """
    params:
        tmax: number of frames to generate
        px: horizontal resolution
        py: vertical resolution
        lt: length scale
        batch: number of videos
        seed: rng seed
        r: radius of ball in pixels
    
    returns:
        vid_batch: batch*tmax*px*py numpy array
    """

    ilt = -0.5/(lt*lt)
    T = np.arange(tmax)
    rr = r*r

    Sigma = np.exp( ilt * (T.reshape(-1,1) - T.reshape(1,-1))**2)
    Mu = np.zeros(tmax)

    np.random.seed(seed)

    traj = np.random.multivariate_normal(Mu, Sigma, (batch, 2))
    traj = np.transpose(traj, (0,2,1))


    # convert trajectories to pixel dims
    traj[:,:,0] = traj[:,:,0] * (px/5) + (0.5*px)
    traj[:,:,1] = traj[:,:,1] * (py/5) + (0.5*py)

    def pixelate_frame(xy):
        """
        takes a single x,y pixel point and converts to binary image with ball
        centered at x,y.
        """
        x = xy[0]
        y = xy[1]

        sq_x = (np.arange(px) - x)**2
        sq_y = (np.arange(py) - y)**2

        sq = sq_x.reshape(-1,1) + sq_y.reshape(1,-1)

        image = 1*(sq < rr)

        return image

    
    def pixelate_series(XY):
        vid = map(pixelate_frame, XY)
        vid = [v for v in vid]
        return np.asarray(vid)


    vid_batch = [pixelate_series(traj_i) for traj_i in traj]

    vid_batch = np.asarray(vid_batch)

    # print(vid_batch.shape)
    # import sys; sys.exit()

    return vid_batch

def play_video(vid_batch, j=0):
    """
    vid_batch: batch*tmax*px*py batch of videos
    j: int, which elem of batch to play
    """

    _, ax = plt.subplots(figsize=(5,5))
    plt.ion()

    for i in range(vid_batch.shape[1]):
        ax.clear()
        ax.imshow(vid_batch[j,i,:,:])
        plt.pause(0.1)

def build_video_batch_graph(tmax=50,
    px=32,
    py=32,
    lt=5,
    batch=1,
    seed=1,
    r=3,
    dtype=tf.float32):

    rr = r*r

    K = tf.range(tmax)
    K = (K.rehspape((tmax, 1)) - K.rehspape((1, tmax)))**2
    K = tf.exp(K*(-0.5/lt**2))
    chol_K = tf.cholesky(K)

    ran_Z = tf.random.normal((2*tmax, batch))

    paths = tf.matmul(chol_K, ran_Z)
    paths = tf.reshape(paths, (2, tmax, batch))
    paths = tf.transpose(paths, (2,1,0))

    paths[:,:,0] = paths[:,:,0]*0.2*px + 0.5*px
    paths[:,:,1] = paths{:,:,1]*0.2*py + 0.5*py

    vid_batch = tf.zeros((batch, tmax, px, py))

    for b in range(batch):
        for t in range(tmax):
            lx = tf.reshape((tf.range(px) - paths[b,t,0])**2, (px, 1))
            ly = tf.reshape((tf.range(py) - paths[b,t,1])**2, (1, py))
            vid_batch[b,t,:,:] = lx + ly < rr

    vid_batch = tfmath_ops.cast(vid_batch, dtype=dtype)

    return vid_batch




def make_checkpoint_folder(base_dir=None):

    # make a "root" dir to store all checkpoints
    if base_dir is None:
        homedir = os.getenv("HOME")
        base_dir = homedir+"/GPVAE_checkpoints/"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
    
    # now make a unique folder inside the root for this experiments
    filenum = str(len(os.listdir(base_dir))) + ":__on__"

    T = dt.now()

    filetime = str(T.day)+"_"+str(T.month)+"_"+str(T.year) + "__at__"
    filetime += str(T.hour)+":"+str(T.minute)+":"+str(T.second)

    checkpoint_folder = base_dir + filenum + filetime

    if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
    
    return checkpoint_folder + "/"


if __name__=="__main__":
    A = Make_Video_batch()
    play_video(A)

    # print(make_checkpoint_folder())

