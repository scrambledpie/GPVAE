import tensorflow as tf
from tensorflow.python.ops import math_ops as tfmath_ops
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime as dt
import sys
from matplotlib.patches import Ellipse
import shutil
import pandas as pd
import pickle
import time
import subprocess as sp

def Make_path_batch(
    batch=40,
    tmax=30,
    lt=5,
    seed=None
    ):
    """
    Samples x(t), y(t) from a GP
    args:
        batch: number of samples
        tmax: length of samples
        lt: GP length scale
    returns:
        traj: nparray (batch, tmax, 2)
    """

    ilt = -0.5/(lt*lt)
    T = np.arange(tmax)

    Sigma = np.exp( ilt * (T.reshape(-1,1) - T.reshape(1,-1))**2)
    Mu = np.zeros(tmax)

    np.random.seed(seed)

    traj = np.random.multivariate_normal(Mu, Sigma, (batch, 2))
    traj = np.transpose(traj, (0,2,1))

    return traj


def Make_Video_batch(tmax=50,
    px=32,
    py=32,
    lt=5,
    batch=40,
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
        traj0: (batch, tmax, 2) numpy array
        vid_batch: (batch, tmax, px, py) numpy array
    """


    traj0 = Make_path_batch(batch=batch, tmax=tmax, lt=lt)

    traj = traj0.copy()

    # convert trajectories to pixel dims
    traj[:,:,0] = traj[:,:,0] * (px/5) + (0.5*px)
    traj[:,:,1] = traj[:,:,1] * (py/5) + (0.5*py)

    rr = r*r

    def pixelate_frame(xy):
        """
        takes a single x,y pixel point and converts to binary image
        with ball centered at x,y.
        """
        x = xy[0]
        y = xy[1]

        sq_x = (np.arange(px) - x)**2
        sq_y = (np.arange(py) - y)**2

        sq = sq_x.reshape(1,-1) + sq_y.reshape(-1,1)

        image = 1*(sq < rr)

        return image

    
    def pixelate_series(XY):
        vid = map(pixelate_frame, XY)
        vid = [v for v in vid]
        return np.asarray(vid)


    vid_batch = [pixelate_series(traj_i) for traj_i in traj]

    vid_batch = np.asarray(vid_batch)

    return traj0, vid_batch

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

    assert px==py, "video batch graph assumes square frames"

    rr = r*r

    ilt = tf.constant(-0.5/(lt**2), dtype=dtype)

    K = tf.range(tmax, dtype=dtype)
    K = (tf.reshape(K, (tmax, 1)) - tf.reshape(K, (1, tmax)))**2

    # print((K*ilt).get_shape())
    # sys.exit()


    K = tf.exp(K*ilt) + 0.00001*tf.eye(tmax, dtype=dtype)
    chol_K = tf.Variable(tf.linalg.cholesky(K), trainable=False)

    ran_Z = tf.random.normal((tmax, 2*batch))

    paths = tf.matmul(chol_K, ran_Z)
    paths = tf.reshape(paths, (tmax, batch, 2))
    paths = tf.transpose(paths, (1,0,2))

    # assumes px = py
    paths = paths*0.2*px + 0.5*px
    # paths[:,:,0] = paths[:,:,0]*0.2*px + 0.5*px
    # paths[:,:,1] = paths[:,:,1]*0.2*py + 0.5*py

    vid_batch = []

    tf_px = tf.range(px, dtype=dtype)
    tf_py = tf.range(py, dtype=dtype)

    for b in range(batch):
        frames_tmax = []
        for t in range(tmax):
            lx = tf.reshape((tf_px - paths[b,t,0])**2, (px, 1))
            ly = tf.reshape((tf_py - paths[b,t,1])**2, (1, py))
            frame = lx + ly < rr
            frames_tmax.append(tf.reshape(frame, (1,1,px,py)))
        vid_batch.append(tf.concat(frames_tmax, 1))
        

    vid_batch = [tfmath_ops.cast(vid, dtype=dtype) for vid in vid_batch]
    vid_batch = tf.concat(vid_batch, 0)

    return vid_batch


def MSE_rotation(X, Y, VX=None):
    """
    Given X, rotate it onto Y
    args:
        X: np array (batch, tmax, 2)
        Y: np array (batch, tmax, 2)
        VX: variance of X values (batch, tmax, 2)

    returns:
        X_rot: rotated X (batch, tmax, 2)
        W: nparray (2, 2)
        B: nparray (2, 1)
        MSE: ||X_rot - Y||^2
        VX_rot: rotated cov matrices (default zeros)
    """

    batch, tmax, _ = X.shape

    X = X.reshape((batch*tmax, 2))


    X = np.hstack([X, np.ones((batch*tmax, 1))])

    
    Y = Y.reshape(batch*tmax, 2)

    W, MSE, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    try:
        MSE = MSE[0] + MSE[1]
    except:
        MSE = np.nan

    X_rot = np.matmul(X, W)
    X_rot = X_rot.reshape(batch, tmax, 2)


    VX_rot = np.zeros((batch, tmax, 2, 2))
    if VX is not None:
        W_rot = W[:2,:]
        W_rot_t = np.transpose(W[:2,:])
        for b in range(batch):
            for t in range(tmax):
                VX_i = np.diag(VX[b,t,:])
                VX_i = np.matmul(W_rot, VX_i)
                VX_i = np.matmul(VX_i, W_rot_t)
                VX_rot[b,t,:,:] = VX_i

    return X_rot, W, MSE, VX_rot


def plot_latents(truevids, 
                 truepath, 
                 reconvids=None, 
                 reconpath=None, 
                 reconvar=None, 
                 ax=None, 
                 nplots=4, 
                 paths=None):
    """
    Plots an array of input videos and reconstructions.
    args:
        truevids: (batch, tmax, px, py) np array of videos
        truepath: (batch, tmax, 2) np array of latent positions
        reconvids: (batch, tmax, px, py) np array of videos
        reconpath: (batch, tmax, 2) np array of latent positions
        reconvar: (batch, tmax, 2, 2) np array, cov mat 
        ax: (optional) list of lists of axes objects for plotting
        nplots: int, number of rows of plot, each row is one video
        paths: (batch, tmax, 2) np array optional extra array to plot

    returns:
        fig: figure object with all plots

    """

    if ax is None:
        _, ax = plt.subplots(nplots,3, figsize=(6, 8))
    
    for axi in ax:
        for axj in axi:
            axj.clear()

    _, tmax, _, _ = truevids.shape


    # get axis limits for the latent space
    xmin = np.min([truepath[:nplots,:,0].min(), 
                   reconpath[:nplots,:,0].min()]) -0.1
    xmin = np.min([xmin, -2.5])
    xmax = np.max([truepath[:nplots,:,0].max(), 
                   reconpath[:nplots,:,0].max()]) +0.1
    xmax = np.max([xmax, 2.5])

    ymin = np.min([truepath[:nplots,:,1].min(), 
                   reconpath[:nplots,:,1].min()]) -0.1
    ymin = np.min([ymin, -2.5])
    ymax = np.max([truepath[:nplots,:,1].max(), 
                   reconpath[:nplots,:,1].max()]) +0.1
    ymax = np.max([xmax, 2.5])

    def make_heatmap(vid):
        """
        args:
            vid: tmax, px, py
        returns:
            flat_vid: px, py
        """
        vid = np.array([(t+4)*v for t,v in enumerate(vid)])
        flat_vid = np.max(vid, 0)*(1/(4+tmax))
        return flat_vid
    
    if reconvar is not None:
        E = np.linalg.eig(reconvar[:nplots,:,:,:])
        H = np.sqrt(E[0][:,:,0])
        W = np.sqrt(E[0][:,:,1])
        A = np.arctan2(E[1][:,:,0,1], E[1][:,:,0,0])*(360/(2*np.pi))

    def plot_set(i):
        # i is batch element = plot column

        # top row of plots is true data heatmap
        tv = make_heatmap(truevids[i,:,:,:])
        ax[0][i].imshow(1-tv, origin='lower', cmap='Greys')
        ax[0][i].axis('off')


        # middle row is trajectories
        ax[1][i].plot(truepath[i,:,0], truepath[i,:,1])
        ax[1][i].set_xlim([xmin, xmax])
        ax[1][i].set_ylim([ymin, ymax])
        ax[1][i].scatter(truepath[i,-1,0], truepath[i,-1,1])


        if reconpath is not None:
            ax[1][i].plot(reconpath[i,:,0], reconpath[i,:,1])
            ax[1][i].scatter(reconpath[i,-1,0], reconpath[i,-1,1])

        if paths is not None:
            ax[1][i].plot(paths[i,:,0], paths[i,:,1])
            ax[1][i].scatter(paths[i,-1,0], paths[i,-1,1])
        
        if reconvar is not None:
            ells = [Ellipse(xy=reconpath[i,t,:], 
                            width=W[i,t], 
                            height=H[i,t], 
                            angle=A[i,t]) for t in range(tmax)]
            for e in ells:
                ax[1][i].add_artist(e)
                e.set_clip_box(ax[1][i].bbox)
                e.set_alpha(0.25)
                e.set_facecolor('C1')

        # Third row is reconstructed video
        if reconvids is not None:
            rv = make_heatmap(reconvids[i,:,:,:])
            ax[2][i].imshow(1-rv, origin='lower', cmap='Greys')
            ax[2][i].axis('off')
    
    
    for i in range(nplots):
        plot_set(i)
    
    return ax
    

def make_checkpoint_folder(base_dir, expid=None, extra=""):
    """
    Makes a folder and sub folders for pics and results
    Args:
        base_dir: the root directory where new folder will be made
        expid: optional extra sub dir inside base_dir
    """

    # make a "root" dir to store all checkpoints
    # homedir = os.getenv("HOME")
    # base_dir = homedir+"/GPVAE_checkpoints/"

    if expid is not None:
        base_dir = base_dir + "/" + expid + "/"

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # now make a unique folder inside the root for this experiments
    filenum = str(len(os.listdir(base_dir))) + ":"+extra+"__on__"

    T = dt.now()

    filetime = str(T.day)+"_"+str(T.month)+"_"+str(T.year) + "__at__"
    filetime += str(T.hour)+":"+str(T.minute)+":"+str(T.second)

    # main folder
    checkpoint_folder = base_dir + filenum + filetime
    os.makedirs(checkpoint_folder)

    # pictures folder
    pic_folder = checkpoint_folder + "/pics/"
    os.makedirs(pic_folder)

    # pickled results files
    res_folder = checkpoint_folder + "/res/"
    os.makedirs(res_folder)

    # source code
    src_folder = checkpoint_folder + "/sourcecode/"
    os.makedirs(src_folder)
    old_src_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    src_files = os.listdir(old_src_dir)
    print("\n\nCopying source Code to "+src_folder)
    for f in src_files:
        if ".py" in f:
            src_file = old_src_dir + f
            shutil.copy2(src_file, src_folder)
            print(src_file)
    print("\n")

    
    return checkpoint_folder + "/"


class pandas_res_saver:
    """
    Takes a file and a list of col names to initialise a
    pandas array. Then accepts extra rows to be added
    and occasionally written to disc.
    """
    def __init__(self, res_file, colnames):
        # reload old results frame
        if os.path.exists(res_file):
            if list(pd.read_pickle(res_file).columns)==colnames:
                print("res_file: recovered ")
                self.data = pd.read_pickle(res_file)
                self.res_file = res_file
            else:
                print("res_file: old exists but not same, making new ")
                self.res_file = res_file + "_" + str(time.time())
                self.data = pd.DataFrame(columns=colnames)
        else:
            print("res_file: new")
            self.res_file = res_file
            self.data = pd.DataFrame(columns=colnames)
            
        self.ncols = len(colnames)
        self.colnames = colnames
    
    def __call__(self, new_data):
        new_data = np.asarray(new_data).reshape((-1, self.ncols))
        new_data = pd.DataFrame(new_data, columns=self.colnames)
        self.data = pd.concat([self.data, new_data])

        if  self.data.shape[0]%10 == 1:
            self.data.to_pickle(self.res_file)
            print("Saved results to file: "+self.res_file)
    

def call_bash(cmd):
    process = sp.Popen(cmd.split(), stdout=sp.PIPE)
    output, error = process.communicate()


def dict_to_flags(mydict):
    cmd = ""
    for k, v in mydict.items():
        cmd += " --" + k + " " + str(v)
    return cmd



if __name__=="__main__":

    traj, vid_batch = Make_Video_batch()

    new_traj = 2*np.flip(traj, 2)+2

    batch, tmax, _ = traj.shape

    new_traj_var = np.eye(2).reshape((1,1,2,2))
    new_traj_var = np.tile(new_traj_var, (batch, tmax, 1, 1))

    
    new_traj_rot, W, MSE, new_traj_var_rot = MSE_rotation(new_traj, 
                                                          traj, 
                                                          new_traj_var)



    # print(traj)
    ax  = plot_latents(truevids=vid_batch,
                       truepath=traj,
                       reconpath=new_traj_rot,
                       reconvids=vid_batch,
                       reconvar=new_traj_var,
                       nplots=4)
    plt.tight_layout()
    plt.show()
    fig = plt.gcf()
