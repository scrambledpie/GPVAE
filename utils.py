import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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


if __name__=="__main__":
    A = Make_Video_batch()
    play_video(A)

