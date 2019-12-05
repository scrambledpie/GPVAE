# GPVAE

Code to reproduce plots and experiments in the paper "Gaussian Process Prior VAE for Latent Dynamics from Pixels".

This was written in Tensorflow 1.13.

Get the repo:
```git clone https://github.com/scrambledpie/GPVAE.git```

Then run the main file:
```cd GPVAE 
python GPVAEmodel.py```

This will train the GP-VAE from scratch using the standard ELBO. All output (checkpoints, plots, training metrics, source code) is stored in a new subdirectory ```debug/(unique run name)```.

If you want to make a video out of the training images, navigate to the pics folder and type
```
ffmpeg -pattern_type glob -i "*.png" -r 5 -pix_fmt yuv420p training.mp4

```


