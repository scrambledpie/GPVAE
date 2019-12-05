# Gaussian Process Prior VAE for Latent Dynamics from Pixels

Watch the GP-VAE training and learning to untangle the latent space

[![Training GP-VAE](https://img.youtube.com/vi/riVhb6K_iMo/0.jpg)](https://www.youtube.com/watch?v=riVhb6K_iMo)


Code to reproduce plots and experiments in the paper "Gaussian Process Prior VAE for Latent Dynamics from Pixels" presented at 2nd Symposioum on Advances in Approximate Bayesian Inference 2019.

Written in Tensorflow 1.13, python 3.6.

Get the repo:
```git clone https://github.com/scrambledpie/GPVAE.git```

Then run the main file:
```
cd GPVAE 
python GPVAEmodel.py
```

This will train the GP-VAE from scratch using the standard ELBO derived for this problem (code also support Neural Process ELBO). All output (checkpoints, plots, training metrics, source code) is stored in a new subdirectory ```GPVAE/debug/(unique run name)```.

If you want to make a video out of the training images, navigate to the pics folder and type
```
ffmpeg -pattern_type glob -i "*.png" -r 5 -pix_fmt yuv420p training.mp4
```


