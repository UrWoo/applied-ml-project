# Applied Machine Learning Project
Final project of Applied Machine Learning.

## Description
In this project, we implemented two types of Generative Adversarial Networks (GAN) to generate images of human faces from random noise. One, which acts as our baseline model is Deep Convolution GAN (DCGAN). The second one is Wasserstein GAN (WGAN). For this task we used dataset CelebA. 

For evaluation of these models we used Fréchet Inception Distance (FID), comparing smaller portion of the original dataset (20k images saved in FID_images) to the generated images by each one of these models.  

### Description of Models

### How to run FID
We decided not to put datasets for evaluation to Github. To make evaluation of our models locally, you will first need to generate a image datasets. This can be done by running files `FID_WGAN_pictures_generation.py` for generation of random images and `FID_random_pictures_generation.py` for generation of images through WGAN-GP. Then the FID can be done using commands:
```
```
`pip install pytorch-fid` 
`python -m pytorch_fid path/to/real_faces path/to/generated_faces`

```
```

## TODO

- Useful readme.md
- Make an API
