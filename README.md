# Applied Machine Learning Project
Final project of Applied Machine Learning.

## Description
In this project, we implemented two types of Generative Adversarial Networks (GANs) to generate images of human faces from random noise. One, which acts as our baseline model, is a Deep Convolution GAN (DCGAN). The second one is Wasserstein GAN (WGAN). For this task, we used the dataset CelebA. 

For the evaluation of these models, we used Fréchet Inception Distance (FID), comparing a smaller portion of the original dataset (20k images saved in FID_images) to the generated images by each one of these models.  

### How to run FID
We decided not to put the datasets for evaluation on GitHub. To evaluate our models locally, you will first need to generate an image dataset. This can be done by running the file `FID_WGAN_pictures_generation.py` for the generation of random images and `FID_random_pictures_generation.py` for the generation of images through WGAN-GP. Then the FID can be done using commands:
```
pip install pytorch-fid
python -m pytorch_fid path/to/real_faces path/to/generated_faces
```

### Run API locally
`Python 3.12` is recommended but it will most likely work on other versions(idk haven't tested)
To run the API, first download the repository as a zip file or use 
```
git clone https://github.com/UrWoo/applied-ml-project.git
```
In the repository folder install dependencies using
```
python3 -m pip install -r requirements.txt
```
Then run the api
```
fastapi run API/wgan_api.py
```

to generate an image, send a GET request to `/image`

### Run streamlit locally
To run the streamlit application, you need to run the following command:
```
streamlit run API/GANs_project.py
```
(presuming you already installed `requirements.txt`)

### Train the models
If you want to train the models on your own data, place the dataset into data/*name of your dataset*/images and decide which model to train.

In models/*model_name*/*model_name*_training,py replace `data_root` variable with the absolute path to your dataset(without /images), and change necessary hyperparameters, mainly `image_channels` to 1 if using grayscale images, and `epochs` to the number of epochs to train your model.

Then just install dependencies and run the model using
```
python3 -m pip install -r requirements.txt
python3 models/*model_name*/*model_name*_training,py
```
