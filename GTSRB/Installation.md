# Code

## Installation

#### Using pip

- Install the python dependencies using this command below
- `pip3 install -r requirements.txt`

#### using Miniconda / Conda

- Install the python dependencies using the command below
- `conda install --file requirements.txt`

## Loading the Dataset

- Download **GTSRB_Final_Training_Images.zip** and extract this zip in the same folder
- Run `python3 init.py` which loads all the images , labels onto a binary file `training.h5`
- which we will load into our data-loader which we will use for training our model

## Training the Model

- Run `python3 train_vgg.py`  which start training our model
- we are saving our model on every epoch in `/checkpoint/GTSRB_VGG_SE_11` folder
- Even when the training is interrupted in the middle , we can run the same script which starts from where it left off

## Visualization / Interpretation of Model

- we are using tensorboard for visualization of model performance
- Run `tensorboard --logdir=runs` which opens a web browser where we can 
  - check model's overall architectures
  - performance stepwise
  - sample images

