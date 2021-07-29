# Tasks

## Build Convolutional Neural Networks

- Build a cnn model that classifies 43 different labels

### Split the dataset

- split the dataset into 3 splits
  - training - 80% from the original dataset
  - testing - 20% from the original dataset
  - validation - 20% from the training split

### Not use Pre-trained Model

- as per the task created VGG-11 Model which is trained from scratch

### Minimal Complexity 

- Have implemented VGG-11 ( slightly modified )
- used these as mentioned 
  - Batch Normalisation
  - Dropout
  - Max/Avg Pooling as mentioned in task 2

### Fixed input size

- have used 28 x 28 as a fixed input size as its standard practice
- all the images are resized to 28 x 28

### Model Overview

- ```
  VGG11(
    (features): Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
      (1): SEBlock(
        (fc): Sequential(
          (0): Conv2d(64, 4, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): ReLU()
          (2): Conv2d(4, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
      )
      (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): ReLU(inplace=True)
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (6): SEBlock(
        (fc): Sequential(
          (0): Conv2d(128, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): ReLU()
          (2): Conv2d(8, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
      )
      (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (8): ReLU(inplace=True)
      (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (11): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (12): ReLU(inplace=True)
      (13): Dropout(p=0.4, inplace=False)
      (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (15): SEBlock(
        (fc): Sequential(
          (0): Conv2d(256, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): ReLU()
          (2): Conv2d(16, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
      )
      (16): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (17): ReLU(inplace=True)
      (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (20): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (21): ReLU(inplace=True)
      (22): Dropout(p=0.4, inplace=False)
      (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (24): SEBlock(
        (fc): Sequential(
          (0): Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): ReLU()
          (2): Conv2d(32, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
      )
      (25): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (26): ReLU(inplace=True)
      (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (29): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (30): ReLU(inplace=True)
      (31): Dropout(p=0.4, inplace=False)
      (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (33): SEBlock(
        (fc): Sequential(
          (0): Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): ReLU()
          (2): Conv2d(32, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
      )
      (34): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (35): ReLU(inplace=True)
      (36): MaxPool2d(kernel_size=1, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (classifier): Sequential(
      (0): Linear(in_features=512, out_features=512, bias=False)
      (1): Dropout(p=0.5, inplace=False)
      (2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): ReLU(inplace=True)
      (4): Dropout(p=0.5, inplace=False)
      (5): Linear(in_features=512, out_features=43, bias=True)
    )
  )
  ```

  

### Loading images Prior training ( Bonus)

- `Final_training/images` folder has the contents of the images . labels
- `init.py`  script placed in the same script will load the images and its respective labels onto `training.h5` file
- this binary file `training.h5` we will use in data loader prior training the model

## SE Block

- as read in this [paper](https://arxiv.org/pdf/1709.01507.pdf)  , added SE block to the already existing VGG architecture

### Placement of SE Block

- As it was suggested to this paper that squeeze and excitation can be used post convolutional layers , and it can be used on the bottom most layer where higher number of inputs are required
- so i have added SE block to every higher Conv layer and that too at the end of each combination

### Performance and Metrics

- I have run the same model without the implementation of SE block and it took so much time , computational power to gain accuracy , lesser loss 
- But after the implementation of SE block , the model gained steady increase in accuracy , decrease in loss in fewer steps/epochs which is amazing

### Test Model

- Created an interface script that loads the model and enables it to eval mode that is that can be used to testing images

### Model Interpretation

- For model interpretation , i have used tensorboard 
  - for plotting the accuracy on every step/epoch
  - for plotting the loss on every step/epoch
  - for plotting the f1-score on every step/epoch
  - for plotting the precision-score on every step/epoch
  - for plotting the  recall-score on every step/epoch
  - and also plotting the histogram of convolutional layer's weights on every step/epoch
  - plotting the histogram of convolutional layer's grad weights on every step/epoch



## Acknowledgement

https://benchmark.ini.rub.de/