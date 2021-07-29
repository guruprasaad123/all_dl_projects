[TOC]

## Installation

- install the required packages/libraries using this command
  - `pip3 install -r requirements.txt`
  - or
  - `conda install --file requirements.txt`
- i have used `python-3.7` for this project

## Intialize dataset

- unzip the **training.zip** file in the root directory ( `/training` )
- we are creating positive and negative pairs of matches in  `utils.py` script
- To load all the image pairs instead of loading the entire dataset to variables prior to training , we are creating a dataset that has all the pairs of data in `/post_training/dataset.h5` file
- we can create this using this command 
  - `python3 utils.py`

## Training Model

- we can start the training using this command
  - `python3 train.py`
- so we are storing the model's state on every epoch `checkpoint/siemese_network_v2`
- and the we are using tensorboard for visualizing our model's performance in `runs/siemese_network_v2`
  - make sure you have installed **tensorboard**
  - run this command `tensorboard --logdir=runs/`

### Notes

- it took 8 hours for me to train the model with my system requirements listed below
  - i7 processor
  - GTX 1050 Graphics Card
  - 8GB Ram

## Testing Model

- Download the trained model files from [here](https://drive.google.com/file/d/1d5HNXDJOT40OnpiLy9Yk9V0Zn7Yf_WJa/view?usp=sharing) and place it in `checkpoint/siemese_network_v2/`

- we can run the `test.py` with our desired input changes

- ```python
  test_obj = {
  'src' :  'tests/jennifer-aniston.jpg',
  'dest' : 'tests/tom-cruise.jpg'
  }
  ```

- images are placed in `/tests` folder


# References

https://stackoverflow.com/questions/13211745/detect-face-then-autocrop-pictures

https://medium.com/@ctrlfizz/automatically-generate-passport-size-pictures-from-a-group-photo-3f9aa1d73f65

[Automatically detects and crops faces from batches of pictures]( https://github.com/leblancfg/autocrop )

