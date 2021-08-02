
## kaggle

https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

## Imbalanced Dataset

- [Imbalanced Data : How to handle Imbalanced Classification Problems](https://www.analyticsvidhya.com/blog/2017/03/imbalanced-data-classification/)

- [How to Deal with Imbalanced Data](https://towardsdatascience.com/how-to-deal-with-imbalanced-data-34ab7db9b100)


## Strategy

- As we visualized the datasets we got to know that dataset is imbalanced ( not having equal examples on each classes )

- To address this issue , we are doing over-sampling using [WeightedRandomSampler](https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#WeightedRandomSampler) so that the dataset will be balanced while training , Testing , Validation


## Model

- We are using the `inception-v3` model , we got above `> 85 %` on test , validation sets

```python
train_utils - INFO - Training Mean Loss on Epoch:29 = 0.07355931910072885
train_utils - INFO - train-acc : 97.3433% train-loss : 0.07356
train_utils - INFO - elapsed time: 678s
train_utils - INFO - Mean Testing Loss on Epoch:29 = 0.379193115234375
train_utils - INFO - test-acc : 85.2564% test-loss : 0.37919
train_utils - INFO - elapsed time: 83s
train_utils - INFO - Mean validation Loss on Epoch:29 = 0.032135009765625
train_utils - INFO - val-acc : 100.0000% val-loss : 0.03214
train_utils - INFO - elapsed time: 3s
```



## Deep Learning | Heathcare

- [Deep Learning in Healthcare — X-Ray Imaging (Part 1 — Basics of X-Rays)](https://towardsdatascience.com/deep-learning-in-healthcare-x-ray-imaging-part-1-basics-of-x-rays-f8e6bad1e421?source=user_profile---------17----------------------------)

- [Deep Learning in Healthcare — X-Ray Imaging (Part 2— Understanding X-Ray Images)](https://towardsdatascience.com/deep-learning-in-healthcare-x-ray-imaging-part-2-understanding-x-ray-images-b8c6155cd51d?source=user_profile---------16----------------------------)

- [Deep Learning in Healthcare — X-Ray Imaging (Part 3-Analyzing images using Python)](https://towardsdatascience.com/deep-learning-in-healthcare-x-ray-imaging-part-3-analyzing-images-using-python-915a98fbf14c)


- [Deep Learning in Healthcare — X-Ray Imaging (Part 4-The Class Imbalance problem)](https://towardsdatascience.com/deep-learning-in-healthcare-x-ray-imaging-part-4-the-class-imbalance-problem-364eff4d47bb?source=user_profile---------14----------------------------)

- [Deep Learning in Healthcare — X-Ray Imaging (Part 5-Data Augmentation and Image Normalization)](https://towardsdatascience.com/deep-learning-in-healthcare-x-ray-imaging-part-5-data-augmentation-and-image-normalization-1ead1c02cfe3)

    - [1000x Faster Data Augmentation | BAIR](https://bair.berkeley.edu/blog/2019/06/07/data_aug/#:~:text=Data%20augmentation%20is%20a%20strategy,to%20train%20large%20neural%20networks.)


## Reference

- [Histogram | matplotlib](https://matplotlib.org/stable/gallery/lines_bars_and_markers/categorical_variables.html)
