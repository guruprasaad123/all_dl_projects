import os
import time

import torch
import torch.utils.data
import torchvision
from torchvision import transforms
from model import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORK_DIR = './data'
EPOCHS = 100
BATCH_SIZE = 32
#LEARNING_RATE = 0.01

MODEL_PATH = './model'
MODEL_NAME = 'Inception_v3.pth'

from utils import *

# Create model
if not os.path.exists(MODEL_PATH):
  os.makedirs(MODEL_PATH)

CHECKPOINT_PATH = os.path.join( 'checkpoint' , 'cifar_10' )

#AUGMENTATIONS
transform = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  #torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.5, hue=0.1),
  transforms.RandomHorizontalFlip(),
  torchvision.transforms.RandomVerticalFlip(),
  # torchvision.transforms.RandomAffine(degrees=0, translate=(0.2,0.2), scale=None,shear=50, resample=False, fillcolor=0),
  torchvision.transforms.RandomRotation((20), resample=False,expand=False, center=None),
  transforms.ToTensor(),
  transforms.Normalize([0.4913997551666284, 0.48215855929893703, 0.4465309133731618], [0.24703225141799082, 0.24348516474564, 0.26158783926049628])
])

# Load data
dataset = torchvision.datasets.CIFAR10(root=WORK_DIR,
                                        download=True,
                                        train=True,
                                        transform=transform)

inputs , labels = dataset[0]
print( 'dataset : ' ,inputs.shape)

dataset_loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True)

# Total parameters
model = inception_v3().to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_total_params


def main():
  print(f"Train numbers:{len(dataset)}")

  model_state = None
  opt_state = None
  last_epoch = None
  loss = None

  # if the checkpoint path does not exists , then create it
  if not os.path.exists( CHECKPOINT_PATH ):
      os.makedirs( CHECKPOINT_PATH )
  # if checkpoint path already exists
  else:
      dirs = os.listdir( CHECKPOINT_PATH )

      if len(dirs) > 0:

          latest_checkpoint = max(dirs)

          checkpoint = torch.load( os.path.join(CHECKPOINT_PATH,latest_checkpoint) )

          model_state = checkpoint['model_state_dict']
          opt_state = checkpoint['optimizer_state_dict']
          last_epoch = checkpoint['epoch']
          loss = checkpoint['loss']

          print(' [*] Model Restored from {} Epoch \n'.format(last_epoch) )

          model_restored = True

  LEARNING_RATE = 0.001
  MOMENTUM=0.9
  # first train run this line
  #model = inception_v3().to(device)
  print(model)

  if model_state:
    model.load_state_dict(model_state)

  #model_save_name = 'Inception_v3e1.pth'
  # model.load_state_dict(torch.load(MODEL_NAME))
  # Load model
  #if device == 'cuda':

    #model = torch.load(MODEL_PATH + MODEL_NAME).to(device)
  #else:
    #model = torch.load(MODEL_PATH + MODEL_NAME, map_location='cpu')
  # cast
  cast = torch.nn.CrossEntropyLoss().to(device)
  # Optimization
  optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM)

  if opt_state:
    optimizer.load_state_dict(opt_state)

  step = 1
  loss_values=[]

  for epoch in range( last_epoch+1 , EPOCHS + 1 ) if last_epoch else range(1, EPOCHS + 1):

    model.train()
    running_loss = 0.0
    
    # cal one epoch time
    start = time.time()
    correct = 0
    total = 0
    batch_iter = 0

    for images, labels in dataset_loader:
      images = images.to(device)
      # print(images.shape)
      labels = labels.to(device)
      
      outputs, aux_outputs = model(images)
      loss1 = cast(outputs, labels)
      loss2 = cast(aux_outputs, labels)
      loss = loss1 + 0.4*loss2
      running_loss =+ loss.item() * images.size(0)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    #   print("epoch: ", epoch)
    #   print(f"Step [{step * BATCH_SIZE}/{NUM_EPOCHS * len(dataset)}], "
    #         f"Loss: {loss.item():.8f}.")
    #   print("Running Loss=",running_loss)

      batch_iter = batch_iter + 1

      printProgressBar( batch_iter , len(dataset_loader) , prefix = 'Training : ', suffix = 'loss : {:8f}'.format(loss.item()), length = 50)

      step += 1
      # equal prediction and acc
      _, predicted = torch.max(outputs.data, 1)
      # val_loader total
      total += labels.size(0)
      # add correct
      correct += (predicted == labels).sum().item()

      print(f"Acc: {correct / total:.4f}.")
        # cal train one epoch time
    end = time.time()
    loss_values.append(running_loss / len(dataset_loader))
    
    print(f"Epoch [{epoch}/{NUM_EPOCHS}], "
          f"time: {end - start} sec!")

    # Save the model checkpoint
    if epoch%20==0:
    #   LEARNING_RATE=LEARNING_RATE/10
    #   torch.save(model, MODEL_PATH + '/' + MODEL_NAME)

      model_save_name = 'Inception_v3_CIFAR10_32BATCH_lr0.001_crop_bflip_rot'+str(epoch)+'.pth'   #WE keep changing this and saving states ,can be found in excel sheet attached
      torch.save(model.state_dict(), model_save_name)
    print("epoch completed and model copy completed")
    
  torch.save(model,MODEL_NAME)
  print(f"Model save to {MODEL_PATH + '/' + MODEL_NAME}.")

if __name__ == '__main__':
  main()
