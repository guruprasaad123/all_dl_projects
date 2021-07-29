import os
import numpy as np
import torch
import torch.nn as nn
from model import Model
from dataset import TextDataset
from utils import encode , decode , split_input_target , to_categorial , to_one_hot
from torch.autograd import Variable

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict(model,start_with,dict_size):

    x = np.array(
                 list(
                      map(
                          lambda x:encode(ord(x)),
                          start_with)
                      )
                 )

    # dict_size=model.getvocab()

    x = to_one_hot(x,num_classes=dict_size-1)

    x = x.reshape((-1,x.shape[0],x.shape[1]))

    x = torch.from_numpy(x)

    x = Variable(x).float().to(device)

    out , hidden = model(x)

    prob = nn.functional.softmax(out[-1], dim=0).data

    # Taking the class with the highest probability score from the output
    char_ind = torch.max(prob, dim=0)[1].item()

    ch = chr(decode(char_ind))

    return ch , hidden


def sample(model,dict_size,start_with='have',out_len=15):

    start_with = [ch for ch in start_with]
    str_len = len(start_with)

    roll_over = out_len - str_len

    for _ in range(roll_over):

        char , hidden = predict(model,start_with,dict_size)

        start_with.append(char)

    sentence = "".join(start_with)

    return sentence


def main():

    checkpoint_dir = 'checkpoint'

    model_state = None

    if os.path.exists(checkpoint_dir):

        dirs = os.listdir(checkpoint_dir)

        if len(dirs) > 0:

            latest_checkpoint=max(dirs)

            print('Recovered from {}'.format(latest_checkpoint))

            checkpoint = torch.load( os.path.join(checkpoint_dir,latest_checkpoint) )

            model_state = checkpoint['model_state_dict']

    file_path = os.path.join('shakespeare.txt')

    dataset = TextDataset(file_path,one_hot=True)

    vocab_size = dataset.getvocab() + 1

    # Instantiate the model with hyperparameters
    model = Model(input_size=vocab_size, output_size=vocab_size, hidden_dim=100, n_layers=512)

    # We'll also set the model to the device that we defined earlier (default is CPU)
    model.to(device)

    if model_state:
        model.load_state_dict(model_state)

    model.eval()

    output_str = sample(model,vocab_size,start_with='ROMEO:',out_len=100)

    print('output : {}'.format(output_str))


if __name__ == '__main__':

    main()
