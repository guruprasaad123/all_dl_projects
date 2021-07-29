import numpy as np
import os
import torch
from torch.nn import functional as f
import random
from torch.autograd import Variable

def decode(c):
    if c == 1:
        return 9
    elif c == 3:
        return 13
    elif c == 127 - 30:
        return 10
    # converting to lower-case again
    elif 32 <= c + 30 <= 126:
        return c + 30
    else:
        return 0

def encode(c):
    # '\t'
    if c == 9:
        return 1
    # '\r'
    elif c == 13:
        return 3
    # '\n'
    elif c == 10:
        return 127 - 30
    # converting to upper-case
    elif 32 <= c <= 126:
        return c - 30
    else:
        return 0


def split_input_target(chunk):

    input_text = chunk[:-1]
    target_text = chunk[1:]

    return (input_text, target_text)


def to_one_hot(x,num_classes=3):

    y = np.squeeze(np.eye(num_classes+1)[x.reshape(-1)])

    return y

def to_categorial(x):

    y = np.argmax(x,-1)

    return y

def generate_text(dataset, seq_lens=(2, 4, 8, 16, 32)):
    """
    select subsequence randomly from text dataset
    """
    # randomly choose sequence length

    index = random.randint(0,len(dataset))

    text , target_sample = dataset[index]

    text = text.tolist()

    text = map(lambda x : chr(decode(x)),text)

    text = ''.join(text)

    seq_len = random.choice(seq_lens)
    # randomly choose start index
    start_index = random.randint(0, len(text) - seq_len - 1)
    seed = text[start_index: start_index + seq_len]
    return seed

def sample_from_probs(probs,top_n=10):

    prob_sorted , sorted_indices = torch.sort(probs)

    # setting probabities after top_n to '0'

    # [:-n] from first , leave the last n elements

    # print('probs : ',probs.shape,sorted_indices.data[:-top_n].shape)

    probs[sorted_indices.data[:-top_n]] = 0

    picked = torch.multinomial(probs,1)

    return picked

def to_tensor(text,device='cpu'):

    x = np.array(
                 list(
                      map(
                          lambda x:encode(ord(x)),
                          text)
                      )
                 )

    x = torch.from_numpy(x)

    with torch.no_grad():
        x = Variable(x).to(device)

    # expand 1 dimensions; (99) -> (99,1)
    x = x.unsqueeze(1)

    return x


def predict(model,x,device='cpu',top_n=10,previous=None):

    if previous:

        previous = to_tensor(previous,device=device)

        previous = previous.view(1,previous.size(0))

        # print('previous : ',previous.shape)

        model.predict(previous)

    x = to_tensor(x,device=device)

    # print('x : ',x.shape)

    out , hidden = model.predict(x)

    char_ind = sample_from_probs(out.squeeze(),top_n=top_n)

    ch = chr(decode(char_ind.data[0]))

    return ch , hidden

def sample(model,start_with='ROMEO:',out_len=100,device='cpu',top_n=10):

    model.to(device)

    model.init_state()

    start_with = [ch for ch in start_with]

    # [-n:] = takes the last n elements
    # [:-n] = leave the n elements from last , take the rest from first

    previous , current = start_with[:-1] , start_with[-1:]

    # print('previous',previous)

    # print('current',current)

    for _ in range(out_len):

        current , hidden = predict(model,current,device=device,top_n=top_n,previous=previous)

        start_with.append(current)

        current = list(current)

        previous = None

    sentence = "".join(start_with)

    return sentence

# def predict(logits):

#     # (sequence_len*batch_size,vocab_size)
#     probs = f.softmax(logits)

#     # (sequence_len,batch_size,vocab_size)
#     probs = probs.view(100,-1,probs.size(1))

#     return probs
