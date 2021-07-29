
import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
# import torch.nn.functional as F
from utils import encode , decode , split_input_target , to_categorial , to_one_hot

class TextDataset(Dataset):

    def __init__(self,file_path,seq_length=64,one_hot=False):

        text = open(file_path,'rb').read().decode(encoding='utf-8')

        self.one_hot = one_hot

        text_len = len(text)

        vocab = sorted(set(text))

        self.vocab_size = max(list(map(lambda x: encode(ord(x)),vocab)))

        self.seq_length = seq_length

        examples_per_epoch = text_len//self.seq_length

        self.batches = [
            np.array(
                list(
                    map(
                    lambda x: encode(ord(x)),
                    text[i:i+seq_length]
                )
                )
            )
            for i in range(0,text_len,seq_length)
        ]

        self.batch_len = len(self.batches)

    def __len__(self):
        return self.batch_len

    def getvocab(self):
        return self.vocab_size

    def __getitem__(self , index):

        chunk = self.batches[index]

        input_seq , target_seq = split_input_target(chunk)

        if self.one_hot == True:

            input_seq = to_one_hot(input_seq,num_classes=self.vocab_size)

        return (input_seq,target_seq)


if __name__ == '__main__':

    one_hot = True

    file_path = os.path.join('shakespeare.txt')

    dataset = TextDataset(file_path,one_hot=one_hot)

    print( 'dataset => ' , len(dataset) )

    print('vocab : ',dataset.getvocab() )

    index = random.randint(0,len(dataset))

    input_sample , target_sample = dataset[index]

    if one_hot==True:
        input_sample = to_categorial(input_sample)

    input_sample = map(lambda x : chr(decode(x)),input_sample)

    input_sample = "".join(input_sample)

    target_sample = map(lambda x : chr(decode(x)),target_sample)

    target_sample = "".join(target_sample)

    print('input_sample : => ', input_sample )

    print('target_sample : => ', target_sample )






