import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as f
from dataset import TextDataset
import random
import os
import numpy as np
from utils import sample_from_probs , predict , decode
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):

        batch_size = x.size(0)

        if x.is_cuda==True:

            # Initializing hidden state for first input using method defined below in GPU
            hidden = self.init_hidden(batch_size,'cuda')
        else:
            # Initializing hidden state for first input using method defined below in CPU
            hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size,device='cpu'):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return hidden


class rnn_model(Module):

    def __init__(self,vocab_size,embedding_dim,rnn_units,layers,dropout=0.0):

        super(rnn_model,self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.layers = layers
        self.rnn_units = rnn_units

        self.prev_state = None
        self.embedding = nn.Embedding(self.vocab_size , self.embedding_dim)

        self.rnn_layer = nn.GRU(self.embedding_dim,
                                self.rnn_units,
                                self.layers,
                                batch_first=True,
                                dropout=0.2)

        self.dropout = nn.Dropout(dropout)

        self.dense = nn.Linear(self.rnn_units,self.vocab_size)

    def init_state(self):

        self.prev_state = None


    def init_hidden(self, batch_size,device='cpu'):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = Variable(
                          torch.zeros(
                                      self.layers,
                                      batch_size,
                                      self.rnn_units
                                      )
                          ).to(device)
        return hidden

    def forward(self,x):

        # batch_size , sequence_length , embedding_dimension
        embedding = self.embedding(x)

        # applying dropout
        embedding = self.dropout(embedding)

        batch_size = x.size(0)

        if self.prev_state == None:

            self.prev_state = self.init_hidden(batch_size,'cuda' if x.is_cuda==True else 'cpu')

        output , current_state = self.rnn_layer(embedding,self.prev_state)

        output = self.dropout(output)

        output = output.contiguous().view(-1,self.rnn_units)

        output = self.dense(output)

        self.prev_state = current_state.detach()

        return output , current_state

    def predict(self,x):

        logits , current_state = self.forward(x)

        # (sequence_len*batch_size,vocab_size)
        probs = f.softmax(logits)

        # (sequence_len,batch_size,vocab_size)
        probs = probs.view(x.size(0),x.size(1),probs.size(1))

        return probs , current_state





if __name__ == '__main__':

    one_hot = False

    file_path = os.path.join('shakespeare.txt')

    dataset = TextDataset(file_path,one_hot=one_hot)

    print( 'dataset => ' , len(dataset) )

    print('vocab : ',dataset.getvocab() )

    index = random.randint(0,len(dataset))

    input_sample , target_sample = dataset[index]

    if one_hot==True:
        input_sample = to_categorial(input_sample)

    input_sample = torch.from_numpy(input_sample).reshape(-1,input_sample.shape[0])

    vocab_size = dataset.getvocab() + 1
    embedding_dim = 32
    rnn_units = 128
    layers = 2

    model = rnn_model(vocab_size,embedding_dim,rnn_units,layers)

    output_seq , hidden = model.predict(input_sample)

    # (seq_len * batch_size , vocab_size)

    print('output_seq : before ',output_seq.shape)

    print('output_seq :  ',output_seq.squeeze().shape)

    # logits = predict(output_seq)

    char_ind = sample_from_probs(output_seq.squeeze(),top_n=150)

    # prob = torch.nn.functional.softmax(last_word_logits, dim=0).detach()
    # char_ind = torch.max(prob, dim=0)[1].item()
    print('word_index : ', chr(decode(char_ind.data[0])) )
    # words.append(dataset.index_to_word[word_index])


