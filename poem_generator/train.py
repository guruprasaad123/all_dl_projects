import os
from dataset import TextDataset
from model import Model , rnn_model
from utils import sample_from_probs , predict , decode
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from logger import get_logger
from utils import generate_text , sample
# from test import sample , predict

device = 'cuda' if torch.cuda.is_available() else 'cpu'

log_dir = os.path.join('logs')

if os.path.exists(log_dir) == False:
    os.makedirs(log_dir)

logger = get_logger(__name__, log_path=os.path.join(log_dir, 'gru_nn.log'), console=True)

def train(model,scaler,optimizer,criterion,loader,writer,logger,device='cpu',epoch=0):

    loop = tqdm(loader,leave=True)
    losses = []
    model.init_state()
    clip_norm = 5
    running_loss = 0

    for idx,(input_seq,target_seq) in enumerate(loop):

        with torch.cuda.amp.autocast():

            optimizer.zero_grad()

            input_seq , target_seq = Variable(input_seq).to(device) , Variable(target_seq).to(device)

            output,hidden = model(input_seq)

        loss = criterion(output,target_seq.view(-1))

        scaler.scale(loss).backward()

        # loss.backward()

        # clip gradient norm
        nn.utils.clip_grad_norm(model.parameters(), clip_norm)

        # optimizer.step()

        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

        losses.append(loss.item())

        running_loss = sum(losses)/len(losses)

        global_step = epoch * len(loader) + idx

        writer.add_scalar('training_loss',running_loss,global_step)


    mean_loss = sum(losses)/len(losses)

    logger.info('Mean Loss on Epoch:{} = {}'.format(epoch,mean_loss))

    print('[*] Mean Loss : {}'.format(mean_loss))

def main():

    model_state = None
    optimizer_state = None
    last_epoch = None

    CHECKPOINT_PATH = os.path.join('checkpoint')

    if os.path.exists(CHECKPOINT_PATH):

        dirs = os.listdir(CHECKPOINT_PATH)

        if len(dirs) > 0:

            latest_checkpoint=max(dirs)

            print('Recovered from {}'.format(latest_checkpoint))

            checkpoint = torch.load( os.path.join(CHECKPOINT_PATH,latest_checkpoint) )

            model_state = checkpoint['model_state_dict']
            optimizer_state = checkpoint['optimizer_state_dict']
            last_epoch = checkpoint['epoch']+1

    else:
        os.makedirs(CHECKPOINT_PATH)

    file_path = os.path.join('shakespeare.txt')

    one_hot = False

    dataset = TextDataset(file_path,seq_length=64,one_hot=one_hot)

    vocab_size = dataset.getvocab() + 1

    batch_size = 64

    embedding_dim = 32
    rnn_units = 128
    layers = 2

    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              num_workers=1,
                              pin_memory=True,
                              shuffle=False,
                              drop_last=True,
                              )

    # Instantiate the model with hyperparameters
    # model = Model(input_size=vocab_size, output_size=vocab_size, hidden_dim=100, n_layers=512)

    model = rnn_model(vocab_size,embedding_dim,rnn_units,layers)

    if model_state:
        model.load_state_dict(model_state)

    # We'll also set the model to the device that we defined earlier (default is CPU)
    model.to(device)

    # Define hyperparameters
    n_epochs = 100
    lr=0.001
    weight_decay = 0.0001

    # Define Loss, Optimizer
    criterion = CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler()

    # criterion.to(device)

    comment = ' rnn_type = {} batch_size = {} lr = {} optimizer = {}'.format(
                                                            'gru',
                                                             batch_size,
                                                             lr,
                                                             'adam'
                                                             )

    writer = SummaryWriter(comment=comment)

    logger.info('Training started with {} instances per epoch'.format(batch_size))

    optimizer = Adam(model.parameters(), lr=lr)

    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    for epoch in range(last_epoch,n_epochs) if last_epoch else range(1,n_epochs):

        print('[*] Epoch : {}'.format(epoch))

        train(model,scaler,optimizer,criterion,train_loader,writer,logger,device=device,epoch=epoch)

        if epoch >=0:

            model.eval()

            start_with = generate_text(dataset)

            logger.info(' Generating with Trigger word : {}'.format(start_with))

            output_str = sample(model,start_with=start_with,out_len=512,device='cpu',top_n=10)

            logger.info(' Generated ouput : {}'.format(output_str))

            # writer.add_text(start_with, output_str , epoch)

            model.to(device)

            model.train()


        torch.save( {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'loss': loss,
                } , os.path.join( CHECKPOINT_PATH , '{}-{:04d}.bin'.format('rnn-model',epoch) ) )





    start_with = generate_text(dataset)

    logger.info(' Generating with Trigger word : {}'.format(start_with))

    output_str = sample(model,start_with=start_with,out_len=1024,device=device,top_n=3)

    logger.info(' Generated ouput : {}'.format(output_str))


if __name__ == '__main__':

    main()

