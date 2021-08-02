import torch
import torch.nn as nn

import os
from torch import nn , optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
from torch.autograd import Variable
from tqdm import tqdm

from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from config import config
from logger import get_logger

EPOCHS = config.EPOCHS
BATCH_SIZE = config.BATCH_SIZE
LEARNING_RATE = config.LEARNING_RATE
WEIGHT_DECAY = config.WEIGHT_DECAY

CHECKPOINT_PATH = os.path.join( 'checkpoint' , 'inception_v3' )

device = 'cuda' if torch.cuda.is_available() else 'cpu'

log_dir = os.path.join('logs')

if os.path.exists(log_dir) == False:
    os.makedirs(log_dir)

logger = get_logger(__name__, log_path=os.path.join(log_dir, 'inception_v3.log'), console=True)

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def train_single_epoch(model,scaler,optimizer,criterion,train_loader,writer,logger,device='cpu',epoch=0):

    loop = tqdm(train_loader,leave=True)
    losses = []

    train_loss , train_correct , train_total = 0 , 0 , 0

    running_loss = 0
    start_point = time.time()
    at_start = True

    for idx,(inputs,targets) in enumerate(loop):

       # if at_start == True:
       #      # create grid of images
       #      img_grid = make_grid(inputs.to(device))

       #      # write to tensorboard
       #      writer.add_image('sample Images', img_grid)

       #      writer.add_graph(model, inputs)

       #      at_start = False

        with torch.cuda.amp.autocast():

            optimizer.zero_grad()

            inputs , targets = Variable(inputs).to(device) , Variable(targets).to(device)

            preds , aux_ops_1 = model(inputs)

        loss = criterion(preds,targets)

        scaler.scale(loss).backward()

        # loss.backward()

        # optimizer.step()

        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

        losses.append(loss.item())

        running_loss = sum(losses)/len(losses)

        train_loss += loss.item()
        train_correct += (preds.argmax(dim=1) == targets).sum().item()
        train_total += len(preds)

        global_step = epoch * len(train_loader) + idx

        writer.add_scalar('training_loss',running_loss,global_step)

    train_accuracy = (100 * train_correct)/train_total

    train_loss = train_loss / len(train_loader)


    mean_loss = sum(losses)/len(losses)

    logger.info('Training Mean Loss on Epoch:{} = {}'.format(epoch,mean_loss))

    logger.info('train-acc : %.4f%% train-loss : %.5f' % (train_accuracy,train_loss))

    logger.info('elapsed time: %ds' % (time.time() - start_point))

    print('[*] Training Mean Loss : {}'.format(mean_loss))

def test_single_epoch(model,criterion,test_loader,writer,logger,device='cpu',epoch=0):

    loop = tqdm(test_loader,leave=True)
    losses = []

    test_loss , test_correct , test_total = 0 , 0 , 0

    running_loss = 0
    start_point = time.time()

    model.eval()

    for idx,(inputs,targets) in enumerate(loop):

        with torch.no_grad():

            with torch.cuda.amp.autocast():

                inputs , targets = Variable(inputs).to(device) , Variable(targets).to(device)

                preds = model(inputs)

            loss = criterion(preds,targets)

            loop.set_postfix(loss=loss.item())

            losses.append(loss.item())

            running_loss = sum(losses)/len(losses)

            test_loss += loss.item()
            test_correct += (preds.argmax(dim=1) == targets).sum().item()
            test_total += len(preds)

            global_step = epoch * len(test_loader) + idx

            writer.add_scalar('testing_loss',running_loss,global_step)

    model.train()

    test_accuracy = (100 * test_correct)/test_total

    test_loss = test_loss / len(test_loader)

    mean_loss = sum(losses)/len(losses)

    logger.info('Mean Testing Loss on Epoch:{} = {}'.format(epoch,mean_loss))

    logger.info('test-acc : %.4f%% test-loss : %.5f' % (test_accuracy,test_loss))

    logger.info('elapsed time: %ds' % (time.time() - start_point))

    print('[*] Testing Mean Loss : {}'.format(mean_loss))

def val_single_epoch(model,criterion,val_loader,writer,logger,device='cpu',epoch=0):

    loop = tqdm(val_loader,leave=True)
    losses = []

    val_loss , val_correct , val_total = 0 , 0 , 0

    running_loss = 0
    start_point = time.time()

    model.eval()

    for idx,(inputs,targets) in enumerate(loop):

        with torch.no_grad():

            with torch.cuda.amp.autocast():

                inputs , targets = Variable(inputs).to(device) , Variable(targets).to(device)

                preds = model(inputs)

            loss = criterion(preds,targets)

            loop.set_postfix(loss=loss.item())

            losses.append(loss.item())

            running_loss = sum(losses)/len(losses)

            val_loss += loss.item()
            val_correct += (preds.argmax(dim=1) == targets).sum().item()
            val_total += len(preds)

            global_step = epoch * len(val_loader) + idx

            writer.add_scalar('validation_loss',running_loss,global_step)

    model.train()

    val_accuracy = (100 * val_correct)/val_total

    val_loss = val_loss / len(val_loader)

    mean_loss = sum(losses)/len(losses)

    logger.info('Mean validation Loss on Epoch:{} = {}'.format(epoch,mean_loss))

    logger.info('val-acc : %.4f%% val-loss : %.5f' % (val_accuracy,val_loss))

    logger.info('elapsed time: %ds' % (time.time() - start_point))

    print('[*] validation Mean Loss : {}'.format(mean_loss))


def run(train_loader,test_loader,val_loader,model):

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

    # net = InceptionV1( training=True , aux_logits=True , num_classes=10 ).cuda()

    comment = ' model = {} batch_size = {} lr = {} optimizer = {}'.format(
                                                            'inception_v3',
                                                             BATCH_SIZE,
                                                             LEARNING_RATE,
                                                             'SGD'
                                                             )

    writer = SummaryWriter(comment=comment)

    scaler = torch.cuda.amp.GradScaler()

    if model_state:
        model.load_state_dict(model_state)

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(model.parameters(),
                          lr=LEARNING_RATE,
                          weight_decay=WEIGHT_DECAY,
                          momentum=.9,
                          nesterov=True
                          )

    if opt_state:
        optimizer.load_state_dict(opt_state)

    logger.info('Starting with {} instances'.format(BATCH_SIZE))

    for epoch in range( last_epoch+1 , EPOCHS + 1 ) if last_epoch else range(1, EPOCHS + 1):

        print('[Epoch %d]' % epoch)

        train_single_epoch(model,scaler,optimizer,criterion,train_loader,writer,logger,device=device,epoch=epoch)

        test_single_epoch(model,criterion,test_loader,writer,logger,device=device,epoch=epoch)

        val_single_epoch(model,criterion,val_loader,writer,logger,device=device,epoch=epoch)

        torch.save( {
             'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
                    } , os.path.join( CHECKPOINT_PATH , 'chest-xray-checkpoint-{:04d}.bin'.format(epoch) ) )

def load_model(label_dim):

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

    model = inception_v3( aux_logits=True , num_classes=label_dim ).cuda()

    if model_state:
        model.load_state_dict(model_state)

    return model

def test( dataset , list_class , net ):

    test_loader = DataLoader( dataset, batch_size=128,shuffle=False, num_workers=2 )

    labels_array = []

    for inputs in test_loader:

        inputs = Variable(inputs).cuda()


        preds = net(inputs)

        predictions_argmax = np.array( preds.argmax(dim=1).cpu() )

        # print( 'predictions_argmax ' ,predictions_argmax )

        def convert(x):

            return list_class[int(x)].decode('utf-8')

        convert_label = np.vectorize( convert )

        labels = convert_label( predictions_argmax )

        # print('labels : ' , labels )

        labels_array.extend( [ *labels.tolist() ]  )
    # print('labels_array' , labels_array)
    return np.array( labels_array )

