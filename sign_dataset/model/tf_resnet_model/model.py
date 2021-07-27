import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np
import os
import time
import sys

def save(sess , checkpoint_dir , step , saver , model_dir='' , model_name="ResNet" , final=False):
    
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    checkpoint_dir_final = os.path.join(checkpoint_dir, 'final')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if final == True:

        if not os.path.exists(checkpoint_dir_final):
            os.makedirs(checkpoint_dir_final)
        
        saver.save( sess , os.path.join( checkpoint_dir_final , model_name+'.model.ckpt' ) )

    else:
        saver.save( sess, os.path.join(checkpoint_dir, model_name+'.model'), global_step=step)




def load(sess , checkpoint_dir, saver ,model_dir='ResNet'):
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(ckpt_name.split('-')[-1])
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0

def build_model(batch_size=64 , img_size=28 , c_dim=3 , label_dim=6 , test_x = None , test_y = None , layers=18 ):

        """ Graph Input """
        train_inputs = tf.placeholder(tf.float32, [batch_size, img_size, img_size, c_dim], name='train_inputs')
        train_labels = tf.placeholder(tf.float32, [batch_size, label_dim], name='train_labels')

        test_inputs = tf.placeholder(tf.float32, [ None , img_size, img_size, c_dim], name='test_inputs')
        test_labels = tf.placeholder(tf.float32, [ None , label_dim], name='test_labels')

        lr = tf.placeholder(tf.float32, name='learning_rate')


        """ Model """
        train_logits = build_network(train_inputs , layers = layers , label_dim = 6 )
        test_logits = build_network(test_inputs,layers = layers , label_dim = 6, is_training=False, reuse=True)

        train_loss, train_accuracy = classification_loss(logit=train_logits, label=train_labels)
        test_loss, test_accuracy = classification_loss(logit=test_logits, label=test_labels)
        
        reg_loss = tf.losses.get_regularization_loss()
        train_loss += reg_loss
        test_loss += reg_loss


        """ Training """
        optim = tf.train.MomentumOptimizer(lr, momentum=0.9).minimize(train_loss)


        """" Summary """
        summary_train_loss = tf.summary.scalar("train_loss", train_loss)
        summary_train_accuracy = tf.summary.scalar("train_accuracy", train_accuracy)


        summary_test_loss = tf.summary.scalar("test_loss", test_loss)
        summary_test_accuracy = tf.summary.scalar("test_accuracy", test_accuracy)


        train_summary = tf.summary.merge([summary_train_loss, summary_train_accuracy])
        test_summary = tf.summary.merge([summary_test_loss, summary_test_accuracy])
        
        return ( 
            
            train_inputs ,
            train_labels ,
            test_inputs ,
            test_labels ,
            lr ,

            train_logits ,
            test_logits ,

            train_accuracy ,
            test_accuracy ,

            train_loss ,
            test_loss ,

            optim ,

            summary_train_loss ,
            summary_train_accuracy ,

            summary_test_loss ,
            summary_test_accuracy ,
            
            train_summary , 
            test_summary 
            )

def train(train_x , train_y ,test_x , test_y):

    img_size = train_x.shape[1]
    c_dim = train_x.shape[3]
    label_dim = test_y.shape[1]
    
    batch_size = 64

    res_n = 18

    (
    train_inputs ,
    train_labels ,
    test_inputs ,
    test_labels ,
    lr ,

    train_logits ,
    test_logits ,

    train_accuracy ,
    test_accuracy ,

    train_loss ,
    test_loss ,

    optim ,

    summary_train_loss ,
    summary_train_accuracy ,

    summary_test_loss ,
    summary_test_accuracy ,
    
    train_summary , 
    test_summary 
    ) = build_model( batch_size=batch_size , img_size=img_size , c_dim=c_dim , label_dim=label_dim , test_x=test_x  , test_y=test_y , layers=res_n )
    
    model_name = 'ResNet'

    log_dir = 'logs'

    checkpoint_dir = 'checkpoint'

    epoch = 100

    init_lr = float(0.1)

    iteration = len(train_x) // batch_size    

    model_dir = "{}{}_{}_{}".format( model_name , res_n , batch_size , init_lr )

    init = tf.global_variables_initializer()

    # saver to save model
    saver = tf.train.Saver()

    early_stopping = False

    with tf.Session( config=tf.ConfigProto(allow_soft_placement=True) ) as sess:

        sess.run( init )

        # summary writer
        writer = tf.summary.FileWriter(log_dir + '/' + model_dir, sess.graph)

        # restore check-point if it exits
        # sess , checkpoint_dir, saver ,model_dir='ResNet'
        could_load, checkpoint_counter = load( sess , checkpoint_dir , saver=saver , model_dir=model_dir )
        if could_load:
            epoch_lr = init_lr
            start_epoch = (int)(checkpoint_counter / iteration)
            start_batch_id = checkpoint_counter - start_epoch * iteration
            counter = checkpoint_counter

            if start_epoch >= int(epoch * 0.75) :
                epoch_lr = epoch_lr * 0.01
            elif start_epoch >= int(epoch * 0.5) and start_epoch < int(epoch * 0.75) :
                epoch_lr = epoch_lr * 0.1
            print(" [*] Load SUCCESS")
        else:
            epoch_lr = init_lr
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, epoch):

            best_loss = max_value()
            patience = 0

            if early_stopping == True :
                break

            if epoch == int(epoch * 0.5) or epoch == int(epoch * 0.75) :
                epoch_lr = epoch_lr * 0.1

            # get batch data
            for idx in range(start_batch_id, iteration):
                batch_x = train_x[idx*batch_size:(idx+1)*batch_size]
                batch_y = train_y[idx*batch_size:(idx+1)*batch_size]

                # batch_x = data_augmentation(batch_x, img_size, dataset_name)

                train_feed_dict = {
                    train_inputs : batch_x,
                    train_labels : batch_y,
                    lr : epoch_lr
                }

                test_feed_dict = {
                    test_inputs : test_x,
                    test_labels : test_y
                }


                # update network
                _, summary_str, train_loss_val, train_accuracy_val = sess.run(
                    [optim, train_summary, train_loss, train_accuracy], feed_dict=train_feed_dict)
                writer.add_summary(summary_str, counter)

                # test
                summary_str, test_loss_val, test_accuracy_val = sess.run(
                    [test_summary, test_loss, test_accuracy], feed_dict=test_feed_dict)
                writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%3d/%3d] time: %4.4f, train_accuracy : %.2f , train_loss : %.2f , test_accuracy : %.2f , test_loss : %.2f , learning_rate : %.4f" \
                        % (epoch, idx, iteration, time.time() - start_time, train_accuracy_val , train_loss_val , test_accuracy_val , test_loss_val , epoch_lr))

                # Early Stopping - based on 'test_loss_val'

                best_loss = min( test_loss_val , best_loss )
                
                if best_loss < test_loss_val :
                    print( ' [*] Found Best Test Loss : {} , patience : {} , diff : {} '.format(best_loss,patience,float( test_loss_val - best_loss )) )

                if best_loss < test_loss_val and ( 0.01 > float( test_loss_val - best_loss ) and 0.001 < float( test_loss_val - best_loss )  ) :

                    patience += 1

                    if patience > 5:

                        print( ' [*] Early Stopping at : \n')

                        print("Epoch: [%2d] [%3d/%3d] time: %4.4f, train_accuracy : %.2f , train_loss : %.2f , test_accuracy : %.2f , test_loss : %.2f , learning_rate : %.4f" \
                                % (epoch, idx, iteration, time.time() - start_time, train_accuracy_val , train_loss_val , test_accuracy_val , test_loss_val , epoch_lr))
                        
                        # Save the Model
                        
                        save( sess , checkpoint_dir, counter , saver = saver , model_dir = model_dir , model_name = model_name , final=True )

                        early_stopping = True

                        break

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            # sess , checkpoint_dir , step , saver , model_dir='' , model_name="ResNet"
            save( sess , checkpoint_dir, counter , saver = saver , model_dir = model_dir , model_name = model_name )

        # save model for final step
        save( sess , checkpoint_dir, counter , saver = saver , model_dir = model_dir , model_name = model_name , final=True)

def build_network(x, layers = 50 , label_dim = 6 , is_training=True , reuse=False):
    
    with tf.variable_scope("network", reuse=reuse):

        if layers < 50 :
            residual_block = resblock
        else :
            residual_block = bottle_resblock

        residual_list = get_residual_layer(layers)

        ch = 32 # paper is 64
        x = conv(x, channels=ch, kernel=3, stride=1, scope='conv')

        for i in range(residual_list[0]) :
            x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))

        ########################################################################################################

        x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')

        for i in range(1, residual_list[1]) :
            x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))

        ########################################################################################################

        x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')

        for i in range(1, residual_list[2]) :
            x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))

        ########################################################################################################

        x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')

        for i in range(1, residual_list[3]) :
            x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))

        ########################################################################################################


        x = batch_norm(x, is_training, scope='batch_norm')
        x = relu(x)

        x = global_avg_pooling(x)
        x = fully_conneted(x, units=label_dim, scope='logit')

        return x


# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf_contrib.layers.variance_scaling_initializer()
weight_regularizer = tf_contrib.layers.l2_regularizer(0.0001)


##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x,
                             filters=channels,
                             kernel_size=kernel, 
                             kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, 
                             use_bias=use_bias, 
                             padding=padding)

        return x


def fully_conneted(x, units, use_bias=True, scope='fully_0'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x


def get_residual_layer(res_n) :
    x = []

    if res_n == 18 :
        x = [2, 2, 2, 2]

    if res_n == 34 :
        x = [3, 4, 6, 3]

    if res_n == 50 :
        x = [3, 4, 6, 3]

    if res_n == 101 :
        x = [3, 4, 23, 3]

    if res_n == 152 :
        x = [3, 8, 36, 3]

    return x


def resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='resblock') :
    with tf.variable_scope(scope) :

        x = batch_norm(x_init, is_training, scope='batch_norm_0')
        x = relu(x)


        if downsample :
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            x_init = conv(x_init, channels, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else :
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')

        x = batch_norm(x, is_training, scope='batch_norm_1')
        x = relu(x)
        x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_1')



        return x + x_init


def bottle_resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='bottle_resblock') :
    with tf.variable_scope(scope) :
        x = batch_norm(x_init, is_training, scope='batch_norm_1x1_front')
        shortcut = relu(x)

        x = conv(shortcut, channels, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_front')
        x = batch_norm(x, is_training, scope='batch_norm_3x3')
        x = relu(x)

        if downsample :
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels*4, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else :
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels * 4, kernel=1, stride=1, use_bias=use_bias, scope='conv_init')

        x = batch_norm(x, is_training, scope='batch_norm_1x1_back')
        x = relu(x)
        x = conv(x, channels*4, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_back')

        return x + shortcut



##################################################################################
# Sampling
##################################################################################

def flatten(x) :
    return tf.layers.flatten(x)

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap

def avg_pooling(x) :
    return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')

##################################################################################
# Activation function
##################################################################################


def relu(x):
    return tf.nn.relu(x)


##################################################################################
# Normalization function
##################################################################################

def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        epsilon=1e-05,
                                        center=True,
                                        scale=True,
                                        updates_collections=None,
                                        is_training=is_training, 
                                        scope=scope
                                        )

##################################################################################
# Loss function
##################################################################################

def classification_loss(logit, label) :
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit))
    prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(label, -1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return loss, accuracy


def max_value():
    return float(sys.maxsize)

def min_value():
    return float( -sys.maxsize - 1 )