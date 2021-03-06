import numpy as np 
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler 
import tensorflow as tf 

housing_data = fetch_california_housing()

#print(housing_data.data.shape)

m_,n=housing_data.data.shape

housing_data_plus_bias = np.c_[np.ones((m_,1)),housing_data.data]

scalar = StandardScaler()

#scalar.fit(housing_data_plus_bias)

scaled_housing_data_plus_bias = scalar.fit_transform(housing_data_plus_bias)

#print(scaled_housing_data_plus_bias)

# print(housing_data_plus_bias)

# print(housing_data_plus_bias.shape)

learning_rate=0.01
epochs = 1000

x= tf.constant(housing_data_plus_bias,dtype=tf.float32,name='x')
y= tf.constant(housing_data.target.reshape(-1,1) ,dtype=tf.float32,name='y')
x_t = tf.transpose(x)
#theta = tf.matmul(tf.matrix_inverse(tf.matmul(x_t,x)),tf.matmul(x_t,y))

#theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0),name="theta")

m = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0),name="m")
b = tf.Variable(tf.random_uniform([1,1],-1.0,1.0),name="b")

y_pred= tf.matmul(x,m)+b

error = tf.square(y_pred - y)
mse=tf.reduce_mean(error,name='mse')

#y_ = (tf.matmul(x,m)+b)

m_gradients = tf.Variable(-2/m_ * tf.matmul(x_t,y-y_pred))

b_gradients = tf.Variable((-2/m_) * tf.reduce_mean(y-y_pred))

# m_gradients_upd = tf.assign(m_gradients,m_gradients)

# b_gradients_upd = tf.assign(b_gradients,b_gradients)


m_ops = tf.assign(m, m- (learning_rate * m_gradients))
b_ops = tf.assign(b, b- (learning_rate * b_gradients))


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(epochs):
        if i % 100 == 0 :
            print("MSE = {}".format(mse.eval()))
        sess.run([m_ops,b_ops])
    
    print("Reduced MSE = {}".format(mse.eval()))
    print("Best m = {} ,b = {}".format(m.eval(),b.eval()))

