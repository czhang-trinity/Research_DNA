import tensorflow as tf
import numpy as np
import array as arr
import pandas as pd
import random




train = pd.read_csv("age_kept_output.txt")
test = pd.read_csv("age_testset.txt")


train = train.drop(columns = ['availability','patient'])
test = test.drop(columns = ['availability','patient'])


train = train.to_numpy(dtype = 'float32')
test = test.to_numpy(dtype = 'float32')
sigmoid_label = np.zeros(700, dtype = int)



'''
train = np.genfromtxt('train.txt', 'float32', delimiter=',')
test = np.genfromtxt('test.txt', 'float32', delimiter = ',')
'''


y_test = np.multiply(test[:,0],1)
x_test = np.delete(test,0,axis = 1)

x = tf.placeholder('float32', [None,45])
y = tf.placeholder('float32',)
z = tf.placeholder('float32',)

n_nodes = 20
batch_size = 10
beta = 0.01
'''
w1 = tf.get_variable("w1",shape=[45,n_nodes], initializer = tf.contrib.layers.xavier_initializer())
w2 = tf.get_variable("w2",shape=[n_nodes,n_nodes], initializer = tf.contrib.layers.xavier_initializer())
w3 = tf.get_variable("w3",shape=[n_nodes,1], initializer = tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("b1",shape=[n_nodes], initializer = tf.zeros_initializer())
b2 = tf.get_variable("b2",shape=[n_nodes], initializer = tf.zeros_initializer())
b3 = tf.get_variable("b3",shape=[1], initializer = tf.zeros_initializer())
'''
w1 = tf.Variable(tf.random_normal([45, n_nodes]))
#w2 = tf.Variable(tf.random_normal([n_nodes, n_nodes]))
w3 = tf.Variable(tf.random_normal([n_nodes, 1]))
b1 = tf.Variable(tf.random_normal([n_nodes]))
#b2 = tf.Variable(tf.random_normal([n_nodes]))
b3 = tf.Variable(tf.random_normal([1]))
'''
def self_loss(labels, prediction):
	constant = tf.constant([0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9], dtype = 'float32')
	diff = tf.abs(tf.subtract(labels, prediction))
	diff = tf.add(constant,diff)
	loss = tf.floor(diff)
	return loss
'''	
	

def network(input):    
	layer1 = tf.nn.leaky_relu(tf.add(tf.matmul(input,w1),b1))

#	if drop:
#		layer1 = tf.nn.dropout(layer1, 0.1)       
	#layer2 = tf.nn.leaky_relu(tf.add(tf.matmul(layer1,w2),b2))
	prediction = tf.nn.leaky_relu(tf.add(tf.matmul(layer1,w3),b3))
	return prediction

def penalty(prediction, labels):
	diff = tf.floor((tf.abs(tf.subtract(labels, prediction))+0.9))
	return diff 
	
	
def train_network(input):
    prediction = network(input)
    diff = penalty(prediction, y)
    #loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=y,predictions = prediction) )
    #cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = z, logits = diff) )
    #loss = cross_entropy
    loss = tf.reduce_mean(tf.losses.absolute_difference(labels=y,predictions = prediction) )
    #regularizer = tf.nn.l2_loss(w1)+tf.nn.l2_loss(w3)
    #loss = tf.reduce_mean(loss+regularizer*beta)
    accuracy = tf.losses.absolute_difference(labels=y,predictions = prediction)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
    
    hm_epochs = 500
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            np.random.shuffle(train)
            y_train = np.multiply(train[:,0],1)
            x_train = np.delete(train,0,axis = 1)
            #print(train)
            epoch_loss = 0
            epoch_accuracy = 0
            i = 0
            while i < len(x_train):
                start = i
                end = i+batch_size
                if end > len(x_train):
                    end = len(x_train)
                batch_x = np.array(x_train[start:end])
                batch_y = np.array(y_train[start:end])
                batch_z = np.array(sigmoid_label[start:end])
                _, c, a, p = sess.run([optimizer, loss,accuracy,prediction], feed_dict={x: batch_x, y: batch_y, z:batch_z})
                epoch_loss += c
                i+=batch_size
                epoch_accuracy += a
                
                if(epoch+1 == hm_epochs):
                    print(a)
                    for j in range(batch_size):
                        print('p: ', p[j], 'a: ', batch_y[j])
                   

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            print('accuracy', epoch_accuracy/70)

        test_prediction = network(input)
        test_accuracy = tf.losses.absolute_difference(labels=y,predictions = prediction)
        test_accuracy = sess.run(test_accuracy,  feed_dict={x: x_test, y: y_test})
        print(test_accuracy)
			
			

        

        

        
train_network(x)
'''
print(x_train.shape)
print(y_train.shape)
print(x_train.dtype)
print(y_train.dtype)
print(len(x_train[0]))
'''
