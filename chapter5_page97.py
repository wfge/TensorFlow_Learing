import tensorflow as tf
import os,sys
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE=784
OUTPUT_NODE=10

LAYER1_NODE=500
BATCH_SIZE=100

LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99
REGULARIZATION_RATE=0.0001
TRAINING_STEPS=30000
MOVING_AVERAGE_DECAY=0.99

def inference(input_tensor,avg_calss,weights1,biases1,weights2,biases2):
    if avg_calss==None:
        layer1=tf.nn.relu(tf.matual(input_tensor,weights1)+biases1)
        return tf.matual(layer1,weights2)+biases2
    else:
        layer1=tf.nn.relu(tf.matmul(input_tensor,avg_calss.average(weights1))+avg_calss.average(biases1))
        return tf.matmul(layer1,avg_calss.average(weights2)+avg_calss.average(biases2))

def train(mnist):
    x=tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_=tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')
    weights1=tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

    weights2=tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    y=inference(x,None,weights1,biases1,weights2,biases2)
    global_step=tf.Variable(0,trainable=False)

    variable_averges=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averges_op=variable_averges.apply(tf.trainable_variables())

    average_y=inference(x,variable_averges,weights1,biases1,weights2,biases2)

    cross_entroy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.arg_max(y_,1))
    cross_entroy_mean=tf.reduce_mean(cross_entroy)

    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization=regularizer(weights1)+regularizer(weights2)

    loss=cross_entroy_mean+regularization

    learning_rate=tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples,
        LEARNING_RATE_DECAY
    )
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

def main(argv=None):
    #mnist=input_data.read_data_sets("/tmp/data",one_hot=True)
    mnist = input_data.read_data_sets("/path/to/MNIST_data", one_hot=True)
    print(mnist.train.num_examples)


