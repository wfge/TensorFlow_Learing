import  glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim

import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

INPUT_DATA='./flower_processed_data.npy'
CKPT_FILE='./inception_v3.ckpt'
TRAIN_FILE='./svae_model'

LEARNING_RTAE=0.0001
STEPS=300
BATCH=32
N_CLASS=5

CHECKPOINT_EXCLUDE_SCOPES='InceptionV3/Logits,InceptionV3/AuxLogits'
TRAINABLE_SCOPES='InceptionV3/Logits,InceptionV3/AuxLogits'

def get_tuned_variables():
    exclusions=[scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]
    variables_to_restore=[]

    for var in slim.get_model_variables():
        exclude1=False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded1=True
                break
            if not excluded1:
                variables_to_restore.append(var)
    return variables_to_restore
def get_trainable_variables():
    scopes=[scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train=[]
    for scope in scopes:
        variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope)
        variables_to_train.extend(variables)
    return variables_to_train
def main(self):
    process_data=np.load(INPUT_DATA)
    training_images=process_data[0]
    n_training_example=len(training_images)
    training_labels=process_data[1]
    validation_images=process_data[2]
    validation_labels=process_data[3]
    testing_images=process_data[4]
    testing_labels=process_data[5]

    print("%d training examples,%d validation examples and %d testing examples."%(n_training_example,len(validation_labels),len(testing_labels)))
    images=tf.placeholder(tf.float32,[None,299,299,3],name='input_images')
    labels=tf.placeholder(tf.int64,[None],name='labels')
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
         logits,_=inception_v3.inception_v3(images,num_classes=N_CLASS)
         trainable_variables=get_trainable_variables()
    tf.losses.softmax_cross_entropy(tf.one_hot(labels,N_CLASS),logits,weights=1.0)
    train_step=tf.train.RMSPropOptimizer(LEARNING_RTAE).minimize(tf.losses.get_total_loss())

    with tf.name_scope('evaluation'):
        correct_prediction=tf.equal(tf.argmax(logits,1),labels)
        evaluation_step=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    load_fn=slim.assign_from_checkpoint_fn(CKPT_FILE,get_tuned_variables(),ignore_missing_vars=True)
    saver =tf.train.saver()
    with tf.Session() as sess:
        init=tf.global_variables_initializer()
        sess.run(init)
        print('Loading tuned variables from %s'%CKPT_FILE)
        load_fn(sess)

        start =0
        end =BATCH
        for i in range(STEPS):
            sess.run(train_step,feed_dict={
                images:training_images[start:end],
                labels:training_labels[start:end]
            })
            if i%30==0 or i+1==STEPS:
                saver.save(sess,TRAIN_FILE,global_step=i)
                validation_acuracy=sess.run(evaluation_step,feed_dict={
                    images:validation_images,
                    labels:validation_labels
                })
                print('Step %d:Validation accuracy=%.lf%'%(i,validation_acuracy*100))
                start =end
                if start==n_training_example:
                    start=0
                    end=start+BATCH
                if end >n_training_example:
                     end=n_training_example
            test_accuary=sess.run(evaluation_step,feed_dict={
                images:testing_images,labels:testing_labels
             })
            print('Final test accuracy%.lf%' % (test_accuary * 100))
if __name__=='__main__':
    tf.app.run()