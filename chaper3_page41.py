import tensorflow as tf

g1=tf.Graph()
with g1.as_default():
    v=tf.get_variable(
        "v",shape=[1],initializer=tf.zeros_initializer
    )

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse =True):
        print(sess.run(tf.get_variable("v")))