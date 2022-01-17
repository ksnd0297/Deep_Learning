import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Node and Graph Def
a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)

d = tf.multiply(a, b) # a * b
e = tf.add(c, b) # c + b
f = tf.subtract(d, e) # d - e

# Make Session Grahp exe
sess = tf.Session()
outs = sess.run(f)
sess.close()
print("outs = {}".format(outs))