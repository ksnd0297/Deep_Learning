import tensorflow as tf

a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)

d = tf.multiply(a, b) # a * b
e = tf.add(c, b) # c + b
f = tf.subtract(d, e) # d - e