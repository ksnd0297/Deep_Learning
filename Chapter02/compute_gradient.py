import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x = [1, 2, 3]
y = [1, 2, 3]

w = tf.Variable(5.0)

hypothesis = x * w

gradient = tf.reduce_mean((w*x-y)*x)*2

cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)

gvs = optimizer.compute_gradients(cost)

apply_gradients = optimizer.apply_gradients(gvs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
  print(step, sess.run([gradient, w, gvs]))
  sess.run(apply_gradients)