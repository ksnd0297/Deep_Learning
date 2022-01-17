import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()
tf.set_random_seed(777)

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32) # A B C D
x_data = xy[:,0:-1] # A B C
y_data = xy[:, [-1]] # D

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1], name='weight'))
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(x_data)
print(y_data)

for step in range(2001):
	cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
	if step % 10 == 0:
		print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

print("Your score will be ", sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))

print("Other scores wii be ", sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))