import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_train = [1, 2, 3]
y_train = [1, 2, 3]

# Shape [1] Tensor
w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


# H = Wx + b
hypothesis = w * x_train + b

# MSE(Mean Squared Error) Lose Function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
#Use Variable Data
sess.run(tf.global_variables_initializer())

for step in range(6001):
    sess.run(train) # Gradient Descent Optimizer
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(w), sess.run(b))