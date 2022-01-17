import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

matrix1 = tf.constant(
	[
		[1., 2.],
		[3., 4.]
	]
)

matrix2 = tf.constant(
	[
		[1.],
		[2.]
	]
)

init_op = tf.global_variables_initializer()

print("Metrix 1 shape", matrix1.shape)
print("Metrix 2 shape", matrix2.shape)
with tf.Session() as sess:
	sess.run(init_op)
	print(tf.matmul(matrix1, matrix2).eval())