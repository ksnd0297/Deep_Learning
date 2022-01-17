import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

one = np.array([1., 2., 3.])
print(one.ndim) # 1 dim
print(one.shape) # (, 3)

two = np.array(
			[
				[1., 2., 3.],
				[4., 5., 6.]
			]
)
print(two.ndim) # 2 dim
print(two.shape) # (2, 3)

three = tf.constant(
			[
				[
					[1., 2.],
					[3., 4.]
				],
				[
					[5., 6.],
					[7., 8.]
				]
			]
)
tf.shape(three).eval()
