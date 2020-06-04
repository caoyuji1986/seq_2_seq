import tensorflow as tf
tf.enable_eager_execution()
a = tf.convert_to_tensor(value=[
	[
		[1,2,3],[4,5,6],[21,22,23],[24,25,26]
	],[
		[7,8,9],[10,11,12],[27,28,29],[210,211,212]
	],[
		[13,14,15],[16,17,18],[213,214,215],[216,217,218]
	]
])
b =tf.gather_nd(params=a, indices=[[0,1],[2,1],[0,1]])
print(a)
print(b)