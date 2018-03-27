import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

x = tf.Variable(np.random.rand(2,1), dtype=tf.float32, name="x")

# our function
# this is going to make a call to pybertini
objective = x[0]**2 + x[1]**2

optimize_op = tf.train.GradientDescentOptimizer(0.01).minimize(objective)
# start at some random point again
sess.run(x.assign(np.random.rand(2,1)))

# optimize
for _ in range(300):
    sess.run(optimize_op)

print(sess.run(objective))
print(sess.run(x))
