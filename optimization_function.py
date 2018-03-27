import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()

# declare all variables
x = tf.Variable(np.random.rand(2,1), dtype=tf.float32, name="x")
# we already make clear, that we are not going to optimize THESE variables
b = tf.Variable([[5],[6]], dtype=tf.float32, trainable=False, name="b")
A = tf.Variable([[9,2],[2,10]], dtype=tf.float32, trainable=False, name="A")
sess.run(tf.global_variables_initializer())

# solving Ax=b for x:
xstar = tf.matrix_solve_ls(A, b)
# now we print the solution
print("x = {}".format(sess.run(xstar)))
objective = 0.5 * tf.matmul(tf.matmul(tf.transpose(xstar), A), xstar) - tf.matmul(tf.transpose(b), xstar) + 42
print ("f(x) = {}".format(sess.run(objective)))
print()

# destroy previous session and make a new one
tf.reset_default_graph()
sess = tf.InteractiveSession()

# define variables in the problem
x = tf.Variable(np.random.rand(2,1), dtype=tf.float32, name="x")
b = tf.Variable([[5],[6]], dtype=tf.float32, trainable=False, name="b")
A = tf.Variable([[9,2],[2,10]], dtype=tf.float32, trainable=False, name="A")
sess.run(tf.global_variables_initializer())

# define expressions
objective = 0.5 * tf.matmul(tf.matmul(tf.transpose(x), A), x) - tf.matmul(tf.transpose(b), x) + 42
grad = tf.matmul(A, x) -b               # this is new line with the gradient
optimize_op = x.assign(x - 0.01 * grad) # this is the new update rule (gradient descent)

# start at some random point again
sess.run(x.assign(np.random.rand(2,1)))
# optimize
for _ in range(300):
    sess.run(optimize_op)

print (sess.run(objective))
print()

grad = tf.gradients(objective, x)[0]    # get gradient from objective wrt. to x
optimize_op = x.assign(x - 0.01 * grad)

# start at some random point again
sess.run(x.assign(np.random.rand(2,1)))
# optimize
for _ in range(300):
    sess.run(optimize_op)

print (sess.run(objective))


import matplotlib.pyplot as plt

def rosenbrock(x, y):
    a, b = 1., 100.
    f = (a - x)**2 + b *(y - x**2)**2
    x_solution = (a, a*a)
    return f, x_solution

# just for visualization
xx, yy = np.meshgrid(np.linspace(-1.3, 1.3, 31), np.linspace(-0.9, 1.7, 31))
zz, solution = rosenbrock(xx, yy)

# destroy previous session and graph
tf.reset_default_graph()
sess = tf.InteractiveSession()

x0 = (-0.5, 0.9)

x = tf.Variable(0, dtype = tf.float64, name="x")
y = tf.Variable(0, dtype = tf.float64, name="y")
objective, _ = rosenbrock(x, y)

optimizer = []
optimizer.append(tf.train.RMSPropOptimizer(0.02).minimize(objective))
optimizer.append(tf.train.GradientDescentOptimizer(0.002).minimize(objective))
optimizer.append(tf.train.AdamOptimizer(0.3).minimize(objective))
optimizer.append(tf.train.MomentumOptimizer(0.002, 0.9).minimize(objective))
optimizer.append(tf.train.AdadeltaOptimizer(0.1).minimize(objective))
optimizer.append(tf.train.AdagradOptimizer(0.1).minimize(objective))

sess.run(tf.global_variables_initializer())

# plot it to see wtf just happened
fig, ax = plt.subplots()
ax.contour(xx, yy, zz, np.logspace(-5, 3, 60), cmap="YlGn_r");
for opt_op in optimizer:
    steps = [x0]
    sess.run([x.assign(x0[0]), y.assign(x0[1])])
    for i in range(100):
        sess.run(opt_op)
        steps.append(sess.run([x,y]))

    steps = np.array(steps)
    ax.plot(steps[:,0], steps[:,1])

ax.plot((x0[0]), (x0[1]), 'o', color='y')
ax.plot(solution[0], solution[1], 'o', color='r');
ax.legend(['GradDesc', 'RMSprop', 'Adam', 'Momentum', 'AdaDelta', 'AdaGrad'], bbox_to_anchor=(1.4, 0.7));
plt.show()
