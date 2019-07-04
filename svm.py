from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')
    
import numpy as np
from matplotlib import pyplot as plt 
import matplotlib as mpl 
#%matplotlib inline 

x = np.array([
    [-2,4,-1],
    [4,1,-1],
    [1,6,-1],
    [2,4,-1],
    [6,2,-1],
])

y = np.array([-1,-1,1,1,1])

for d, sample in enumerate(x):
    if d < 2:
        plt.scatter(sample[0], sample[1], s = 120 ,marker= "_", linewidths= 2)
    else:
        plt.scatter(sample[0], sample[1], s = 120 ,marker= "+", linewidths= 2)

plt.plot([-2,6],[6,0.5])
plt.show()

def svm_sdg_plot(x,y):

    w = np.zeros(len(x[0]))

    eta = 1
    epochs = 100000
    errors = []

    for epoch in range(1,epochs):
        error = 0
        for i , x in enumerate(x):
            if (y[i]*np.dot(x[i],w)) < 1:
                w = w + eta * ((x[i] * y[i]) + (-2 * (1/epoch)* w))
                error = 1
            else:
                w = w + eta *  (-2 *(1/epoch)* w)
        errors.append(error)

    plt.plot(errors, '|')
    plt.ylim(0.5, 1.5)
    plt.axes().set_yticklabels([])
    plt.xlabel('Epoch')
    plt.ylabel('Misclassified')
    plt.show()

    return w

w = svm_sdg_plot(x,y)

for d, sample in enumerate(x):
    if d < 2:
        plt.scatter(sample[0], sample[1], s = 120 ,marker= "_", linewidths= 2)
    else:
        plt.scatter(sample[0], sample[1], s = 120 ,marker= "+", linewidths= 2)

plt.scatter(2,2, s=120, marker='_', linewidths=2, color = 'yellow')
plt.scatter(4,3, s=120, marker='+', linewidths=2, color = 'blue')

x2 = [w[0], w[1], -w[1], w[0]]
x3 = [w[0], w[1], w[1], -w[0]]

x2x3 = np.array([x2,x3])
x,y,U,V = zip(*x2x3)
ax = plt.gca()
ax.quiver(x,y,U,V,scale = 1, color = 'blue')
plt.show()