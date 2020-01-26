

'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from  global_optimization.go_benchmark import Schwefel26

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


#fig = plt.figure()
#ax = fig.gca(projection='3d')


schwefel = Schwefel26(2)
# Make data.
n = 50
X = np.linspace(-500, 500, n)
Y = np.linspace(-500, 500, n)

x2, y2 = np.meshgrid(X,Y)
z2 = x2
print (np.size(z2))
x = []
y = []
z = []
for i in range(0, n):
    for j in range(0, n):
        #print (X[i], Y[j])
        x.append(X[i])
        y.append(Y[j])
        z.append(schwefel.evaluator(X[i], Y[j]))

        #z2[i][j] = schwefel.evaluator(x2[i][j], y[i][j] )

#print(x)
#print(y)

fig = plt.figure()
plt.plot(x, y, 'o')

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(x, y , z)
'''
#fig = plt.figure()
#ax.plot_trisurf(x2, y2, z2, linewidth=0.2, antialiased=True)
'''
plt.show()



