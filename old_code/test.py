from GPyOpt.methods import BayesianOptimization 
from global_optimization.go_benchmark import Schwefel26
import sobol_seq
import numpy as np
import matplotlib.pyplot as plt
from pylab import savefig
from scipy import optimize
#Import Modules

#GPyOpt - Cases are important, for some reason
import GPyOpt



import pandas as pd

#Plotting tools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numpy.random import multivariate_normal



dim = 2
fun = Schwefel26(dim)

bounds = list(fun.bounds)

domain = []
for i in range(0,fun.dimensions):
    domain.append({'name': 'var_1', 'type': 'continuous', 'domain': bounds[i]})
 
myBopt_1d = BayesianOptimization(f=fun.evaluator, domain=domain)
myBopt_1d.run_optimization(max_iter=5)
myBopt_1d.plot_acquisition()
