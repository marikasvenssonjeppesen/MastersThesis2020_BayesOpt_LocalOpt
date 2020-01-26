from global_optimization import go_benchmark
from global_optimization import optimization_strategy
import os
import GPy
import numpy as np
import matplotlib.pyplot as plt
from pylab import savefig

dims = (1, 2)
fmins = []
xmin_obs = []
nbr_fcn_evalss = []
for dim in dims:
    print ('running dimenstion: ', dim)

    schweifel = go_benchmark.Schwefel26(dim)
    base = './test190913NelderMead/'
    basefile = base + '/dim' + str(dim) + 'function_' + 'schweifel26' + '/'
    if not os.path.isdir( base ):
        os.system( "mkdir " +  base )
    if not os.path.isdir( basefile ):
        os.system( "mkdir " +  basefile )
    bayesSpec = optimization_strategy.BayesianOptimizationSpec(nbr_of_points = 10*dim, 
                                                            acquisition_type = None, 
                                                            batch_size = 1, 
                                                            num_cores = 4, 
                                                            basefilelocation = basefile, 
                                                            bayes_iter = 200, 
                                                            max_iter = 20,
                                                            kernel = None)
    

    gl_strategy = optimization_strategy.GlobalStrategy(fcn = schweifel, 
                                                    bayes_spec = bayesSpec)
    
    [xobs, fobs, nbr_fcn_evals] = gl_strategy.compare_to_pure_Nelder_mead()


    fmin = -1*np.max(-1*np.array(fobs))
    xobsmin = np.argmax(-1*np.array(fobs))
    fmins.append(fmin)
    xmin_obs.append(xobs[xobsmin])
    nbr_fcn_evalss.append(nbr_fcn_evals)


print ('Nelder-Mead result :', fmins)

plt.figure('Nelder-Mead results')
plt.plot(dims, fmins, '--vm' )
plt.xlabel('f*')
plt.ylabel('Dimension')
plt.ylim(-100, 2750)
savefig(basefile + '/summary.png')
plt.show()

print ('nbr_fcn_evalss', nbr_fcn_evalss)