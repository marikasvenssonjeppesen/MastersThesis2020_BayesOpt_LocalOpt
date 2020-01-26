from global_optimization import go_benchmark
from global_optimization import optimization_strategy
import os
import GPy

dims = (1,)
acquisition_types = ('EI',)# 'MPI')
kernels_name = ('Matern52',)# 'Matern32', 'Exponential')
for dim in dims:
    for acquisition_type in acquisition_types:
        print ("--------------------test for dim ", dim, "------------------")
        for kernel_name in kernels_name:

            if kernel_name == 'Matern52':
                kernel = GPy.kern.Matern32(dim, ARD=True, variance=1.)
            elif kernel_name == 'Matern32':
                kernel = GPy.kern.Matern52(dim, ARD=True, variance=1.)
            elif kernel_name == 'Exponential':
                kernel = GPy.kern.Exponential(dim, ARD=True, variance=1.)
            else:
                kernel = None
                
            assert kernel is not None, kernel

            schweifel = go_benchmark.Schwefel26(dim)
            # Data will be saved in this folder
            base = './test190911_bayes_acq_type' + acquisition_type + \
            'kernel' + kernel_name + '/'
            basefile = base + '/dim' + str(dim) + 'function_' + 'schweifel26' + '/'
            if not os.path.isdir( base ):
                os.system( "mkdir " +  base )
            if not os.path.isdir( basefile ):
                os.system( "mkdir " +  basefile )
            bayesSpec = optimization_strategy.BayesianOptimizationSpec(nbr_of_points = 10*dim, 
                                                                    acquisition_type = acquisition_type, 
                                                                    batch_size = 1, 
                                                                    num_cores = 4, 
                                                                    basefilelocation = basefile, 
                                                                    bayes_iter = 10, 
                                                                    max_iter = 10,
                                                                    kernel = kernel)
            print('running Bayesian Optimization Algorithm:')
            gl_strategy = optimization_strategy.GlobalStrategy(fcn = schweifel, 
                                                            bayes_spec = bayesSpec)
            
            
            gl_strategy.run()