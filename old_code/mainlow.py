from global_optimization import go_benchmark
from global_optimization import optimization_strategy
import os
import GPy

dims = (1, 2)#, 4,  6, 8, 10)
acquisition_types = ('EI',)#('LCB', 'EI', 'MPI')
kernels_name = ('Matern52',)# 'Matern32', 'Exponential')
for dim in dims:
    for acquisition_type in acquisition_types:
        print ("--------------------test for dim ", dim, "------------------")
        for kernel_name in kernels_name:
            if True:
                if kernel_name == 'Matern52':
                    kernel = GPy.kern.Matern52(dim, ARD=True, variance=1.)
                elif kernel_name == 'Matern32':
                    kernel = GPy.kern.Matern32(dim, ARD=True, variance=1.)
                elif kernel_name == 'Exponential':
                    kernel = GPy.kern.Exponential(dim, ARD=True, variance=1.)
                else:
                    kernel = None

                schweifel = go_benchmark.Ackley(dim)
                base = './test190917_bayes_acq_type' + acquisition_type + \
                'kernel' + kernel_name + '/'
                basefile = base + '/dim' + str(dim) + 'function_' + 'schweifel26' + '/'
                if not os.path.isdir( base ):
                    os.system( "mkdir " +  base )
                if not os.path.isdir( basefile ):
                    os.system( "mkdir " +  basefile )
                bayesSpec = optimization_strategy.BayesianOptimizationSpec(nbr_of_points = 10*dim, 
                                                                        acquisition_type = acquisition_type, 
                                                                        batch_size = 1, 
                                                                        num_cores = 8, 
                                                                        basefilelocation = basefile, 
                                                                        bayes_iter = 20, 
                                                                        max_iter = 10,
                                                                        kernel = kernel)

                gl_strategy = optimization_strategy.GlobalStrategy(fcn = schweifel, 
                                                                bayes_spec = bayesSpec)
                gl_strategy.run()