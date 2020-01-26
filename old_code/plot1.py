from global_optimization import go_benchmark
from global_optimization import optimization_strategy
import os
import matplotlib.pyplot as plt
from pylab import savefig


kernel = 'Matern52'
dims = ( 1,)
acquisition_types = ('LCB', 'EI', 'MPI')
kernels_name = ('Matern52', 'Matern32', 'Exponential')
colvec =  ('b', 'g', 'r')*3#, 'c', 'm', 'y', 'k', 'b', 'g', 'b', 'g')
tvec = ('.', '.', '.')*3 #, '.', '.', '.', 'v', 'v', 'v'  )
fontsize = '15'

tmp = -1    
for acquisition_type in acquisition_types:
    for kernel_name in kernels_name:
        nfevs = []
        Yoptimal_BayNelders = []
        Yoptimal_Bays = []
        tmp +=1
        for dim in dims:
            schweifel = go_benchmark.Schwefel26(dim)
            base = './test190913_bayes_acq_type' + acquisition_type + \
            'kernel' + kernel_name + '/'
            basefile = base + '/dim' + str(dim) + 'function_' + 'schweifel26' + '/'
            
            bayesSpec = optimization_strategy.BayesianOptimizationSpec(nbr_of_points = 10*dim, 
                                                                    acquisition_type = acquisition_type, 
                                                                    batch_size = 1, 
                                                                    num_cores = 4, 
                                                                    basefilelocation = basefile, 
                                                                    bayes_iter = 200,
                                                                    max_iter = 10,
                                                                    kernel = kernel_name)
            

            local_optimizer = optimization_strategy.LocalStrategy(schweifel, bayesSpec)
            
            [nfev, Yoptimal_BayNelder, Yoptimal_Bay] = local_optimizer.run_nelder_mead()

            nfevs.append(nfev)
            Yoptimal_BayNelders.append(Yoptimal_BayNelder)
            Yoptimal_Bays.append(Yoptimal_Bay)
            ''' Possible local optimizers  '''
            #local_optimizer.run_trust_region()
            #local_optimizer.run_BFGS()
            #local_optimizer.read_BayesOptData(tvec[tmp], colvec[tmp])   
        plt.figure( acquisition_type)
        plt.title('Schwefel26' + ' - '+  acquisition_type)
        plt.plot(list(dims), Yoptimal_BayNelders, '-*' + colvec[tmp])
        plt.plot(list(dims), Yoptimal_Bays, '-o' + colvec[tmp])
        plt.xlabel('Dimension')
        plt.ylabel('f*')
        plt.legend( loc='upper right', ncol=1, borderaxespad=0.1,  fontsize=fontsize)
    savefig(base + acquisition_type + '.png')

plt.show()
    