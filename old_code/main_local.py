from global_optimization import go_benchmark
from global_optimization import optimization_strategy
import os
import matplotlib.pyplot as plt

acquisition_type = 'MPI'
kernel = 'Matern52'
dims = ( 1, 2)
acquisition_types = ('LCB', 'EI', 'MPI')
kernels_name = ('Matern52','Matern32', 'Exponential')
colvec =  ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g')
tvec = ('.', '.', '.')*3 #, '.', '.', '.', 'v', 'v', 'v'  )
nfev_means = []
for dim in dims:
    #tmp = -1
    tmp =0
    nfev_mean = 0
    print ("--------------------test for dim ", dim, "------------------")
    for acquisition_type in acquisition_types:
        #tmp +=1
        #plt.close('all')
        for kernel_name in kernels_name:
            
            tmp +=1
            print ()
    
            schweifel = go_benchmark.Schwefel26(dim)

            #base = './test190911_bayes_acq_type' + acquisition_type + 'EIkernel' + str(kernel)'/'
            #basefile = base + '/dim' + str(dim) + 'function_' + 'schweifel26' + '/'
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
            assert False, bayesSpec

            local_optimizer = optimization_strategy.LocalStrategy(schweifel, bayesSpec)
            #print (local_optimizer.readInitialPoints( ) )
            
            [nfev, fun, yglobal] = local_optimizer.run_nelder_mead()

            nfev_mean +=nfev
            print ('nfev', nfev)
            #local_optimizer.run_trust_region()
            #local_optimizer.run_BFGS()

            #local_optimizer.read_BayesOptData(tvec[tmp], colvec[tmp])
    nfev_means.append(int(nfev_mean/tmp))
#plt.show()
#print(basefile)

print ('nfev_mean', nfev_means)