#!/users/marsve/my-local4/bin/python3
# -*- coding: utf-8 -*-
from bayes_opt import BayesianOptimization
from GPyOpt.methods import BayesianOptimization as  BayesianOptimizationGPyOpt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib import gridspec
from bayes_opt.util import UtilityFunction, Colours, acq_max
import GPy
import time
#from pyqaoa.qaoa import *
#from pyqaoa.optimization import *
#from pyqaoa.util import *
import math as m
import sys
import os
from pylab import savefig
import sobol_seq
import random

import numpy as np
import math as m
from   pyqaoa.convergence_gpyopt import ConvergenceGPyOpt 
from pyqaoa.go_benchmark import Ackley, Deceptive, Hartmann3, Hartmann6, Rastrigin, Rosenbrock, Schwefel26, Sphere

class ackleyfcn():

    def __init__(self, dim):
        """
        Simply initiating the function
        """
        self.fglob = 0

    def toString(self):
        return 'ackley'

    def evaluator(self, x_tuple):
        
        if x_tuple.__class__.__name__ in ('tuple'):
            x = x_tuple[0]
        elif len(np.shape(x_tuple)) == 2:
            x = x_tuple[0]
        else:
            x = x_tuple
        #print("np.shape(x)", np.shape(x), 'len(x)', len(x))
        d = len(x)
        
        A = 20.0
        B = 0.5
        C = np.pi * 2
        
        sum_sq_term = -A * np.exp(-B * np.sqrt(sum(x*x) / d))
        cos_term = -np.exp(sum(np.cos(C*x) / d))
        return A + np.exp(1) + sum_sq_term + cos_term
    
  

def bayesian_optimization_gpyopt_general(fobj, dim, bounds, basefile,  convergence="all", 
                                        kappa=10**(-9), computational_budget=10000, simple_regret_e=10**(-9), 
                                        acquisition_type='EI',
                                        batch_size=1, 
                                        max_iter = 10, 
                                        num_cores=2):
    """ This function optimizes f(x)
    Returns
    -------
    convergence= "all", maxAcqusition", "simpleRegret", "computationalBudget"
    """

    # define the domain for the Bayesian optimization object
    domain = []
    for i in range(0,dim):
        domain.append({ 'domain' : bounds[i]})
    print ('domain: \n', domain)
    # create the optimizer object
    # TODO : calculate initial datapoints randomly
    
    Xinit = sobol_seq.i4_sobol_generate(dim, 10*dim + 1)[1:]
    #print(Xinit)
    #print(bounds)
    #print(sobol_seq.i4_sobol_generate(dim, 10*dim + 1))
    for i in range(0,dim*10):
        for j in range(0, dim):
            lenint = (bounds[j])[1] - (bounds[j])[0]
            maxint = (bounds[j])[1]
            #print(i, '  ', j, np.shape(Xinit), lenint, maxint, Xinit[i][j] )
            
            Xinit[i][j] *= lenint 
            Xinit[i][j] -= maxint
            #else:
            #    Xinit[i, j:j+1] *=  ((bounds[j])[1] - (bounds[j])[0])/2.0
                
    
    '''
    Xinit = np.zeros((10*dim, dim))

    for ii in range(0, dim):
        for kk in range(0, 10*dim):
            Xinit[kk][ii] = random.uniform((bounds[ii])[0], (bounds[ii])[1])
    '''
    #print(np.shape(Xinit) , '  ' , 3*dim, '\n', Xinit )
    Yinit = np.zeros( (10*dim, 1) )
    tmp = 0
    for xinit in Xinit:
        #print(xinit)
        Yinit[tmp] = fobj.evaluator(xinit)
        tmp +=1

    #plt.plot(Xinit[:, 0:p], Xinit[:, p:2*p], 'bo')
    #plt.show(block=False)
    # probe the optimization
    kernel = GPy.kern.Matern32(dim,ARD=True, variance=1.)
    print (acquisition_type)
    optimizer = BayesianOptimizationGPyOpt(f=fobj.evaluator,
                                        domain=domain,
                                        model_type='GP',
                                        X=Xinit,
                                        Y=Yinit,
                                        #initial_design_numdata=5*dim,
                                        initial_design_type='random',
                                        acquisition_type=acquisition_type,
                                        normalize_Y=True,
                                        exact_feval=False,
                                        acquisition_optimizer_type='lbfgs',
                                        model_update_interval=1,
                                        evaluator_type="random",
                                        batch_size=batch_size,
                                        num_cores=num_cores,
                                        verbosity=False,
                                        verbosity_model=False,
                                        maximize=False,
                                        de_duplication=False,
                                        ARD=True)
                                        #cost_withGradients=None,
                                  
    #print('\n optimizer.model.__dict', optimizer.model.__dict__)
    #print("----------------------------------------------------------------------------------")
    #print('\n optimizer.model.model.__dict', optimizer.model.model.__dict__)
    #print("----------------------------------------------------------------------------------")
    # optimization parameters
    max_time="undef"
    eps=1e-08
    # Optimize 
    nbr_restarts = 0
    convBay = ConvergenceGPyOpt(kappa=kappa, fmin = fobj.fglob)
    
    while True :
        optimizer.run_optimization(max_iter=max_iter,
                                eps=1e-08,
                                verbosity=False,
                                save_models_parameters=True,
                                )
        nbr_restarts +=1
        
        convBay.update(optimizer, dim)

        if  (convergence == "all" or convergence == "simpleRegret" ) and convBay.simple_regret < simple_regret_e :
            break
        if  (convergence == "all" or convergence == "maxAcqusition") and convBay.is_gpyopt_converged(optimizer, dim):
            break
        if  (convergence == "all" or convergence == "computationalBudget") and (nbr_restarts*max_iter >= computational_budget):
            break

    print(" Number of restarts of Bayesian optimization restarts: ", nbr_restarts)

    instance = fobj.toString()
    # Save the data . users need to currently modify this themself. Should be changed in the future
    #"/Users/turbotanten/Documents/GitHub/pyqaoa/gpyoptResults"
    simulation = "_dim" + str(dim) + "_max_it" + str(max_iter) + "_eps" + str(eps) + \
        "max_t" + str(max_time) + "_inst" + instance

    #os.system( "mkdir " +  basefile )
    optimizer.save_evaluations(evaluations_file=basefile + "evals" + simulation + ".txt")
    optimizer.save_models(models_file=basefile + "models"  + simulation +".txt")
    optimizer.save_report(report_file=basefile + "report" + simulation +".txt")

    # plot if f \in R^2
    if dim <= 2:
        optimizer.plot_acquisition(filename=basefile + "acquisition" +  simulation + ".png")
        
    # plot convergence
    optimizer.plot_convergence(filename=basefile + "convergence" +  simulation + ".png" )
    
    convBay.plot_acquisition_convergence(filename=basefile + "convergence_acquisition" +  simulation + ".png")
    convBay.plot_simple_regret_convergence(filename=basefile + "convergence_simple_regret" +  simulation + ".png")
    convBay.save_acquisition_convergence_report(filename=basefile + "report_convergence_acquisition" +  simulation + ".txt")
    convBay.save_X_best_Y_best(optimizer, filename=basefile + "report_X_best_Y_best" +  simulation + ".txt")
    
    # ------------- Naive attempt to calculate the trust region ----------------------------------
    # ----------------- attempt nelder mead from X_best, Y_best -----------------------------------
    from scipy import optimize
    x0 = optimizer.x_opt

    #lb = optimizer.x_opt - 0.5 
    #ub = optimizer.x_opt + 0.5
    #bounds_sci = optimize.Bounds(lb, ub)
    result = optimize.minimize(fobj.evaluator,
                                x0,
                                method='Nelder-Mead', 
                                tol=1e-6,
                                options = {'xatol' : 0.0001, 
                                            'fatol': 0.0001, 
                                                'disp' : True, 
                                                'adaptive' : True})
    #print("-----------------------------" , result, "--------------------------------")
    fnelder = open(basefile + "report_Nelder_after_bay" +  simulation + ".txt", 'w')
    fnelder.write(str(result))
    
    print('optimizer.fx_opt: ', optimizer.fx_opt, ' optimizer.x_opt: ', optimizer.x_opt)
    print('result.fun: ', result.fun, ' result.x: ', result.x)
    print()
    
    '''
    result = optimize.minimize(fobj.evaluator,
                                x0,
                                method='L-BFGS-B', 
                                tol=1e-6,
                                options = {'disp': True, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 
                                'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 
                                'iprint': -1, 'maxls': 20})
    #print("-----------------------------" , result, "--------------------------------")
    fnelder = open(basefile + "report_L-BFGS-B_after_bay" +  simulation + ".txt", 'w')
    fnelder.write(str(result))
    
    print('optimizer.fx_opt: ', optimizer.fx_opt, ' optimizer.x_opt: ', optimizer.x_opt)
    print('result.fun: ', result.fun, ' result.x: ', result.x)
    
    return optimizer.x_opt
    '''


#------------------ run  for testfunctions ----------------------------
'''
for dim in (10, 6):
    #fobj = Ackley(dim)#ackleyfcn(dim)
    print(dim)
    fobj = Rastrigin(dim)#Hartmann6(dim)#Deceptive(dim)#ackleyfcn(dim)
    print("\n \n #--------- Results for  function ",  fobj.toString(), "with dim: ", dim , "-----to=10e-6---------------#" )
    bounds = list(fobj.bounds)

    print (bounds)
    kappa = 10**(-9)
    
    
    # ------------set the kernel for the GP -------------
    for acquisition_type in ('EI', 'MPI', 'LCB' ):
        print ("\n result for " + acquisition_type + "-------------------------------------\n" )
        computational_budget = 250
        batch_size =  1
        max_iter   =  25
        num_cores   = 2
        basefile00 = "/users/marsve/QUANTUM_COMPUTING_JEPPESEN_AIRLINE_PLANNING_PHD_2019_2025/" + \
            "code/PyQAOA/gpyoptBenchResults20190629/"
        basefile0= basefile00+ fobj.toString() + "/"
        basefile=basefile0 + "sobol_no_middle" + "compBud" + str(computational_budget) + acquisition_type + "/"
        if not os.path.isdir( basefile00 ):
            os.system( "mkdir " +  basefile00 )
        if not os.path.isdir( basefile0 ):
            os.system( "mkdir " +  basefile0 )
        if not os.path.isdir( basefile):
            os.system( "mkdir " +  basefile )
        x_opt = bayesian_optimization_gpyopt_general(fobj, dim, bounds, 
                                                    basefile,
                                                    convergence="computationalBudget", 
                                                    kappa=10**(-9), 
                                                    computational_budget=computational_budget, 
                                                    simple_regret_e=10**(-9),
                                                    acquisition_type= acquisition_type )
        print ("x_opt = ", x_opt)


    
    #plt.show()

    print ("\n Correct answer: ", fobj.fglob, '  ', fobj.global_optimum)
    
'''

#----------------- run  for testfunctions ----------------------------

for dim in (4,8):
    #fobj = Ackley(dim)#ackleyfcn(dim)
    print(dim)
    fobj = Ackley(dim)#Hartmann6(dim)#Deceptive(dim)#ackleyfcn(dim)
    print("\n \n #--------- Results for  function ",  fobj.toString(), "with dim: ", dim , "-----to=10e-6---------------#" )
    bounds = list(fobj.bounds)

    print (bounds)
    kappa = 10**(-9)
    
    
    # ------------set the kernel for the GP -------------
    for acquisition_type in ('EI','MPI', 'LCB' ):
        print ("\n result for " + acquisition_type + "-------------------------------------\n" )
        computational_budget = 1000
        batch_size =  1
        max_iter   =  10
        num_cores   = 8
        basefile00 = "/users/marsve/QUANTUM_COMPUTING_JEPPESEN_AIRLINE_PLANNING_PHD_2019_2025/" + \
            "code/PyQAOA/gpyoptBenchResults20190629/"
        basefile0= basefile00+ fobj.toString() + "/"
        basefile=basefile0 + "sobol_no_middle" + "compBud" + str(computational_budget) + acquisition_type + "batch" + str(batch_size) + "/"
        if not os.path.isdir( basefile00 ):
            os.system( "mkdir " +  basefile00 )
        if not os.path.isdir( basefile0 ):
            os.system( "mkdir " +  basefile0 )
        if not os.path.isdir( basefile):
            os.system( "mkdir " +  basefile )
        x_opt = bayesian_optimization_gpyopt_general(fobj, dim, bounds, 
                                                    basefile,
                                                    convergence="computationalBudget", 
                                                    kappa=10**(-9), 
                                                    computational_budget=computational_budget, 
                                                    simple_regret_e=10**(-9),
                                                    acquisition_type= acquisition_type,
                                                    batch_size = batch_size, 
                                                    max_iter = max_iter, 
                                                    num_cores = num_cores )
        print ("x_opt = ", x_opt)


    
    #plt.show()

    print ("\n Correct answer: ", fobj.fglob, '  ', fobj.global_optimum)
    
