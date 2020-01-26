from GPyOpt.methods import BayesianOptimization 
import sobol_seq
import numpy as np
import matplotlib.pyplot as plt
from pylab import savefig
from scipy import optimize

class GlobalStrategy(object):

    def __init__(self, fcn, bayes_spec):
        self.function = fcn
        self.bayes_spec = bayes_spec
        self.domain = []
        self.max_ack = []
        self.iterations = []

    def run(self):

        bounds = list(self.function.bounds)

        domain = []
        for i in range(0,self.function.dimensions):
            domain.append({'name': 'var_1', 'type': 'continuous', 'domain': bounds[i]})

        self.create_initial_points(bounds)

        if self.bayes_spec.kernel == None:
            self.optimizer = BayesianOptimization(f=self.function.evaluator, 
                                            domain=domain,
                                            model_type='GP',
                                            X=self.Xinit,
                                            Y=self.Yinit, 
                                            initial_design_type='random',
                                            acquisition_type=self.bayes_spec.acquisition_type,
                                            normalize_Y=True,
                                            exact_feval=False,
                                            acquisition_optimizer_type='lbfgs',
                                            model_update_interval=1,
                                            evaluator_type="random",
                                            batch_size=self.bayes_spec.batch_size,
                                            num_cores=self.bayes_spec.num_cores,
                                            verbosity=False,
                                            verbosity_model=False,
                                            maximize=False,
                                            de_duplication=False)       
        else:
            print("kernel is not None", self.bayes_spec.kernel)
            self.optimizer = BayesianOptimization(f=self.function.evaluator, 
                                            domain=domain,
                                            model_type='GP',
                                            X=self.Xinit,
                                            Y=self.Yinit, 
                                            initial_design_type='random',
                                            acquisition_type=self.bayes_spec.acquisition_type,
                                            normalize_Y=True,
                                            exact_feval=False,
                                            acquisition_optimizer_type='lbfgs',
                                            model_update_interval=1,
                                            evaluator_type="random",
                                            batch_size=self.bayes_spec.batch_size,
                                            num_cores=self.bayes_spec.num_cores,
                                            verbosity=False,
                                            verbosity_model=False,
                                            maximize=False,
                                            de_duplication=False, 
                                            kernel = self.bayes_spec.kernel)
        # run the optimization method                
        for i in range(1,int(self.bayes_spec.bayes_iter/self.bayes_spec.max_iter)+1):
            self.optimizer.run_optimization(self.bayes_spec.max_iter,
                                    eps=1e-08,
                                    verbosity=True,
                                    save_models_parameters=True,
                                    )
            self.max_ack.append( (abs(self.optimizer.acquisition.optimize()[1])[0])[0] )
            self.iterations.append(self.bayes_spec.max_iter*i)

        # Save information 
        self.optimizer.plot_acquisition(filename=self.bayes_spec.basefilelocation + 'acquisition.png')
        self.optimizer.plot_convergence(filename=self.bayes_spec.basefilelocation + 'convergence.png')
        self.optimizer.save_evaluations(evaluations_file=self.bayes_spec.basefilelocation + "evals" +  ".txt")
        self.optimizer.save_models(models_file=self.bayes_spec.basefilelocation + "models"  + ".txt")
        self.optimizer.save_report(report_file=self.bayes_spec.basefilelocation + "report" + ".txt")
        
        # print information of how far the best found solution is from the optimal solution 
        self.quality_solution()
        plt.show()
        
        
    def create_initial_points(self, bounds):
        dim = self.function.dimensions
        self.Xinit = sobol_seq.i4_sobol_generate(dim, self.bayes_spec.nbr_of_points + 1)[1:]
        

        for i in range(0,self.bayes_spec.nbr_of_points):
            for j in range(0, dim):
                lenint = (bounds[j])[1] - (bounds[j])[0]
                maxint = (bounds[j])[1]
                self.Xinit[i][j] *= lenint 
                self.Xinit[i][j] -= maxint

        self.Yinit = np.zeros( (10*dim, 1) )
        tmp = 0
        for xinit in self.Xinit:
            self.Yinit[tmp] = self.function.evaluator(xinit)
            tmp +=1

    def quality_solution(self):

        print("---------------- QUALITY of sol------------------")
        print ('Value, Best found: ', self.optimizer.x_opt, 'Argument, Best found: ', self.optimizer.fx_opt)
        print('Value, Global optimum', self.function.global_optimum,'Argument, global optimum: ',  self.function.fglob)
        print("-------------------------------------------------", self.bayes_spec.basefilelocation)
        f = open(self.bayes_spec.basefilelocation + 'results.csv', 'w')
        f.write('Best found Argument(BayesOpt),')
        for x in self.optimizer.x_opt:
            f.write(str(x) + ',',  )
        f.write('\n')
        f.write('Best found Value(BayesOpt),')
        f.write( str(self.optimizer.fx_opt) + '\n')
        
        f.write('Argument(global optimum),')
        for x in self.function.global_optimum:
            f.write(str(x) + ',' ) 
        f.write('\n')
        f.write('Value(global optimum),')
        f.write(str(self.function.fglob) + '\n')
        f.close()

        plt.figure("Acquisition convergence plot")
        plt.plot(np.log10(self.max_ack), '--bo')
        plt.xlabel('')
        plt.ylabel('log10| max acquisition (x) | ')
        savefig(self.bayes_spec.basefilelocation + 'max_acquisition.png')
        plt.close()
        plt.show()

        f = open(self.bayes_spec.basefilelocation + 'max_acquisition_results.csv', 'w')
        tmp =0
        for x in self.max_ack:
            f.write(str(self.iterations[tmp]) + ',' +  str(x)+ '\n')
            tmp +=1
        f.close()

    def compare_to_pure_Nelder_mead(self):
        bounds = list(self.function.bounds)        
        self.create_initial_points(bounds)
        fobs = []
        xobs = []
        self.Yinit = np.zeros( (10*self.function.dimensions, 1) )
        tmp = 0
        for xinit in self.Xinit:
            self.Yinit[tmp] = self.function.evaluator(xinit)
            tmp +=1
        nbr_fcn_evals = 0
        for i in self.Xinit:
            result_tmp = optimize.minimize(self.function.evaluator,
                                            i, 
                                            method='Nelder-Mead', 
                                            tol=1e-8,
                                            options = {'xatol' : 0.0001, 
                                                        'fatol': 0.0001, 
                                                        'disp' : False, 
                                                        'adaptive' : True})
            fobs.append(result_tmp.fun)
            xobs.append(result_tmp.x)
            nbr_fcn_evals += result_tmp.nfev        
        return xobs, fobs, nbr_fcn_evals
        

class BayesianOptimizationSpec(object):
    ''' Dummy class for determining parameters for bayesian optimization '''

    def __init__(self, nbr_of_points, 
                acquisition_type, batch_size, 
                num_cores, 
                basefilelocation, 
                bayes_iter, 
                max_iter, 
                kernel=None):
        self.nbr_of_points = nbr_of_points
        self.acquisition_type = acquisition_type
        self.batch_size = batch_size
        self.num_cores = num_cores
        self.basefilelocation = basefilelocation
        self.bayes_iter = bayes_iter
        self.max_iter = max_iter
        self.kernel = kernel


class LocalStrategy(object):
    ''' Options for running a local optimization algorithm '''
    def __init__(self,
                function, 
                BayesianOptimizationSpec):
        self.function = function
        self.BayesianOptimizationSpec = BayesianOptimizationSpec

    def run_nelder_mead(self):
        [x0global, y0global] = self.readInitialPoints()
        self.result = optimize.minimize(self.function.evaluator,
                                        x0global,
                                        method='Nelder-Mead', 
                                        tol=1e-8,
                                        options = {'xatol' : 0.0001, 
                                                    'fatol': 0.0001, 
                                                    'disp' : False, 
                                                    'adaptive' : True})
        
        print('Number of fcn evals', 'local optimization\t', 'Global optimization', 'Global optima')
        print (self.result.nfev, self.result.fun,'\t', y0global, '\t', self.function.fglob, 'Nelder-Mead')
        #print (self.result.x  ,'\t', x0global)

        return self.result.nfev, self.result.fun, y0global

    def run_BFGS(self):
        [x0global, y0global] = self.readInitialPoints()
        self.result = optimize.minimize(self.function.evaluator,
                                        x0global,
                                        method='BFGS', 
                                        tol=1e-8,
                                        options = {'gtol': 1e-05,  
                                                    'eps': 1.4901161193847656e-08, 
                                                    'maxiter': None, 
                                                    'disp': False, 
                                                    'return_all': False})
        
        #print('Number of fcn evals', 'local optimization\t', 'Global optimization')
        print (self.result.nfev, self.result.fun,'\t', y0global, 'BFGS')
        #print (self.result.x  ,'\t', x0global)

    def run_trust_region(self):
        [x0global, y0global] = self.readInitialPoints()
        self.result = optimize.minimize(self.function.evaluator,
                                        x0global,
                                        method='trust-constr', 
                                        tol=1e-8,
                                        options = {'xtol': 1e-08, 
                                             'gtol': 1e-08, 
                                            'barrier_tol': 1e-08, 
                                            'sparse_jacobian': None, 
                                            'maxiter': 1000, 'verbose': 0, 
                                            'finite_diff_rel_step': None, 
                                            'initial_constr_penalty': 1.0, 
                                            'initial_tr_radius': 1.0, 
                                            'initial_barrier_parameter': 0.1, 
                                            'initial_barrier_tolerance': 0.1,
                                            'factorization_method': None, 
                                            'disp': False})
        #print ('trust-constr')
        #print('Number of fcn evals', 'local optimization\t', 'Global optimization')
        print (self.result.nfev, self.result.fun,'\t', y0global, 'trust-constr')
        #print (self.result.x  ,'\t', x0global)

    def printLocalOptimizerResults(self):
        f = open(self.BayesianOptimizationSpec.evaluations_file + 'local_results.txt', 'w')
        f.write (str(self.result.__dict__))


    def readInitialPoints(self):
        f = open(self.BayesianOptimizationSpec.basefilelocation + 'results.txt', 'r')
        
        while True:
            line = f.readline()
            
            if not line:
                break
            elif "Bayes result" in line:
                data = (f.readline()).strip(']').strip('[').strip('\n').strip(' ')
                data = data.replace(']', '')
                
                
                data = data.split( '\t')
                #xdata = data[0].split('  ')
                #y0 = float(data[1])
                x0y0 = [float(xi) for xi in data]
                y0 = x0y0[-1]
                x0 = x0y0[:-1]

        f.close()

        return x0, y0


    def read_BayesOptData(self, t, c):
        evals = open(self.BayesianOptimizationSpec.basefilelocation + 'evals.txt')
   

        print(evals.readline())
        fcn_eval = []
        Y = []
        X = []
        Ybest_tmp = 10**5
        Y_best = []
        X_old = np.ones((self.function.dimensions, 1))
        distance = []
        while True:
            line = evals.readline().strip('\n').strip(' ')
            #print(line)
            
            if not line:
                break
            else:
                data = line.split('\t')
                
                data = [float(i) for i in data]
                fcn_eval.append(data[0])
                Ytmp = data[1]
                Xtmp = data[2:]

                X_new = np.transpose(np.array([Xtmp]))
                dx = X_old - X_new 
                dx = np.multiply(dx, dx)
                dx = np.sqrt(np.sum(dx))
                distance.append(dx)

                if (Ybest_tmp > Ytmp ):
                    Ybest_tmp = Ytmp
                Y_best.append(Ybest_tmp)

                X_old =X_new
        evals.close()
        #  determine how to color:        
        plt.figure("dx")
        plt.plot(fcn_eval, distance, '--' +  t + c)
        plt.xlabel('Function evation')
        plt.ylabel(' | xn - xn+1|')
        savefig(self.BayesianOptimizationSpec.basefilelocation + 'convergence_dx_ALL' + self.BayesianOptimizationSpec.acquisition_type + '.png')


        plt.figure("Ybest")
        plt.plot(fcn_eval, Y_best, '--' + t + c)
        plt.xlabel('Function evaluation')
        plt.ylabel(' Best Y')
        savefig(self.BayesianOptimizationSpec.basefilelocation + 'convergence_Y_best_ALL' + self.BayesianOptimizationSpec.acquisition_type + '.png')    

        max_acquisition_results = open(self.BayesianOptimizationSpec.basefilelocation + 'max_acquisition_results.txt')
        iteration = []
        max_acquisition = []
        while True:
            line = max_acquisition_results.readline().strip('\n').strip(' ')
            if not line:
                break
            else:
                data = line.split('\t')
                data = [float(i) for i in data]
                iteration.append(data[0])
                max_acquisition.append(data[1])
        max_acquisition_results.close()

        plt.figure("max_acquisition")
        plt.plot(iteration, np.log10(max_acquisition), '--' + t + c)
        print (t, c)
        plt.xlabel('Iteration')
        plt.ylabel('log10 |Max acquisition|')
        
        savefig(self.BayesianOptimizationSpec.basefilelocation + 'convergence_max_acquisition_ALL' + self.BayesianOptimizationSpec.acquisition_type + '.png')
        return fcn_eval, Y_best, distance, iteration, max_acquisition
        



        







