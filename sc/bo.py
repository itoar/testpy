import GPy
import GPyOpt
import numpy as np
import matplotlib.pyplot as plt

def function(x):
    #x = np.floor(16*x)
    return -(1.4 - 3.0 * x) * np.sin(18.0 * x)

class Optimizer:
    def __init__(self, bound):
        self.bound_min = bound[0]
        self.bound_max = bound[1]

#        self.input_param = np.array([0.5*(self.bound_min + self.bound_max)])
        self.input_param = np.array([0.5])
        self.input_param = self.input_param.reshape(-1,1)

        self.values = function(self.input_param)
        self.values = self.values.reshape(-1,1)

        self.trial = 0

    def initialize(self):
        domain = [{'name': 'x', 'type': 'continuous', 'domain': (self.bound_min, self.bound_max)}]

        self.bo = GPyOpt.methods.BayesianOptimization(f=self.objective_function,
                                                model_type='GP',
                                                domain=domain,
                                                acquisition='EI',
                                                maximize = False,
                                                X=self.input_param,
                                                Y=self.values)


    
    def objective_function(self, x):
                
        self.trial += 1
        
        self.input_param = np.append(self.input_param,x)
        self.input_param = self.input_param.reshape(-1,1)
        
        y = function(x)
        self.values = np.append(self.values,y)
        self.values = self.values.reshape(-1,1)

        print(f'trial : {self.trial}')
        print('x :',x)
        print('y :',y)

        return y
    
    def run(self,iter = 10):
        self.bo.run_optimization(max_iter=iter)

    def plot(self):
        x_pred = np.linspace(self.bound_min, self.bound_max, 100).reshape(-1, 1)
        y_pred, y_std = self.bo.model.predict(x_pred)
        y_pred, y_std = self.bo.model.predict(x_pred)

        y_true = function(x_pred)

        plt.plot(x_pred, y_pred)
        plt.plot(x_pred, y_pred + 2 * y_std)
        plt.plot(x_pred, y_pred - 2 * y_std)
        plt.plot(x_pred, y_true)

        plt.grid(linestyle='--', linewidth=0.5, color='gray')
        plt.title('Prediction')

        plt.show()

        print("opt:", self.bo.x_opt)

bound = [0,1.2]
opt = Optimizer(bound)

opt.initialize()
opt.run(15)
opt.plot()