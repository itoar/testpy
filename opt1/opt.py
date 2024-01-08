import numpy as np
import autograd.numpy as np
from autograd import grad, jacobian, elementwise_grad
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.patches import Ellipse
from scipy.optimize import minimize


class CMAES:
    def __init__(self, centroid, sigma, lamda):
        self.dim = len(centroid)

        self.lamda = lamda if lamda else int(4 + 3*np.log(self.dim ))
        self.mu = int(np.floor(self.lamda/2.0))

        self.centroid = np.array(centroid, dtype=np.float64)
        self.c_m = 1.0

        weights = [np.log(0.5*(self.lamda + 1)) - np.log(i) for i in range(1, 1+self.mu)]
        weights = np.array(weights).reshape(1, -1)
        self.weights = weights / weights.sum()
        self.mu_eff = 1.0 / (self.weights**2).sum()

        #stepsize: evolution path and ratio
        self.sigma = float(sigma)
        self.p_sigma = np.zeros(self.dim)
        self.c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max( 0, np.sqrt((self.mu_eff - 1)/(self.dim + 1)) - 1 ) + self.c_sigma

        #covariance matrix
        self.C = np.identity(self.dim)
        self.p_c = np.zeros(self.dim)
        self.c_c = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.c_1 = 2.0 / ((self.dim+1.3)**2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1, 2.0 * (self.mu_eff - 2 + 1/self.mu_eff) / ((self.dim + 2) ** 2 + self.mu_eff) )

    def sample_new_population_of_search_points(self):
        Z = np.random.normal(0, 1, size=(self.lamda, self.dim))
        D2, B = np.linalg.eigh(self.C)
        D = np.sqrt(D2)
        BD = B @ np.diag(D) 

        Y = (BD @ Z.T).T
        X = self.centroid + self.sigma * Y
        
        return X

    def update(self, X, target, gen):
        old_centroid = self.centroid
        old_sigma = self.sigma

        ## Selection and recombination
        #fitnessesが上位muまでのindexを抽出
        elite_indices = np.argsort(target)[:self.mu]

        X_elite = X[elite_indices, :]
        Y_elite = (X_elite - old_centroid) / old_sigma

        X_w = np.matmul(self.weights, X_elite)[0]
        Y_w = np.matmul(self.weights, Y_elite)[0]

        self.centroid = (1 - self.c_m) * old_centroid + self.c_m * X_w

        ## Step-size control

        D2, B = np.linalg.eigh(self.C)
        D = np.sqrt(D2)
        inv_D = 1.0/D
        C_ = B @ np.diag(inv_D) @ B.T

        new_p_sigma = (1.0 - self.c_sigma)* self.p_sigma
        new_p_sigma += np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * C_ @ Y_w
        self.p_sigma = new_p_sigma

        E_norm = np.sqrt(self.dim) * (1 - 1/(4 * self.dim) + 1/(21 * self.dim **2))
        self.sigma = self.sigma * np.exp( (self.c_sigma / self.d_sigma) * (np.sqrt((self.p_sigma ** 2).sum()) / E_norm - 1) )

        ##Covariant matrix adaptation
        left = np.sqrt((self.p_sigma ** 2).sum()) / np.sqrt(1 - (1 - self.c_sigma) ** (2 * (gen+1)))
        right = (1.4 + 2 / (self.dim + 1)) * E_norm
        hsigma = 1 if left < right else 0
        d_hsigma = (1 - hsigma) * self.c_c * (2 - self.c_c)

        #update p_c
        new_p_c = (1 - self.c_c) * self.p_c
        new_p_c += hsigma * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * Y_w
        self.p_c = new_p_c

        #update C
        new_C = (1 + self.c_1 * d_hsigma - self.c_1 - self.c_mu) * self.C
        new_C += self.c_1 * np.outer(self.p_c, self.p_c)

        wyy = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            y_i = Y_elite[i]
            wyy += self.weights[0, i] * np.outer(y_i, y_i)
        new_C += self.c_mu * wyy
        
        self.C = new_C