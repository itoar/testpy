import opt
import plot

import numpy as np
from matplotlib.patches import Ellipse
from scipy.optimize import minimize

def func(c):
    x, y = c
#    return x**2 + y**2
#    return (x - 1)**2 + (y + 3)**2
#    return (x - 1)**4 + (y + 3)**4

    return (x**2 + y - 11)**2 + ( x + y**2 - 7)**2
#    return (1 - x)**2 + 100*( y - x**2)**2

def main():
    print("start")
    cmaes = opt.CMAES(centroid=[0.0, 0.0], sigma = 1.0, lamda=12)
    n_generations = 30
    p = plot.PLOTY()
    p.setFunc(func)
    p.colorplotlog()

    for gen in range(n_generations):

        X = cmaes.sample_new_population_of_search_points()
        Z = np.array([X[:, 0], X[:, 1]])
        fitnesses = func(Z)

        im_list = []
        im = p.scatter_plot(X)
        im_list.append(im)
        lambda_, v = np.linalg.eig(cmaes.C)
        lambda_ = np.sqrt(lambda_)
        for j in range(1, 4):
            ell = Ellipse(xy=(cmaes.centroid[0], cmaes.centroid[1]),
                          width=lambda_[0]*j*2*cmaes.sigma,
                          height=lambda_[1]*j*2*cmaes.sigma,
                          angle=np.rad2deg(np.arccos(v[0, 0])),
                          fc="none", ec="firebrick", ls="--")
            im = p.add_patch(ell)
            im_list.append(im)
        p.add_images(im_list)
        print(cmaes.centroid)
        #: パラメータ更新
        cmaes.update(X, fitnesses, gen)

    p.write_animation()
    print("end")

main()