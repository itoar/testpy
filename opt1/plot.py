import numpy as np
import autograd.numpy as np
from autograd import grad, jacobian, elementwise_grad
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.patches import Ellipse
from scipy.optimize import minimize


class PLOTY:
    def __init__(self):
        x = np.arange(-10, 10, 0.05) #x軸の描画範囲の生成。0から10まで0.05刻み。
        y = np.arange(-10, 10, 0.05) #y軸の描画範囲の生成。0から10まで0.05刻み。

        self.X, self.Y = np.meshgrid(x, y)
        self.Z = np.zeros_like(self.X)
        self.fig, self.ax = plt.subplots()

        self.images = []
        
    def setFunc(self, f):
        z = np.array([self.X, self.Y])
        self.Z = f(z)

    def colorplot(self):
        self.ax.pcolormesh(self.X, self.Y, self.Z) # 等高線図の生成。cmapで色付けの規則を指定する。

    def colorplotlog(self):
        self.ax.pcolormesh(self.X, self.Y, np.log(self.Z)) # 等高線図の生成。cmapで色付けの規則を指定する。

    def scatter_plot(self, X):
        return self.ax.scatter(X[:, 0], X[:, 1], c="firebrick", ec="white")

    def add_patch(self, ell):
        return self.ax.add_patch(ell)

    def add_images(self, im):
        self.images.append(im)

    def write_animation(self):
        savepath = r"C:\sandbox\python\opt1\test.gif"
        ani = animation.ArtistAnimation(self.fig, self.images, interval=400)
        ani.save(savepath, writer='pillow')
    
    def plot(self):
        plt.show()