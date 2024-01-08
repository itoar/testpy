import numpy as np
import autograd.numpy as np
from autograd import grad, jacobian, elementwise_grad
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#def func(x,y):
#    return x**2 + y**2

def func(c):
    x, y = c
#    return (x - 1)**2 + (y + 3)**2
#    return (x - 1)**4 + (y + 3)**4

    return (x**2 + y - 11)**2 + ( x + y**2 - 7)**2
#    return (1 - x)**2 + 100*( y - x**2)**2

class ploty:
    def __init__(self):
        
        x = np.arange(-10, 10, 0.05) #x軸の描画範囲の生成。0から10まで0.05刻み。
        y = np.arange(-10, 10, 0.05) #y軸の描画範囲の生成。0から10まで0.05刻み。

        self.X, self.Y = np.meshgrid(x, y)
        self.Z = np.zeros_like(self.X)

    def setFunc(self, f):
        z = np.array([self.X, self.Y])
        self.Z = func(z)

#        self.Z = func(self.X, self.Y)
    def colorplot(self):

        plt.pcolormesh(self.X, self.Y, self.Z, cmap='hsv') # 等高線図の生成。cmapで色付けの規則を指定する。
        plt.pcolor(self.X, self.Y, self.Z, cmap='hsv') # 等高線図の生成。cmapで色付けの規則を指定する。
        pp=plt.colorbar (orientation="vertical") # カラーバーの表示 
        pp.set_label("Label", fontname="Arial", fontsize=24) #カラーバーのラベル

    def contourplot(self):
        cont=plt.contour(self.X, self.Y, self.Z,  5, vmin=-1,vmax=1, colors=['black'])

    def plot_trajectory(self, k_max, x_history):
        for i in range(k_max):
            plt.plot(x_history[i, 0], x_history[i, 1], "-o", color = "red", markeredgecolor = "black", markersize = 5, markeredgewidth = 0.5);

    def plot_trajectory2(self, k_max, x_history):
        for i in range(k_max):
            plt.plot(x_history[i, 0], x_history[i, 1], "-o", color = "blue", markeredgecolor = "black", markersize = 5, markeredgewidth = 0.5);


    def plot(self):
        plt.xlabel('X', fontsize=24)
        plt.ylabel('Y', fontsize=24)

        plt.show()

def gradient_descent(f, init_x, lr, k_max):
    x = init_x
    x_history = []
    g = jacobian(f)
    print(g(x))
    for i in range(k_max):

        x -= lr * g(x)
        x_history.append( x.copy() )
    return x, np.array(x_history)

def newton_method(f, init_x, lr, k_max):
    x = init_x
    x_history = []
    g = jacobian(f)
    h = jacobian(g)
    for i in range(k_max):
        print(h(x))
        h_inv = np.linalg.inv(h(x))
        x -= lr * h_inv@g(x)
        x_history.append( x.copy() )
    return x, np.array(x_history)

p = ploty()
p.setFunc(func)
p.colorplot()
p.contourplot()

lr = 1.0                          #learning rate
k_max = 100                         #イタレーション数
init_x = np.array([-1.0, 0.0])      #初期位置
x, x_history = newton_method(func, init_x, lr, k_max)
p.plot_trajectory(k_max,x_history)

lr_g = 0.01                          #learning rate
kg_max = 100                         #イタレーション数
init_xg = np.array([-1.0, 0.0])      #初期位置
xg, xg_history = gradient_descent(func, init_xg, lr_g, kg_max)
p.plot_trajectory2(kg_max,xg_history)

x0 = np.array([-1.0, 0.0])
# 最適化の実行
result = minimize(func, x0, method='BFGS')

print(x)
print(result)
p.plot()
