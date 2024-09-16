import matplotlib.pyplot as plt
import numpy as np

def Bernstein(t):
    J = np.array( [(1 - t)**3, 3*t*(1 - t)**2, 3*t**2*(1 - t), t**3 ])
    return J
def Bez(p, t):
    return p @ Bernstein(t)

def BezierInterpolate(P0,P1,data):
    t = np.linspace(0, 1, 100) #Bezier t
    if data.size == 0:
        length = (P1[0] - P0[0])/4
        w0 = length
        w1 = length
        rad0 = 0.0
        rad1 = 0.0
        P0_out = P0 + w0*np.array([np.cos(rad0), np.sin(rad0)])
        P1_in = P1 + w1*np.array([-np.cos(rad1), np.sin(rad1)])
        ini_B = np.vstack((P0 , P0_out, P1_in, P1)).T
        return ini_B
    elif data.shape[1] != 1:
        j = np.linspace(0, data.shape[0]+2, 4, True, dtype=int)
        ini_B = np.vstack((P0 , data[j[1] - 1], data[j[2] - 1], P1)).T
        data = np.vstack((P0, data, P1))
    else:
        #データ点が一つ
        ini_B = np.vstack((P0 , data.T, data.T, P1)).T
        data = np.vstack((P0, data.T, P1))
    sampled_data_x = data.T[0]
    sampled_data_y = data.T[1]
    B = ini_B
    for i in range(5):
        f = Bernstein(t)
        idx = np.abs(Bez(B, t).T[...,None][:,0] - sampled_data_x ).argmin(0)
        inv = np.linalg.pinv(f[:, idx] @ f[:, idx].T)
        x_ = inv @ (sampled_data_x * f[:, idx]).sum(1, keepdims=True)
        y_ = inv @ (sampled_data_y * f[:, idx]).sum(1, keepdims=True)
        new_B = np.c_[x_, y_].T # Bを更新
        B.T[1] = new_B.T[1]
        B.T[2] = new_B.T[2]
    return B

def plotBezier(B):
    t_ = np.linspace(0, 1, 100)
    plt.plot(*Bez(B, t_), c='r', lw=1)
    plt.plot(*(B.T[0]).T, marker='o', ls='--', lw=.5, c='k', alpha=.5)
    plt.plot(*(B.T[3]).T, marker='o', ls='--', lw=.5, c='k', alpha=.5)
    plt.plot(*(B.T[1]).T, marker='o', ls='--', lw=.2, c='r', alpha=.5)
    plt.plot(*(B.T[2]).T, marker='o', ls='--', lw=.2, c='r', alpha=.5)
    

## start and end points
P0 = np.array([0.0, 0.0])
P1 = np.array([5.0, 5.0])
P2 = np.array([10.0, 80.0])

## making data 
data = np.empty(0)
#d1 = np.array([2.0, 0.1])
#d2 = np.array([4.0, 1.0])
#d3 = np.array([4.5, 3.0])
#data = np.vstack((d1 , d2, d3))

B_param = BezierInterpolate(P0, P1, data)
B_param_1 = BezierInterpolate(P1, P2, data)

#plt.scatter(sampled_data_x, sampled_data_y, s=5)

plotBezier(B_param)
plotBezier(B_param_1)

plt.show()

print(*(B_param.T[0]).T)
