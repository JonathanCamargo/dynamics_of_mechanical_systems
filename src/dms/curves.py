import scipy
import numpy as np
import matplotlib.pyplot as plt

# Center and normalize the trajectory
def NormalizeTrajectory(traj):    
    center=traj.mean(axis=0)    
    traj=traj-center
    svd=np.linalg.svd(traj)     
    scale=np.max(svd.S)
    traj=np.matmul(svd.Vh,traj.T).T/scale
    rot=svd.Vh
    return traj,center,scale,rot

class ClosedSpline:
    def __init__(self, points,n=100):
        self.points = np.array(points)
        self.tt = np.linspace(0, 1, n)
        self.interp,u = scipy.interpolate.splprep([self.points[:,0],self.points[:,1]],s=0,per=True)
        self.xx,self.yy = scipy.interpolate.splev(self.tt,self.interp)
        
    def plot(self,ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.xx, self.yy)
        ax.plot(self.points[:,0], self.points[:,1], 'k.', markersize=10)
        ax.plot(self.xx, self.yy, 'b.', markersize=2)
        ax.axis('equal')

def CurveLength(xx,yy):   
    return np.sum(np.sqrt(np.diff(xx)**2+np.diff(yy)**2))

def CompareCurves(xx,yy,xx_star,yy_star,length_weight=0.01):
    kdt=scipy.spatial.KDTree(np.array([xx_star,yy_star]).T)
    er=0
    for i,xi in enumerate(xx):
        distance,idx=kdt.query([xi,yy[i]])
        er=er+distance
    l=CurveLength(xx,yy)
    l_star=CurveLength(xx_star,yy_star)
    return er/len(xx)+length_weight*np.abs(l-l_star)

# Create a circle with center and radius
def circle(center, radius, n=100):
    tt = np.linspace(0, 2*np.pi, n)
    xx = center[0] + radius*np.cos(tt)
    yy = center[1] + radius*np.sin(tt)
    return xx, yy

def elipse(center, a, b, n=100):
    tt = np.linspace(0, 2*np.pi, n)
    xx = center[0] + a*np.cos(tt)
    yy = center[1] + b*np.sin(tt)
    return xx, yy

def cubic_interp(x0, v0, xf, vf):
    """Return a cubic interpolator f(tt) with f(0)=x0, f'(0)=v0, f(1)=xf, f'(1)=vf.

    Works for scalars and R^n vectors. For vector inputs, f(tt) with
    tt of shape (N,) returns an array of shape (N, dim).
    """
    a0 = np.asarray(x0)
    a1 = np.asarray(v0)
    a2 = 3*(np.asarray(xf) - a0) - 2*a1 - np.asarray(vf)
    a3 = 2*(a0 - np.asarray(xf)) + a1 + np.asarray(vf)
    def f(tt):
        tt = np.asarray(tt, dtype=float)
        if a0.ndim == 0:
            return a0 + a1*tt + a2*tt**2 + a3*tt**3
        t = tt[..., np.newaxis]
        return a0 + a1*t + a2*t**2 + a3*t**3
    return f

def quintic_interp(x0, v0, a0, xf, vf, af):
    """Return a quintic interpolator f(tt) with f(0)=x0, f'(0)=v0, f''(0)=a0,
    f(1)=xf, f'(1)=vf, f''(1)=af.

    Works for scalars and R^n vectors. For vector inputs, f(tt) with
    tt of shape (N,) returns an array of shape (N, dim).
    """
    x0 = np.asarray(x0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    a0 = np.asarray(a0, dtype=float)
    xf = np.asarray(xf, dtype=float)
    vf = np.asarray(vf, dtype=float)
    af = np.asarray(af, dtype=float)
    c0 = x0
    c1 = v0
    c2 = a0 / 2
    c3 = 10*(xf - x0) - 6*v0 - 1.5*a0 + 0.5*af - 4*vf
    c4 = -15*(xf - x0) + 8*v0 + 1.5*a0 - af + 7*vf
    c5 = 6*(xf - x0) - 3*v0 - 0.5*a0 + 0.5*af - 3*vf
    def f(tt):
        tt = np.asarray(tt, dtype=float)
        if c0.ndim == 0:
            return c0 + c1*tt + c2*tt**2 + c3*tt**3 + c4*tt**4 + c5*tt**5
        t = tt[..., np.newaxis]
        return c0 + c1*t + c2*t**2 + c3*t**3 + c4*t**4 + c5*t**5
    return f