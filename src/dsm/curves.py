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
        t=np.linspace(0, 1, len(self.points))
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