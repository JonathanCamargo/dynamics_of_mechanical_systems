import sympy
from sympy import symbols
from sympy.physics.mechanics import dynamicsymbols,ReferenceFrame
from dsm import getComponents
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import animation

def Animate(fivebar,trajectory_star=None):       
    fig, ax = plt.subplots()    
    marker,thetas=GetTrajectory(fivebar)    
    min_lims=marker.min(axis=0) if not np.isnan(marker).any() else np.array([-1,-1])
    max_lims=marker.max(axis=0) if not np.isnan(marker).any() else np.array([1,1])
    center=(min_lims+max_lims)/2
    range=(max_lims-min_lims)
    
    def update(frame):
        ax.cla()
        fivebar.plot(thetas[frame,0],ax=ax,theta2=thetas[frame,1],theta3=thetas[frame,2])
        ax.plot(marker[0:frame+1,0],marker[0:frame+1,1],'r-')
        ax.plot(marker[frame,0],marker[frame,1],'r*')
        if trajectory_star is not None:
            ax.plot(trajectory_star[:,0],trajectory_star[:,1],'b.')
        ax.set_xlim(center[0]-2*range.max(),center[0]+2*range.max())
        ax.set_ylim(center[1]-2*range.max(),center[1]+2*range.max())
    
        
    ani = animation.FuncAnimation(fig, update, frames=marker.shape[0], interval=50)
    return ani,fig

def GetTrajectory(fivebar, n_points=40):
    theta_array = np.linspace(0+0.4, 2*np.pi+0.4, n_points)
    trajectory=np.nan*np.ones((len(theta_array),2))
    thetas=np.nan*np.ones((len(theta_array),4))
    for i in range(len(theta_array)):
        theta1=theta_array[i]
        [theta2,theta3],fkout=fivebar.FK(theta1)
        theta4=fivebar.getTheta4(theta1)
        if fkout.cost>1e-3:            
            break
        points=fivebar.ComputePoints(theta1,theta2,theta3)
        # Point 'S' in bar AB
        marker=points['S']
        trajectory[i,:]=marker
        thetas[i,:]=[theta1,theta2,theta3,theta4]
    if np.isnan(trajectory).all():
        trajectory[0,:]=[100,100]
        trajectory[1,:]=[-100,-100]        
    return trajectory,thetas

#fivebar
class FiveBar:
    def __init__(self,l0,l1,l2,l3,l4,l2b):
        # A five bar mechanism
    
        theta1,theta2,theta3,theta4=dynamicsymbols('theta1 theta2 theta3 theta4')
        N=ReferenceFrame('N')
        A=N.orientnew('A','Axis',[theta1,N.z])
        B=N.orientnew('B','Axis',[theta2,N.z])
        C=N.orientnew('C','Axis',[theta3,N.z])
        D=N.orientnew('D','Axis',[theta4,N.z])
 
        r0=l0*N.x
        r1=l1*A.x
        r2=l2*B.x
        r3=l3*C.x
        r4=l4*D.x

        eqLoop=r1+r2-r3-r4-r0

        #Create points
        points={'O':0*N.x,'A':r1,'B':r1+r2,'C':r0+r4,'BPrime':r0+r4+r3,'D':r0,'S':r1+r2+l2b*B.x}
   
        #Create lambdified functions
        points_fun={k:sympy.lambdify([theta1,theta2,theta3,theta4],getComponents(v,N)[0:-1]) for k,v in points.items()}
        pos_fun=sympy.lambdify([theta1,theta2,theta3,theta4],getComponents(eqLoop,N)[0:-1])
        
        #Create lambdified functions
        self.pos_fun=pos_fun
        self.points_fun=points_fun        
        self.zpos=np.deg2rad([45,90+45])
        self.oloc=np.array([0,0])
        self.lengths=[l0,l1,l2,l3,l4,l2b]
        self.rotm=np.eye(2)
        self.GR=-2
        self.theta40=0

    def setGR(self,GR):
        self.GR=GR

    def setTheta40(self,theta40):
        self.theta40=theta40
    
    def setOloc(self,x,y):
        self.oloc=np.array([x,y])

    def setRot(self,theta):
        self.rotm=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    
    def setRotm(self,rotm):
        self.rotm=rotm

    def getTheta4(self,theta1):
        theta4=theta1*self.GR+self.theta40
        return theta4

    def ComputePoints(self,theta1,theta2=None,theta3=None):
        theta4=self.getTheta4(theta1)
        if theta2 is None or theta3 is None:
            z,out=self.FK(theta1)
            theta2,theta3=z
        point_vals={k:np.matmul(self.rotm,point(theta1,theta2,theta3,theta4))+self.oloc for k,point in self.points_fun.items()}
        return point_vals
    
    def plot(self,theta1,ax=None,theta2=None,theta3=None):
        if ax is None:
            ax=plt.gca()
        point_vals=self.ComputePoints(theta1,theta2,theta3)
        for k,p in point_vals.items(): 
            if k!='S': 
                ax.plot(p[0],p[1],'ko')
        ax.plot([point_vals['O'][0],point_vals['D'][0]],[point_vals['O'][1],point_vals['D'][1]],'k')
        ax.plot([point_vals['O'][0],point_vals['A'][0]],[point_vals['O'][1],point_vals['A'][1]])
        ax.plot([point_vals['A'][0],point_vals['S'][0]],[point_vals['A'][1],point_vals['S'][1]])
        ax.plot([point_vals['BPrime'][0],point_vals['C'][0]],[point_vals['BPrime'][1],point_vals['C'][1]])
        ax.plot([point_vals['B'][0],point_vals['BPrime'][0]],[point_vals['B'][1],point_vals['BPrime'][1]],'k--')
        ax.plot([point_vals['C'][0],point_vals['D'][0]],[point_vals['C'][1],point_vals['D'][1]])
        return ax

    def FK(self,theta1,zpos=None):
        theta4=self.getTheta4(theta1)
        if zpos is None:
            zpos=self.zpos
        out=scipy.optimize.least_squares(lambda x: self.pos_fun(theta1,*x,theta4),zpos)
        kThreshold=1e-3
        if out.cost<kThreshold:
            self.zpos=out.x
        return out.x,out