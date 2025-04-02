import sympy
from sympy import symbols
from sympy.physics.mechanics import dynamicsymbols,ReferenceFrame
from dsm import getComponents
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import animation


def Animate(fourbar,trajectory_star=None):       
    fig, ax = plt.subplots()    
    marker,thetas=GetTrajectory(fourbar)    
    min_lims=marker.min(axis=0)
    max_lims=marker.max(axis=0)
    center=(min_lims+max_lims)/2
    range=(max_lims-min_lims)
    
    def update(frame):
        ax.cla()
        fourbar.plot(thetas[frame,0],ax=ax,theta2=thetas[frame,1],theta3=thetas[frame,2])
        ax.plot(marker[0:frame+1,0],marker[0:frame+1,1],'r-')
        ax.plot(marker[frame,0],marker[frame,1],'r*')
        if trajectory_star is not None:
            ax.plot(trajectory_star[:,0],trajectory_star[:,1],'b.')
        ax.set_xlim(center[0]-2*range.max(),center[0]+2*range.max())
        ax.set_ylim(center[1]-2*range.max(),center[1]+2*range.max())
    
        
    ani = animation.FuncAnimation(fig, update, frames=marker.shape[0], interval=50)
    return ani,fig


def GetTrajectory(fourbar, n_points=40):
    theta_array = np.linspace(0+0.4, 2*np.pi+0.4, n_points)
    trajectory=np.nan*np.ones((len(theta_array),2))
    thetas=np.nan*np.ones((len(theta_array),3))
    for i in range(len(theta_array)):
        theta1=theta_array[i]
        [theta2,theta3],fkout=fourbar.FK(theta1)
        if fkout.cost>1e-3:            
            break
        points=fourbar.ComputePoints(theta1,theta2,theta3)
        # Point in the middle of AB
        midpoint_local=fourbar.lengths[2]/2*np.array([np.cos(theta2),np.sin(theta2)])
        marker=points['A']+np.matmul(fourbar.rotm,midpoint_local)
        trajectory[i,:]=marker
        thetas[i,:]=[theta1,theta2,theta3]	
    return trajectory,thetas

#fourbar
class FourBar:
    def __init__(self,l0,l1,l2,l3):
        # A four bar mechanism
        # l0: fixed link assumed horizontal. Use method setAbsoluteReference to set the reference
        # l1 is the driving link, driving with theta1
        
        #l0,l1,l2,l3=symbols('l0 l1 l2 l3')
        theta1,theta2,theta3=dynamicsymbols('theta1 theta2 theta3')
        N=ReferenceFrame('N')
        A=N.orientnew('A','Axis',[theta1,N.z])
        B=N.orientnew('B','Axis',[theta2,N.z])
        C=N.orientnew('C','Axis',[theta3,N.z])
 
        r0=l0*N.x
        r1=l1*A.x
        r2=l2*B.x
        r3=l3*C.x

        eqLoop=r0+r3-r1-r2

        #Create points
        points={'O':0*N.x,'A':r1,'B':r1+r2,'Bprime':r0+r3,'C':r0,}
        points_fun={k:sympy.lambdify([theta1,theta2,theta3],getComponents(v,N)[0:-1]) for k,v in points.items()}

        #Create lambdified functions
        self.pos_fun=sympy.lambdify([theta1,theta2,theta3],getComponents(eqLoop,N)[0:-1])
        self.points_fun=points_fun
        self.zpos=[0.1,0.1]
        self.oloc=np.array([0,0])
        self.lengths=[l0,l1,l2,l3]
        self.rotm=np.eye(2)
    
    def setOloc(self,x,y):
        self.oloc=np.array([x,y])
    
    def setRotm(self,rotm):
        self.rotm=rotm

    def setRot(self,theta):
        self.rotm=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

    def ComputePoints(self,theta1,theta2=None,theta3=None):
        if theta2 is None or theta3 is None:
            z,out=self.FK(theta1)
            theta2,theta3=z
        point_vals={k:np.matmul(self.rotm,point(theta1,theta2,theta3))+self.oloc for k,point in self.points_fun.items()}
        return point_vals
    
    def plot(self,theta1,ax=None,theta2=None,theta3=None):
        if ax is None:
            ax=plt.gca()
        point_vals=self.ComputePoints(theta1,theta2,theta3)
        for k,p in point_vals.items():  
            ax.plot(p[0],p[1],'ko')
        ax.plot([point_vals['O'][0],point_vals['C'][0]],[point_vals['O'][1],point_vals['C'][1]],'k')
        ax.plot([point_vals['O'][0],point_vals['A'][0]],[point_vals['O'][1],point_vals['A'][1]])
        ax.plot([point_vals['A'][0],point_vals['B'][0]],[point_vals['A'][1],point_vals['B'][1]])
        ax.plot([point_vals['B'][0],point_vals['Bprime'][0]],[point_vals['B'][1],point_vals['Bprime'][1]],'k:')
        ax.plot([point_vals['Bprime'][0],point_vals['C'][0]],[point_vals['Bprime'][1],point_vals['C'][1]])
    

    def FK(self,theta1,zpos=None):
        if zpos is None:
            zpos=self.zpos
        out=scipy.optimize.least_squares(lambda x: self.pos_fun(theta1,*x),zpos)
        kThreshold=1e-3
        if out.cost<kThreshold:
            self.zpos=out.x
        return out.x,out
