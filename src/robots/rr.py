import sympy
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols
from sympy.physics.mechanics import dynamicsymbols,ReferenceFrame
from dsm import getComponents
from scipy.optimize import least_squares

class rr:
    def __init__(self,params=None):
        L1,L2=symbols('L1,L2')
        if params==None:
            self.params={'L1':1,'L2':1}
        else:
            self.params=params
        # Define symbolic variables for each joint
        theta1,theta2=dynamicsymbols('theta1,theta2')
        self.joint_angles=[theta1,theta2]
        # Define symbolic variables for constant parameters
        # Define reference frames for each segment
        N=ReferenceFrame('N')
        A=N.orientnew('A','Axis',(theta1,N.z))
        B=A.orientnew('B','Axis',(theta2,N.z))
        self.frames=[N,A,B]
        r1=A.x*L1
        r2=B.x*L2    
        
        points={'O':0*N.x,'A':r1,'EEF':r1+r2,'B':r1+0.95*r2,'deltaEffX':L1*0.075*B.x,'deltaEffY':L1*0.05*B.y}
        self.points=points
        self.points_fun=dict([(key,self.lambdifier(points[key])) for key in points.keys()])

    def lambdifier(self,point):
        return sympy.lambdify(self.joint_angles,[eq.subs(self.params) for eq in getComponents(point,self.frames[0])])

    def plot(self,theta1,theta2):
            points_num=dict([(key,np.array(self.points_fun[key](theta1,theta2))) for key in self.points_fun.keys()])
            # effector:
            C_left=points_num['B']-points_num['deltaEffY']
            C_right=points_num['B']+points_num['deltaEffY']
            D_left=C_left+points_num['deltaEffX']
            D_right=C_right+points_num['deltaEffX']
            plt.plot([points_num['O'][0],points_num['A'][0]],[points_num['O'][1],points_num['A'][1]],'b')
            plt.plot([points_num['A'][0],points_num['B'][0]],[points_num['A'][1],points_num['B'][1]],'r')
            plt.plot([points_num['B'][0],C_left[0]],[points_num['B'][1],C_left[1]],'r')
            plt.plot([points_num['B'][0],C_right[0]],[points_num['B'][1],C_right[1]],'r')
            plt.plot([C_left[0],D_left[0]],[C_left[1],D_left[1]],'r')
            plt.plot([C_right[0],D_right[0]],[C_right[1],D_right[1]],'r')
            plt.axis('equal')
            
    def er2_fun(self,thetas,pos_deseada):
            pos=np.array(self.points_fun['EEF'](*thetas))[0:2]
            er2=np.sum((pos-pos_deseada)**2)
            return er2
            
    def IK(self,pos_deseada,x0=None):
        if x0 is not list:
            if x0 is None:
                x0=[0.2,0.2]
        result=least_squares(self.er2_fun,x0,args=[pos_deseada])
        return result
