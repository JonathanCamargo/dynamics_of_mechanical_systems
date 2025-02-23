import numpy as np
from numpy import sin, cos

def dq_dt(q, t=None, u=None):
    if u == None:
        u = np.zeros(4)
    qdot = np.zeros(6)
    qdot[0] = q[3]
    qdot[1] = q[4]
    qdot[2] = q[5]
    qdot[3] = thetaddot_fun(*q, *u)
    qdot[4] = xddot_fun(*q, *u)
    qdot[5] = yddot_fun(*q, *u)
    return qdot

def thetaddot_fun(_Dummy_88, _Dummy_87, _Dummy_86, _Dummy_85, _Dummy_84, _Dummy_83, u1, u2, u3, u4):
    return -42.4264068711929 * u1 + 42.4264068711929 * u2 - 42.4264068711929 * u3 + 42.4264068711929 * u4

def xddot_fun(_Dummy_94, _Dummy_93, _Dummy_92, _Dummy_91, _Dummy_90, _Dummy_89, u1, u2, u3, u4):
    return u1 * sin(_Dummy_94 + 0.785398163397448) + u2 * sin(_Dummy_94 - 0.785398163397448) + u3 * sin(_Dummy_94 - 0.785398163397448) + u4 * sin(_Dummy_94 + 0.785398163397448)

def yddot_fun(_Dummy_100, _Dummy_99, _Dummy_98, _Dummy_97, _Dummy_96, _Dummy_95, u1, u2, u3, u4):
    return -u1 * cos(_Dummy_100 + 0.785398163397448) - u2 * cos(_Dummy_100 - 0.785398163397448) - u3 * cos(_Dummy_100 - 0.785398163397448) - u4 * cos(_Dummy_100 + 0.785398163397448)