import sympy
from sympy import symbols
from sympy.physics.mechanics import dynamicsymbols
import numpy as np

def getComponents(expr,frame,simplify=True):
    if simplify:
        return [expr.dot(frame.x).simplify(),expr.dot(frame.y).simplify(),expr.dot(frame.z).simplify()]
    else:
        return [expr.dot(frame.x),expr.dot(frame.y),expr.dot(frame.z)]
    