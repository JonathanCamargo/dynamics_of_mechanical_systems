import sympy
from sympy import symbols
from sympy.physics.mechanics import dynamicsymbols
import numpy as np
import matplotlib.pyplot as plt

def getComponents(expr,frame,simplify=True):
    if simplify:
        return [expr.dot(frame.x).simplify(),expr.dot(frame.y).simplify(),expr.dot(frame.z).simplify()]
    else:
        return [expr.dot(frame.x),expr.dot(frame.y),expr.dot(frame.z)]

def prettyPlot(*args,**kwargs):
    ax=plt.subplot(1,1,1)
    ax.plot(*args)
    prettyAx(ax,**kwargs)
    return ax

def prettyAx(ax, title='', xlabel='x', ylabel='y', leyenda=False, xlim=[None, None], ylim=[None, None]):
    ''' Modifica un axes para que se vea bonito ''' 
    ax.set_title(title, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.tick_params(direction='out', length=5, width=0.75, grid_alpha=0.3)
    ax.tick_params(axis='x', rotation=0)  # Corrected this line
    ax.minorticks_on()
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlim(xlim[0], xlim[1])
    ax.grid(True)
    ax.grid(visible=True, which='major', color='grey', linestyle='-')
    ax.grid(visible=True, which='minor', color='lightgrey', linestyle='-', alpha=0.2)
    
    if leyenda:
        ax.legend(loc='best')  # Corrected this line
    
    plt.tight_layout()
    
def ft2m(x):
    #Convierte de pies a metros
    return x*12*25.4/1000

def rpm2rad_s(x):
    return x*2*np.pi/60

def km_h2m_s(x):
    return x*1000/3600