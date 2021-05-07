'''
========================================================================
Conjugate Gradients Algorithm
------------------------------------------------------------------------
Excercise 3
========================================================================
'''
#
# ======================================================================
# imports
# ======================================================================
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as col
#
#plt.ion() # interactive plotting on
#
# ======================================================================
# define tubaf color map
# ======================================================================
#
geo=np.array([224,134,3])/255   # geo (orange)
ing=np.array([35,186,226])/255    # ing (blue)
ene=np.array([181,18,62])/255   # ener (red)
mat=np.array([26,150,43])/255   # math (green)
tubaf = col.LinearSegmentedColormap.from_list('tubaf', [ing,mat,geo,ene],N=256)
#
# ======================================================================
# define abaqus color map
# ======================================================================
#
abq = col.LinearSegmentedColormap.from_list("abq", [(0,0,1),(0,1,1),(0,1,0),(1,1,0),(1,0,0)],N=256)
#
# ======================================================================
# 2d test function
# ======================================================================
#
def func_2d(x):
  '''
  ======================================================================
  objective function
  ----------------------------------------------------------------------
  x: parameter vector x=np.array([x0,x1])
  ======================================================================
  '''
  f=(x[0]-2*x[1])**2 + (x[0]-2)**4
  return f
# 
def grad_2d(x):
  '''
  ======================================================================
  gradient of objective function
  ----------------------------------------------------------------------
  x: parameter vector x=np.array([x0,x1])
  ======================================================================
  '''
  dfdx0=2*(2*(x[0]-2)**3+x[0]-2*x[1])
  dfdx1=4*(2*x[1]-x[0])
  g=np.array([dfdx0,dfdx1])
  return g
#
def hess_2d(x):
  '''
  ======================================================================
  hessian of objective function
  ----------------------------------------------------------------------
  x: parameter vector x=np.array([x0,x1])
  ======================================================================
  '''
  d2fdx0x0=2*(6*(x[0]-2)**2+1)
  d2fdx0x1=-4
  d2fdx1x0=-4
  d2fdx1x1=8
  H=np.array([[d2fdx0x0,d2fdx0x1],[d2fdx1x0,d2fdx1x1]])
  return H
#
# ======================================================================
# rosenbrook function
# ======================================================================
#
def func_rb(x):
  '''
  ======================================================================
  objective function (Rosenbrook)
  ----------------------------------------------------------------------
  x: parameter vector x=np.array([x0,x1])
  ======================================================================
  '''
  f=(1-x[0])**2 + 100*(x[1]-x[0]**2)**2
  return f
#
def grad_rb(x):
  '''
  ======================================================================
  gradient of objective function (Rosenbrook)
  ----------------------------------------------------------------------
  x: parameter vector x=np.array([x0,x1])
  ======================================================================
  '''
  dfdx0=2*(200*x[0]**3-200*x[0]*x[1]+x[0]-1)
  dfdx1=200*(x[1]-x[0]**2)
  g=np.array([dfdx0,dfdx1])
  return g
#
def hess_rb(x):
  '''
  ======================================================================
  hessian of objective function (Rosenbrook)
  ----------------------------------------------------------------------
  x: parameter vector x=np.array([x0,x1])
  ======================================================================
  '''
  d2fdx0x0=1200*x[0]**2-400*x[1]+2
  d2fdx0x1=-400*x[0]
  d2fdx1x0=-400*x[0]
  d2fdx1x1=200
  H=np.array([[d2fdx0x0,d2fdx0x1],[d2fdx1x0,d2fdx1x1]])
  return H
#
# ======================================================================
# plot function
# ======================================================================
#
def plotfunc(f,opt=[0,0],n=[101,101,11],xlim=[0,6],ylim=[0,6],zlim=[0.1,250],levels=None,fmt='%.0f',title=r'$f(x)$',figsize=(8,8),dpi=72):
  '''
  ======================================================================
  plot 2d objective function as contour plot
  ----------------------------------------------------------------------
  f: function 2d (callable)
  opt: [x,y] position of minimum
  n: [nx,ny,nz] resolution and number of contours
  xlim: [xmin,xmax]
  ylim: [ymin,ymax]
  zlim: [zmin,zmax]
  levels: list of contour levels
  fmt: format for displaying level values
  title: title for plot
  figsize: (xsize,ysize) in inches
  dpi: screen resolution in dots per inches (adapt to your screen) 
  ======================================================================
  '''
  #
  x=np.linspace(xlim[0],xlim[1],n[0])
  y=np.linspace(ylim[0],ylim[1],n[1])
  X,Y=np.meshgrid(x,y)
  Z=f([X,Y])
  #
  if not levels:
    levels=np.geomspace(zlim[0],zlim[1],n[2])
  plt.figure(figsize=figsize,dpi=dpi)
  CP=plt.contour(X,Y,Z,cmap=tubaf,levels=levels)
  LP=plt.plot([opt[0]],[opt[1]],color='red',marker='x',linewidth=2,markersize=12,markeredgewidth=2)
  plt.clabel(CP,inline=True,fmt=fmt)
  plt.xlabel(r'$x_0$', fontsize=0.2*dpi)
  plt.ylabel(r'$x_1$', fontsize=0.2*dpi)
  plt.title(title, fontsize=0.2*dpi)
#
# ======================================================================
# 1d newton-raphson
# ======================================================================
#
def newton1d(x,grad,hess,s,max_iter=100,gtol=1e-8):
  '''
  ======================================================================
  1d minimization algorithm using the Newton-Raphson algorithm
  find a such that g(x+s*a) == 0
  ----------------------------------------------------------------------
  x: start point x=np.array([x0,..,xn])
  grad: gradient of objective function (callable)
  hess: hessian of objective function (callable)
  s: search direction
  max_iter: maximum number of iterations allowed
  gtol: accuracy for result
  ----------------------------------------------------------------------
  return
  a: optimal point a where g(x+a*s)*s=0
  ======================================================================
  '''
  a=0
  g=np.dot(grad(x),s)
  H=np.dot(np.dot(s,hess(x)),s)
  i=1
  while abs(g) > gtol:
    if i > max_iter:
      print('maximum number of Newton-Raphson iterations %d exceeded' %(max_iter))
      break
    a=a-g/H
    g=np.dot(grad(x+a*s),s)
    H=np.dot(np.dot(s,hess(x+a*s)),s)
    i+=1
  return a
#
# ======================================================================
# conjugated gradients
# ======================================================================
#
def cg(x,func,grad,hess,update='FR',max_iter=50,xtol=1e-6):
    '''
    ======================================================================
    conjugated gradient
    find minimum of function f
    ----------------------------------------------------------------------
    x: start parameter np.array([x0,...,xn])
    func: function to minimize (callable)
    grad: gradient of f (callable)
    hess: hessian of f (callable)
    update: ['FR','PR','HS'] 
    max_iter: maximum number of iterations
    xtol: stop criteria dx < tol
    ======================================================================  
    '''
    print('CG optimizer using %s update' %(update))
    d=len(x)  # dimension of the problem
    xlist=[x]
    i=1
    res=1
    while abs(res) > xtol:
        if i > max_iter:
            print('maximum number of iteration exceeded')
            break
        #
        # initial gradient
        #
        g0=grad(x)
        #
        # initial search direction
        #
        s=-g0
        #
        for j in range(d):
            #
            # compute alpha using newton1d()
            #
            a=newton1d(x,grad,hess,s)
            #
            # update x
            #
            dx=a*s
            x=x+dx
            #
            # compute res
            #
            res=np.linalg.norm(dx)
            #
            # append x to xlist
            #
            xlist.append(x)
            #
            # evaluate f and g
            #
            f1=func(x)
            g1=grad(x)
            #
            # compute beta (with choice for FR, PR, HS)
            #
            if update=='FR':
                if np.dot(g0,g0)>0:
                    beta=np.dot(g1,g1)/np.dot(g0,g0)
                else:
                    beta=0
            elif update=='PR':
                if np.dot(g0,g0)>0:
                    beta=np.dot(g1,(g1-g0))/np.dot(g0,g0)
                else:
                    beta=0
            elif update=='HS':
                if np.dot(s,(g1-g0))>0:
                    beta=np.dot(g1,(g1-g0))/np.dot(s,(g1-g0))
                else:
                    beta=0
            else:
                print('unknown update rule')
                beta=0
            #
            if j != d-1:
                #
                # update gradient
                #
                g0=g1
                #
                # upddate search direction
                #
                s=-g1+beta*s
            #
            # print out
            #
            print('%3d x=%s f(x)=%g g(x)=%s' %(i,str(x),f1,str(g1)))
            #
            # check residual
            #
            if abs(res) < xtol:
                return xlist
            i+=1
    return xlist
#
# ======================================================================
# main program
# ======================================================================
#
if __name__ == '__main__':
    #
    # loop update relations
    #
    for update in ['FR','PR','HS']:
        #
        # plot objective function as contour plot (adjust opt, xlim, ylim, zlim)
        #
        plotfunc(func_2d,opt=[2,1],xlim=[-2,6],ylim=[-3,5],zlim=[0.1,250],fmt='%.1f',title=r'$f(x)$',figsize=(8,8),dpi=72)
        #
        # start value
        #
        x0=np.array([3,5])
        #
        # run cg optimizer
        #
        xlist=cg(x0,func_2d,grad_2d,hess_2d,update=update)  # 'FR', 'HS', 'PR'
        #
        # plot optimization path
        #
        X=np.array(xlist)[:,0]
        Y=np.array(xlist)[:,1]
        plt.plot(X,Y,'b-o')
        plt.savefig('f2d-%s.png' %(update))
        plt.show()
        #inp=input('press key to continue')
        #
        # plot objective function as contour plot (adjust opt, xlim, ylim, zlim)
        #
        plotfunc(func_rb,opt=[1,1],xlim=[-2,2],ylim=[-1,3],zlim=[0.1,1000],levels=[1,3,10,30,60,110,250,500,1000],fmt='%.0f',title=r'$f(x)$',figsize=(8,8),dpi=72)
        #
        # start value
        #
        x0=np.array([-0.5,2.0])
        #
        # run cg optimizer
        #
        xlist=cg(x0,func_rb,grad_rb,hess_rb,update=update)  # 'FR', 'HS', 'PR'
        #
        # plot optimization path
        #
        X=np.array(xlist)[:,0]
        Y=np.array(xlist)[:,1]
        plt.plot(X,Y,'b-o')
        plt.savefig('frb-%s.png' %(update))
        plt.show()
        #inp=input('press key to continue')
