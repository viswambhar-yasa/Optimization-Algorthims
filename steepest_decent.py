'''
========================================================================
Exercise2.py
------------------------------------------------------------------------
solve an unconstrained optimization problem using the steepest decent
method with a line search algorithm and optimal step size
========================================================================
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as col

geo=np.array([224,134,3])/255        # geo (orange)
ing=np.array([35,186,226])/255    # ing (blue)
ene=np.array([181,18,62])/255        # ener (red)
mat=np.array([26,150,43])/255        # math (green)
tubaf = col.LinearSegmentedColormap.from_list('tubaf', [ing,mat,geo,ene],N=256)
abq = col.LinearSegmentedColormap.from_list("abq", [(0,0,1),(0,1,1),(0,1,0),(1,1,0),(1,0,0)],N=256)

def ga(x,g,a,s):
    '''
    ======================================================================
    evaluate line search gradient df/da
    g(x+a*s)=g(x)*s
    ----------------------------------------------------------------------
    arguments:
    x: start point of line 
    g: gradient function to evaluate (callable)
    a: step size
    s: line direction
    ----------------------------------------------------------------------
    return:
    G: gradient along s
    ======================================================================
    '''
    G=np.dot(g(x+a*s),s)
    return G

def ha(x,h,a,s):
    '''
    ======================================================================
    evaluate line search hessian d2f/da2
    h(x+a*s)=s*h(x)*s
    ----------------------------------------------------------------------
    arguments:
    x: start point of line 
    h: hessian function to evaluate (callable)
    a: step size
    s: line direction
    ----------------------------------------------------------------------
    return:
    H: hessian along s
    ======================================================================
    '''
    H=np.einsum('i,ij,j',s,h(x+s*a),s)
    return H

def newtonLineSearch(x,g,h,s,gtol=1e-8,a=0):
    '''
    ======================================================================
    newton algorithm to find a minimum of a nd-function along direction s
    ---------------------------------------------------------------------
    x: start point
    g: gradient of function to be minimized
    h: hessian of function to be minimized
    s: search direction
    gtol: break criterion
    ----------------------------------------------------------------------
    return: 
    a: optimal step size
    ======================================================================
    '''
    i=0
    G=1
    while G>gtol:
        G=ga(x,g,a,s)
        H=ha(x,h,a,s)
        a=a-G/H
        i+=1
    return a

def armijoBacktracking(x,f,g,s,a=1.0,c1=1e-4,tau=0.7,disp=True):
    '''
    ======================================================================
    armijo backtracking algorithm
    find a suitable step length a, which leads to a sufficient decrease
    f(x+a*s) < f(x)
    ----------------------------------------------------------------------
    arguments:
    x: start point
    f: function to be minimized (callable)
    g: gradient of f (callable)
    s: search direction
    a: initial step length
    c1: armijo parameter  0 < c1
    ----------------------------------------------------------------------
    return:
    a: acceptable step size (fulfilling the armijo criteria)
    ======================================================================
    '''
    i=0
    while f(x+a*s) > f(x)+c1*a*np.dot(g(x),s):
        a=tau*a
        i+=1
    if disp:
        print('armijo backtracking successful after %d steps a=%g' %(i,a))
    return a

def lineSearch(x,f,g,h,gtol=1e-6,disp=True,maxiter=10000):
    '''
    ======================================================================
    line search algorithm to find a minimum of a function
    ----------------------------------------------------------------------
    arguments:
    x: start point
    f: function to be minimized
    g: gradient of function to be minimized
    h: hessian of function to be minimized
    gtol: break criterion
    ----------------------------------------------------------------------
    return:
    xlist: list of points xlist[-1] contains the optimal point
    ======================================================================
    '''
    i=0
    xlist=[x]
    res=1
    while res>gtol and i<maxiter:
        G=g(x)
        H=h(x)
        #
        # search direction (steepest descent)
        #
        #s=-G
        #
        # Newton direction
        #
        s=-np.dot(np.linalg.inv(H),G)
        #
        # newton line search
        #
        a=newtonLineSearch(x,g,h,s,gtol=1e-8,a=0)
        #
        # armijo steplength
        #
        #a=armijoBacktracking(x,f,g,s)
        #
        # update
        #
        x=x+a*s
        G=g(x)
        res=np.linalg.norm(G)
        xlist.append(x)
        i+=1
        #
        if disp:
            print('i=%d a=%g x=[%g %g] g=%g f=%g' %(i,a,x[0],x[1],res,f(x)))
        #
    if i>=maxiter:
        print('!!! Not converged after 1000 iterations !!!')
    return xlist

def plotfunc(f,opt=[0,0],n=[101,101,11],xlim=[0,6],ylim=[0,6],zlim=[0.1,250],levels=None,fmt='%.0f',title=r'f(x)',figsize=(8,8),dpi=120):
    '''
    ======================================================================
    plot 2d objective function as contour plot
    ----------------------------------------------------------------------
    arguments:
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
    x=np.linspace(xlim[0],xlim[1],n[0])
    y=np.linspace(ylim[0],ylim[1],n[1])
    X,Y=np.meshgrid(x,y)
    Z=f([X,Y])
    if not levels:
        levels=np.geomspace(zlim[0],zlim[1],n[2])
    plt.figure(figsize=figsize,dpi=dpi)
    CP=plt.contour(X,Y,Z,cmap=abq,levels=levels)
    LP=plt.plot([opt[0]],[opt[1]],color='red',marker='x',linewidth=2,markersize=12,markeredgewidth=2)
    plt.clabel(CP,inline=True,fmt=fmt)
    plt.title(title)
    plt.xlabel(r'$x_0$')
    plt.ylabel(r'$x_1$')
    plt.grid()

if __name__ == '__main__':
    
    def f(x):
        '''
        ====================================================================
        objective function
        --------------------------------------------------------------------
        arguments:
        x: point where to compute function value
        --------------------------------------------------------------------
        return: 
        f: function value f(x)
        ====================================================================
        '''
        f=(x[0]-2*x[1])**2+(x[0]-2)**4
        return f

    def g(x):
        '''
        ====================================================================
        gradient of function
        --------------------------------------------------------------------
        arguments:
        x: point where to compute gradient
        --------------------------------------------------------------------
        return: 
        g: gradient of f(x)
        ====================================================================        
        '''
        dfdx0=2*x[0]-4*x[1]+4*(x[0]-2)**3
        dfdx1=-4*x[0]+8*x[1]
        g=np.array([dfdx0,dfdx1])
        return g

    def h(x):
        '''
        ====================================================================
        hessian of objective function
        -----------------------------
        arguments:
        x: point where to compute hessian
        --------------------------------------------------------------------
        return: 
        h: hessian of f(x)
        ====================================================================        
        '''
        d2dx0dx0=2+12*(x[0]-2)**2
        d2dx0dx1=-4
        d2dx1dx0=-4
        d2dx1dx1=8
        h=np.array([[d2dx0dx0,d2dx0dx1],[d2dx1dx0,d2dx1dx1]])
        return h
    
    def f2(x):
        return x[0]**2+x[1]**2
    def g2(x): 
        return np.array([2*x[0],2*x[1]])
    def h2(x):
        return np.array([[2,0],[0,2]])
    
    def f3(x):
        return (1-x[0])**2+100*(x[1]-x[0]**2)**2
    
    def g3(x):
        dfdx0=2*(-1 + 200*x[0]**3 + x[0]*(1 - 200*x[1]))
        dfdx1=-200*(x[0]**2 - x[1])
        g=np.array([dfdx0,dfdx1])
        return g
    
    def h3(x):
        d2dx0dx0=1200*x[0]**2-400*x[1]+2
        d2dx0dx1=-400*x[0]
        d2dx1dx0=-400*x[0]
        d2dx1dx1=200
        h=np.array([[d2dx0dx0,d2dx0dx1],[d2dx1dx0,d2dx1dx1]])
        return h
    
    x=np.array([0.5,2.5])
    xlist=lineSearch(x,f3,g3,h3)
    plotfunc(f3,opt=[1,1],n=[201,201,15],xlim=[-1,2],ylim=[-1,2],zlim=[0.1,1000.0],fmt='%.1f',title=r'$f(x)=(x_0-2x_1)^2+(x_0-2)^4$')
    X=np.array(xlist)[:,0]
    Y=np.array(xlist)[:,1]
    plt.plot(X,Y,'b-o')
    plt.show()
