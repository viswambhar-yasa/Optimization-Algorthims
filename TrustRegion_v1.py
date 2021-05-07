#
# trust region algorithm
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as col

np.set_printoptions(precision=6)

geo=np.array([224,134,3])/255   # geo (orange)
ing=np.array([35,186,226])/255    # ing (blue)
ene=np.array([181,18,62])/255   # ener (red)
mat=np.array([26,150,43])/255   # math (green)
tubaf = col.LinearSegmentedColormap.from_list('tubaf', [ing,mat,geo,ene],N=256)

def trustRegion(x0,func,grad,r0,rmax=10,eta=0.125,gtol=1e-6,imax=100,jmax=20,disp=True):
    '''
    ======================================================================
    trust region algorithm
    ----------------------------------------------------------------------
    x0: start point
    func: objective function (callable)
    grad: gradient of objective function (callable)
    r0: initial trust region radius
    rmax: maximum trust region radius
    eta: parameter for accepting a step
    gtol: break criterion
    imax: maximum number of steps
    jmax: maximum number of trust region reductions
    disp: if True print informations about each iteration
    ----------------------------------------------------------------------
    return: xopt
    ======================================================================
    '''
    d=len(x0)             # dimension of problem
    f0=func(x0)           # initial function value
    g0=grad(x0)           # initial gradient
    B0=np.eye(d)          # initial B matrix for model m
    H0=B0                 # initial inverse Hessian (inv(B0)) 
    x=[x0]                # list of points
    #
    if disp:
        print('='*80)
        print('   Trust Region Algorithm with BFGS Hessian Approximation')
    #
    normg=np.linalg.norm(g0)
    i=1             # counter for steps
    j=1             # counter for trust-region 
    while normg > gtol and i < imax and j < jmax:
        if disp:
            print('='*80)
            print('TR iteration %d/%d' %(i,j))
            print('-'*20)
            print('      r0=%g' %(r0))
            print('      x0=%s' %(str(x0)))
            print('      f0=%g' %(f0))
            print('      g0=%s' %(str(g0)))
        #
        # BFGS
        #
        if i > 1:
            B1=BupdateBFGS(B0,p,q)
            H1=HupdateBFGS(H0,p,q)
        else:
            B1=B0
            H1=H0
        #
        # compute step
        #
        sB=-np.dot(H1,g0)                                                   # compute full step using inverse Hessian
        if np.linalg.norm(sB) <= r0:                                        # if full step is inside trust region
            if disp:
                print('  full step')
                print(' ','-'*40)
                print('      s=%s' %(str(sB)))
            s1=sB                                                             # accept full step
        else:                                                               # if full step is outside trust region
            sU=-np.dot(g0,g0)/(np.dot(g0,np.dot(B1,g0)))*g0                   # unbounded steepest descent step
            if np.linalg.norm(sU) > r0:                                       # unbounded steepest descent step outside trust region
                s1=-r0*g0/np.linalg.norm(g0)                                    # cut unbounded step to trust region radius
                if disp:
                    print('  steepest descent step')
                    print(' ','-'*40)
                    print('     sU=%s' %(str(sU))) 
                    print('      s=%s' %(str(s1)))
            else:                                                             # use dogleg method
                tau=dogleg(sU,sB,r0)                                            # compute tau for dogleg
                s1=sU+(tau-1)*(sB-sU)                                           # compute dogleg step
                if disp:
                    print('  dogleg step')
                    print(' ','-'*40)
                    print('     tau=%g' %(tau))
                    print('       s=%s' %(str(s1)))
        sL=np.linalg.norm(s1)                                               # step length
        if disp:
            print('      sL=%g' %(sL))
        #
        # compute approximation
        #
        f1=func(x0+s1)                                                      # get actual function value  
        m1=quadraticModel(s1,f0,g0,B1)                                      # get actual model value for s=si
        m0=f0
        a=(f0-f1)/(m0-m1)                                                   # actual approximation
        #
        # adjust trust region radius
        #
        print('       a=%g' %(a))
        if a < 0.25:                                                        # approximation < 0.25 (bad)
            r1=0.25*np.linalg.norm(r0)                                      
        else:
            if a > 0.75 and abs(sL - r0) < 1e-8:                              # approximation > 0.75 (good) and step size == trust region radius
                r1=min(2*r0,rmax)                                               # increase trust region radius
            else:                                                             # approximation OK
                r1=r0                                                           # keep trust region radius
        if a > eta:                                                         # approximation good enough?
            if disp:
                print('  step accepted')
                print(' ','-'*40)
            x1=x0+s1                # compute new point
            g1=grad(x1)             # evaluate gradient
            p=x1-x0
            q=g1-g0
            x0=x1
            f0=f1
            g0=g1
            B0=B1
            H0=H1
            r0=r1
            x.append(x1)
            #
            j=1   # if we accept a step we reset the counter for rejected steps
            i+=1
        else:   # reject step
            x0=x0 # keep point
            r0=r1 # change trust region radius
            #
            x1=x0 # for disp
            g1=g0 # for disp
            f1=f0 # for disp
            r1=r1 # for disp
            if disp:
                print('  step rejected')
                print(' ','-'*40)
            j+=1
        #
        normg=np.linalg.norm(g1)  # ||g||
        if disp:
            print('      r1=%g' %(r1))
            print('      x1=%s' %(str(x1)))
            print('      f1=%g' %(f1))
            print('      g1=%s' %(str(g1)))
            print('   ||g||=%s' %(normg))
    #
    # end
    #
    if disp:
        print('='*80)
        print('    xopt=%s' %(str(x[-1])))
        print('='*80)
    return x                                                            # return x list
  
def BupdateBFGS(B,p,q):
    '''
    ======================================================================
    Broyden-Fletcher-Goldfarb-Shanno (BFGS) update for inverse Hessian H
    ----------------------------------------------------------------------
    B: symmetric matrix
    p: change of location (x[i]-x[i-1])
    q: change of gradient (g[i]-g[i-1])
    ======================================================================
    ''' 
    Bp=np.dot(B,p)
    pB=np.dot(p,B)
    pBp=np.dot(p,Bp)
    BppB=np.outer(Bp,pB)
    qq=np.outer(q,q)
    pq=np.dot(p,q)
    return B-BppB/pBp+qq/pq
  
def HupdateBFGS(H,p,q):
    '''
    ======================================================================
    Broyden-Fletcher-Goldfarb-Shanno (BFGS) update for inverse Hessian H
    ----------------------------------------------------------------------
    H: symmetric matrix
    p: change of location (x[i]-x[i-1])
    q: change of gradient (g[i]-g[i-1])
    ======================================================================
    ''' 
    pHq=p-np.dot(H,q)
    pHqp=np.outer(pHq,p)
    ppHq=np.outer(p,pHq)
    pq=np.dot(p,q)
    pHqq=np.dot(pHq,q)
    pp=np.outer(p,p)
    H=H+(pHqp+ppHq)/pq-pHqq/(pq*pq)*pp
    return H

def quadraticModel(dx,f0,g0,h0):
    '''
    ======================================================================
    quadratic model
    ----------------------------------------------------------------------
    dx: step
    f0: function value f(dx=0)
    g0: gradient g(dx=0)
    h0: Hessian h(dx=0)
    ----------------------------------------------------------------------
    return
    f1: function f(dx)
    ======================================================================
    '''
    f1=f0+np.dot(g0,dx)+0.5*np.dot(dx,np.dot(h0,dx))
    return f1
  
def dogleg(a,b,c):
    '''
    ======================================================================
    trust region intersection
    ----------------------------------------------------------------------
    a: vector
    b: vector
    c: radius
    ----------------------------------------------------------------------
    return
    x1: intersection point (1 <= x1 <= 2)
    ======================================================================
    '''
    k1=np.dot(a,b-a)
    k2=np.linalg.norm(b-a)**2
    t1=(k1-k2)/k2
    t2=(np.linalg.norm(a)**2-2*k1+k2-c**2)/k2
    d=t1**2-t2
    x1=-t1+np.sqrt(d)
    return x1

def plotfunc(f,opt=[0,0],n=[101,101,11],xlim=[0,6],ylim=[0,6],zlim=[0.1,250],levels=None,fmt='%.0f',title=r'$f(x)$',figsize=(8,8),dpi=96):
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
    plt.xlabel(r'$x_0$')
    plt.ylabel(r'$x_1$')
    plt.title(title)

if __name__ == '__main__':
    #
    # f3 (Rosenbrock, quartic function)
    #
    def f3(x):
        F=(1-x[0])**2 + 100*(x[1]-x[0]**2)**2
        return F
    def g3(x):
        dfdx0=2*(200*x[0]**3-200*x[0]*x[1]+x[0]-1)
        dfdx1=200*(x[1]-x[0]**2)
        G=np.array([dfdx0,dfdx1])
        return G
    def h3(x):
        d2fdx0x0=1200*x[0]**2-400*x[1]+2
        d2fdx0x1=-400*x[0]
        d2fdx1x0=-400*x[0]
        d2fdx1x1=200
        H=np.array([[d2fdx0x0,d2fdx0x1],[d2fdx1x0,d2fdx1x1]])
        return H
    f3options={'opt':[1,1],'n':[201,201,11],'xlim':[-2,2],'ylim':[-1,3],'zlim':[0.1,1000],'levels':[1,3,10,30,60,110,250,500,1000],'fmt':'%.0f','title':r'$f(x)=(1-x_0)^2 + 100*(x_1-x_0^2)^2$'}
    plotfunc(f3,figsize=(6,6),dpi=120,**f3options)  
    x0=np.array([-0.5,2])
    r0=1.0
    x=trustRegion(x0,f3,g3,r0)
    X=np.array(x)
    plt.plot(X[:,0],X[:,1],'b-o')
    plt.show()
    #
    # f4 
    #
    def f4(x):
        return x[0]**2+x[1]**2+10*np.exp(-x[0]**2-x[1]**2)+x[0]+x[1]
    def g4(x):
        return np.array([-20*x[0]*np.exp(-x[0]**2-x[1]**2)+2*x[0]+1, -20*x[1]*np.exp(-x[0]**2-x[1]**2)+2*x[1]+1])
    f4options={'opt':[-1.19287,-1.19287],'n':[101,101,11],'xlim':[-3,3],'ylim':[-3,3],'zlim':[0.1,20],'levels':[1.1,1.25,1.5,2,3,4,5,7.5,10,12.5,15,20],'fmt':'%.2f','title':r'$f(x)=x_0^2+x_1^2+10 exp(-x_0^2-x_1^2)+x_0+x_1$'}
    plotfunc(f4,figsize=(6,6),dpi=120,**f4options)
    x0=np.array([0.5,1.0])
    r0=1.0
    x=trustRegion(x0,f4,g4,r0)
    X=np.array(x)
    plt.plot(X[:,0],X[:,1],'b-o')
    plt.show()
