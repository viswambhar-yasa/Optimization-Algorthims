# test.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as col

np.set_printoptions(precision=6,linewidth=160)
    
geo=np.array([224,134,3])/255   # geo (orange)
ing=np.array([35,186,226])/255    # ing (blue)
ene=np.array([181,18,62])/255   # ener (red)
mat=np.array([26,150,43])/255   # math (green)
tubaf = col.LinearSegmentedColormap.from_list('tubaf', [ing,mat,geo,ene],N=256)

def newtonLine(x,g,h,s,gtol=1e-6,disp=True):
    '''
    ======================================================================
    newton algorithm to find a minimum of a nd-function along direction s
    ---------------------------------------------------------------------
    x: start point
    g: gradient of function to be minimized
    h: hessian of function to be minimized
    s: search direction
    gtol: break criterion
    ======================================================================
    '''
    G=1
    a=0
    i=0
    if disp:
        print('Newton line search')
    while abs(G) > gtol:
        if disp:
            print('%d a=%g' %(i,a))
        G=np.dot(g(x+a*s),s)
        H=np.dot(np.dot(h(x+a*s),s),s)
        da=-G/H
        a=a+da
        i+=1
    if disp:
        print('%d a=%g' %(i,a))
    return a
  
def armijo_backtracking(x,f,g,s,a=1,c1=1e-4,tau=0.7,disp=True):
    '''
    ======================================================================
    armijo backtracking algorithm
    find a suitable step length a, which leads to a sufficient decrease
    f(x+a*s) < f(x)
    ----------------------------------------------------------------------
    x: start point
    f: function to be minimized (callable)
    g: gradient of f (callable)
    s: search direction
    a: initial step length
    c1: armijo parameter  0 < c1
    ----------------------------------------------------------------------
    return:
    a: acceptable step size (fullfilling the armijo criteria)
    ======================================================================
    '''
    i=1
    while f(x+a*s) > f(x)+c1*a*np.dot(g(x),s):
        a=tau*a
        i+=1
    if disp:  
        print('%d armijo backtracking steps a=%g' %(i,a))
    return a

def strongWolfe(x,func,grad,srch,a1=1,c1=1e-4,c2=0.4,c3=1.5,amax=10,imax=10,jmax=20):
    '''
    ======================================================================
    line search algorithm which fulfilles the strong Wolfe conditions
    see Nocedal Algorithm 3.2 p.59
    ----------------------------------------------------------------------
    x: start point
    func: function to be minimized (callable)
    grad: gradient of f (callable)
    srch: search direction
    a1: initial step length
    c1: constant for armijo condition (default c1=1e-4)
    c2: constant for curvature condition 0 < c1 < c2 < 1 (for Newton c2=0.9, for CG c2 < 0.5)
    c3: constant for increasing step length (default c3=2)
    amax: maximum step size
    imax: maximum number of increments
    jmax: maximum number of zoom increments
    ====================================================================== 
    '''

    def interpolate(jlo,jhi):
        '''
        ======================================================================
        choose interpolation scheme accoriding to available information
        ----------------------------------------------------------------------
        jlo: index where f[jlo]<=f[jhi]
        jhi: index where f[jhi]>=f[jlo]
        --------------------------------------------------------------------
        a[0,...] list of points
        f[0,...] list of function values
        g[0,...] list of gradient values
        ====================================================================
        '''
        print('-----------')
        print('interpolate(jlo=%d,jhi=%d)' %(jlo,jhi))
        print('a:',a)
        print('f:',f)
        print('g:',g)
        #print('a[jlo]:',a[jlo])
        #print('f[jlo]:',f[jlo])
        #print('g[jlo]:',g[jlo])
        #print('a[jhi]:',a[jhi])
        #print('f[jhi]:',f[jhi])
        #print('g[jhi]:',g[jhi])
        if len(a)==3 and not g[1]:                                      # we have a0=0,a1,f0,f1,g0
            aopt=ipq(a[1],f[0],f[1],g[0])                               # quadratic interpolation
            print('ipq: aopt=%f' %(aopt))
            return aopt
        if len(a)>3 and not g[jhi]:                                     # we have a0=0,a1,a2,f0,f1,f2,g0
            if len(a)>4: 
                aopt=ipc1(a[-3],a[-2],f[0],f[-3],f[-2],g[0])            # cubic interpolation f[-1]=None, g[-1]=None
            else:
                aopt=ipc1(a[1],a[-2],f[0],f[1],f[-2],g[0])              # cubic interpolation f[-1]=None, g[-1]=None
            print('ipc1: aopt=%f' %(aopt))
            return aopt
        if f[jlo] and f[jhi] and g[jlo] and g[jhi]:                     # we have a0,a1,f0,f1,g0,g1
            if a[jlo]<a[jhi]:
                aopt=ipc2(a[jlo],a[jhi],f[jlo],f[jhi],g[jlo],g[jhi])    # cubic interpolation
            else:
                aopt=ipc2(a[jhi],a[jlo],f[jhi],f[jlo],g[jhi],g[jlo])    # cubic interpolation
            print('ipc2: aopt=%f' %(aopt))
            return aopt

    def zoom(jlo,jhi):
        '''
        ==========================================================
        zoom algorithm for line search with stron wolfe conditions
        see Nocedal Algorithm 3.3 p.60
        ----------------------------------------------------------
        jlo: index where f[jlo]<=f[jhi]
        jhi: index where f[jhi]>=f[jlo]
        ----------------------------------------------------------
        a[0,...] list of points
        f[0,...] list of function values
        g[0,...] list of gradient values
        ==========================================================
        '''
        j=len(a)                                                        # number of known points
        while j<jmax:
            print('zoom j: %d' %(j))
            a.append(None)                                              # append new point to lists
            f.append(None)
            g.append(None)
            a[j]=interpolate(jlo,jhi)                                   # set a[j]
            f[j]=func(x+a[j]*srch)                                      # set f[j]
            if f[j] > f[0] + c1*a[j]*g[0] or f[j] >= f[jlo]:
                print('Armijo conditions not OK')
                print('jhi <- j')
                jhi = j
            else:
                print('Armijo conditions OK')
                g[j]=np.dot(grad(x+a[j]*srch),srch)                     # set g[j]
                print('check Wolfe %f < %f' %(abs(g[j]),-c2*g[0]))
                if abs(g[j]) <= -c2*g[0]:
                    print('Wolfe conditions OK')
                    aopt=a[j]
                    fopt=f[j]
                    gopt=g[j]
                    return aopt,fopt,gopt
                if g[j]*(a[jhi]-a[jlo]) >= 0:
                    print('jhi <- jlo')
                    jhi=jlo
                print('jlo <- j')
                jlo=j
            #print('a:',a)
            #print('f:',f)
            #print('g:',g)
            #print('jlo: %d' %(jlo))
            #print('jhi: %d' %(jhi))
            j+=1
    #
    # start line search
    #
    a=[0]                               # list for points [0,...]
    f=[func(x+a[0]*srch)]               # list of function values [f(x+a[0]*s),...]
    g=[np.dot(grad(x+a[0]*srch),srch)]  # list of gradient values [g(x+a[0]*s)*s,...]
    i=1
    print('------------------------')
    print('strong wolfe line search')
    print('------------------------')
    while i<imax:
        a.append(None)
        f.append(None)
        g.append(None)
        a[i]=a[i-1]+a1
        f[i]=func(x+a[i]*srch)                                              # evaluate function at a[i]
        if f[i] > f[0] + c1*a[1]*g[0] or (f[i] >= f[i-1] and i > 1):        # Armijo not fulfilled, but we have a minimum in [a0,a1]
            print('case a: Armijo not fulfilled')
            aopt,fopt,gopt=zoom(i-1,i)                                        # zoom in until strong wolfe conditions fulfilled
            print('%d aopt=%f fopt=%f gopt=%f' %(i,aopt,fopt,gopt))
            print('----------------------------------------------')
            return aopt,fopt,gopt
        g[i]=np.dot(grad(x+a[i]*srch),srch)                                 # Armijo fulfilled, evaluate gradient at a1
        if abs(g[i]) <= -c2*g[0]:                                           # Wolfe conditions fulfilled (g0 is always < 0)
            print('case b: Wolfe fulfilled')
            aopt=a[i]
            fopt=f[i]
            gopt=g[i]
            print('%d aopt=%f fopt=%f gopt=%f' %(i,aopt,fopt,gopt))
            print('----------------------------------------------')
            return aopt,fopt,gopt
        if g[i] >= 0:                                                       # gradient positive at a1, we have a minimum in [a1,a0]
            print('case c: positive gradient at a1')
            aopt,fopt,gopt=zoom(i,i-1)
            print('%d aopt=%f fopt=%f gopt=%f' %(i,aopt,fopt,gopt))
            print('----------------------------------------------')
            return aopt,fopt,gopt
        print('case d')
        a1=min(c3*a1,amax)
        print('step length increased a=%f f=%f s=%f' %(a1,f[i],np.linalg.norm(srch)))
        i+=1
        

def ipq(x1,f0,f1,g0):
    '''
    ======================================================================
    1D quadratic interpolation 
    ----------------------------------------------------------------------
    x0=0 !!!
    x1: point 1
    f0: function value at point x=x0
    f1: function value at point x=x1
    g0: gradient at point x=0
    ----------------------------------------------------------------------
    return:
    xopt: point where function f(0+x) -> min
    ======================================================================
    '''
    t1=-g0*x1**2
    t2=2*(f1-f0-x1*g0)
    if t2==0:
        print('!!! ipq: t2 == 0 !!!')
        exit()
    xopt=t1/t2
    return xopt
  
def ipc1(x1,x2,f0,f1,f2,g0):
    '''
    ======================================================================
    1D cubic interpolation 
    ----------------------------------------------------------------------
    x0=0 !!!
    x1: point 1
    x2: point 2
    f0: function value at point x=x0
    f1: function value at point x=x1
    f2: function value at point x=x2
    g0: gradient at point x=0
    ----------------------------------------------------------------------
    return:
    xopt: point where function f(0+x) -> min
    ======================================================================
    '''
    k=1/(x1**2*x2**2*(x2-x1))
    A=np.array([[x1**2,-x2**2],[-x1**3,x2**3]])
    b=np.array([f2-f0-g0*x2,f1-f0-g0*x1])
    c=k*np.dot(A,b)
    t1=c[1]**2-3*c[0]*g0
    t2=3*c[0]
    if t1 < 0:
        print('!!! ipc1: t1 < 0 !!!')
        xopt=-c[1]/(3*c[0])
        return xopt
        exit()
    if t2 == 0:
        print('!!! ipc1: t2 == 0 !!!')
        exit()
    xopt=(-c[1]+np.sqrt(c[1]**2-3*c[0]*g0))/(3*c[0])
    return xopt
  
def ipc2(x0,x1,f0,f1,g0,g1):
    '''
    ======================================================================
    1D cubic interpolation 
    ----------------------------------------------------------------------
    x0: point 0
    x1: point 1
    f0: function value at point x=x0
    f1: function value at point x=x1
    g0: gradient at point x=x0
    g1: gradient at point x=x1
    ----------------------------------------------------------------------
    return:
    xopt: point where function f(0+x) -> min
    ======================================================================
    '''
    d1=g0+g1-3*(f0-f1)/(x0-x1)
    t1=d1**2-g0*g1
    if (t1) < 0:
        print('!!! ipc2: d1**2-g0*g1 < 0 !!!',d1**2-g0*g1)
        t1=0
        exit()
    d2=np.sqrt(t1)
    xopt=x1-(x1-x0)*(g1+d2-d1)/(g1-g0+2*d2)
    return xopt

def linesearch(x,f,g,h,gtol=1e-6,maxincr=1000,direction='BFGS',steplength='Wolfe',disp=True,plot=False):
    '''
    ======================================================================
    line search algorithm to find a minimum of a function
    ----------------------------------------------------------------------
    x: start point
    f: function to be minimized
    g: gradient of function to be minimized
    h: hessian of function to be minimized
    gtol: break criterion
    maxincr: maximum number of line search increments
    direction: ['Newton','Steepest','SR1','BFGS']
    steplength: ['Newton','Armijo','Wolfe']
    disp: if True print some information
    ======================================================================
    '''
    eps=1e-8
    ltype='b-o'
    i=0
    xlist=[x]
    res=1
    while res > gtol and i < maxincr:
        G=g(x)  # evaluate gradient
        #
        # determine search direction
        #
        if direction == 'Newton':
            H=h(x)  # evaluate hessian
            s=-np.dot(G,np.linalg.inv(H)) # newton direction
            a=1
        elif direction == 'Steepest':
            s=-G
            a=1                           # steepest decent direction
        elif direction == 'SR1':
            if i==0:
                #
                # for first step
                #
                B=np.eye(len(x))
                y=G
                s=-G
                a=0.001 # small step
            else:
                s=x-xold
                Bs=np.dot(B,s)
                y=G-Gold
                yBs=y-Bs
                syBs=np.dot(s,y-Bs)
                #
                # check if update is necesarry
                #
                if abs(syBs)>eps*np.linalg.norm(s)*np.linalg.norm(yBs):
                    B=B+np.outer(yBs,yBs)/syBs
                    a=1.0
                else:
                    print('no change in gradient')
                #
                # check if all eigenvalues of B are positive
                #
                eigval,eigvec=np.linalg.eig(B)
                detB=np.linalg.det(B)
                if (eigval > 0).all():
                    #
                    # update search direction using approximated hessian B
                    #
                    s=-np.dot(G,np.linalg.inv(B))
                    a=1.
                    ltype='b-o' # blue line
                else:
                    print('B has negative eigenvalues')
                    print('B:',B)
                    print('eigval:',eigval)
                    print('eigvec:',eigvec)
                    print('det(B):',detB)
                    #
                    # steepest decent with small step size
                    #
                    B=np.eye(len(x))
                    s=-G
                    a=1
                    ltype='r-o' # red line
        elif direction == 'BFGS':
            if i==0:
                #
                # for first step
                #
                B=np.eye(len(x))
                y=G
                s=-G
                a=0.001 # small step
            else:
                s=x-xold
                y=G-Gold
                Bs=np.dot(B,s)
                sB=np.dot(s,B)
                sBs=np.dot(s,Bs)
                BssB=np.outer(Bs,sB)
                yy=np.outer(y,y)
                ys=np.dot(y,s)
                if ys>eps:
                    B=B-BssB/sBs+yy/ys
                else:
                    B=B
                a=1.0
            #
            # update s
            #
            s=-np.dot(G,np.linalg.inv(B))
            ltype='b-o'
        else:
           print('no %s method for direction' %(direction))
        #
        # determine step length
        #
        if steplength == 'Newton':
            if direction=='Newton':
                a=0.99
            else:
                a=newtonLine(x,g,h,s,gtol=gtol,disp=True) # determine optimal step length
        elif steplength == 'Armijo':
            a=armijo_backtracking(x,f,g,s,a=a,c1=1e-4,tau=0.7,disp=True)
        elif steplength == 'Wolfe':
            a=1.0
            a,fopt,gnew=strongWolfe(x,f,g,s,a1=a)
        else:
            print('no %s method for steplength' %(direction))   
        #
        # update x
        #
        xold=x
        Gold=G
        x=x+a*s
        G=g(x)
        res=np.linalg.norm(G)
        xlist.append(x)
        i+=1
        if disp:
            print('i=%d x=%s g=%g f=%g' %(i,str(x),res,f(x)))
            #
            # plot
            #
        if plot:
            X=np.array(xlist)[-2:,0]
            Y=np.array(xlist)[-2:,1]
            plt.plot(X,Y,ltype)
    return xlist

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
#
########################################################################
#
if __name__ == '__main__':
    #
    # f1 (quadratic function)
    #
    a11=1
    a22=4
    a12=0.5
    b1=1
    b2=-0.5
    c=0.633333
    A=np.array([[a11,a12],[a12,a22]])
    B=np.array([b1,b2])
    def f1(x):
        F=0.5*(a11*x[0]**2+a22*x[1]**2+2*a12*x[0]*x[1])+b1*x[0]+b2*x[1]+c
        return F
    def g1(x):
        G=np.dot(A,x)+B
        return G
    def h1(x):
        H=A
        return H
    x=np.array([0.5,2.5])
    plotfunc(f1,opt=[-1.13333,0.266667],n=[101,101,11],xlim=[-4,4],ylim=[-4,4],zlim=[0.1,10],levels=[0.1,0.2,0.5,1,2,5,10,20,40],fmt='%.2f',title=r'$f(x) = 0.5*x*A*x + B*x+c$')
    xlist=linesearch(x,f1,g1,h1,direction='BFGS',steplength='Newton')
    X=np.array(xlist)[:,0]
    Y=np.array(xlist)[:,1]
    plt.plot(X,Y,'b-o')
    plt.show()
    #
    # f2 (quartic function)
    # 
    def f2(x):
        F=(x[0]-2*x[1])**2 + (x[0]-2)**4
        return F
    def g2(x):
        dfdx0=2*(2*(x[0]-2)**3+x[0]-2*x[1])
        dfdx1=4*(2*x[1]-x[0])
        G=np.array([dfdx0,dfdx1])
        return G
    def h2(x):
        d2fdx0x0=2*(6*(x[0]-2)**2+1)
        d2fdx0x1=-4
        d2fdx1x0=-4
        d2fdx1x1=8
        H=np.array([[d2fdx0x0,d2fdx0x1],[d2fdx1x0,d2fdx1x1]])
        return H
    x=np.array([0.5,2.5])
    plotfunc(f2,opt=[2,1],n=[101,101,11],xlim=[0,4],ylim=[-1,3],zlim=[0.1,10],levels=[0.1,0.2,0.5,1,2,5,10,20,40],fmt='%.1f',title=r'$f(x)=(x_0-2 x_1)^2 + (x_0-2)^4$')
    xlist=linesearch(x,f2,g2,h2,direction='BFGS',steplength='Steepest')
    X=np.array(xlist)[:,0]
    Y=np.array(xlist)[:,1]
    plt.plot(X,Y,'b-o')
    plt.show()
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
    x=np.array([-0.5,2])
    plotfunc(f3,opt=[1,1],n=[201,201,11],xlim=[-2,2],ylim=[-1,3],zlim=[0.1,1000],levels=[1,3,10,30,60,110,250,500,1000],fmt='%.0f',title=r'$f(x)=(1-x_0)^2 + 100*(x_1-x_0^2)^2$')
    xlist=linesearch(x,f3,g3,h3,direction='Newton',steplength='Wolfe')
    X=np.array(xlist)[:,0]
    Y=np.array(xlist)[:,1]
    plt.plot(X,Y,'b-o')
    plt.show()
    #
    # f4 (quadratic + exponential)
    #
    def f4(x):
        return x[0]**2+x[1]**2+10*np.exp(-x[0]**2-x[1]**2)+x[0]+x[1]
    def g4(x):
        return np.array([-20*x[0]*np.exp(-x[0]**2-x[1]**2)+2*x[0]+1, -20*x[1]*np.exp(-x[0]**2-x[1]**2)+2*x[1]+1])
    def h4(x): 
        return np.array([[2*np.exp(-x[0]**2-x[1]**2)*(np.exp(x[0]**2+x[1]**2)+20*x[0]**2-10),40*x[0]*x[1]*np.exp(-x[0]**2-x[1]**2)],[40*x[0]*x[1]*np.exp(-x[0]**2-x[1]**2),2*np.exp(-x[0]**2-x[1]**2)*(np.exp(x[0]**2+x[1]**2)+20*x[1]**2-10)]])
    x=np.array([0.25,1.35])
    x=np.array([0.00,1.00])
    plotfunc(f4,opt=[-1.19287,-1.19287],n=[101,101,11],xlim=[-3,3],ylim=[-3,3],zlim=[0.1,20],levels=[1.1,1.25,1.5,2,3,4,5,7.5,10,12.5,15,20],fmt='%.2f',title=r'$f(x)=x_0^2+x_1^2+10 exp(-x_0^2-x_1^2)+x_0+x_1$')
    xlist=linesearch(x,f4,g4,h4,direction='BFGS',steplength='Wolfe')
    X=np.array(xlist)[:,0]
    Y=np.array(xlist)[:,1]
    plt.plot(X,Y,'b-o')
    plt.show()

