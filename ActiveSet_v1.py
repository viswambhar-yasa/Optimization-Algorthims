'''
========================================================================
activeSet.py
------------------------------------------------------------------------
Implementation of the active set strategy
using numpy and sympy
inequality and equality constraints
------------------------------------------------------------------------
==========================
'''
#
# imports
#
import numpy as np              # numerics
import sympy as sp              # symbolic python
import matplotlib.pyplot as plt # plotting
import matplotlib.colors as col # for defining color maps
#
# tubaf colors
#
geo=np.array([224,134,3])/255   # geo (orange)
ing=np.array([35,186,226])/255  # ing (blue)
ene=np.array([181,18,62])/255   # ener (red)
mat=np.array([26,150,43])/255   # math (green)
tubaf = col.LinearSegmentedColormap.from_list('tubaf', [ing,mat,geo,ene],N=256)
#
# print header
#
def printHeader(fsymb,Gsymb,Hsymb,width=80):
    '''
    ======================================================================
    print header
    ----------------------------------------------------------------------
    arguments:
    ----------
    fsymb:       symbolic objective function y=f(x0,...,xn)
    Gsymb:       list of ng inequality constraints G=[g0,...,gng]
    Hsymb:       list of nh equality constraints G=[h0,...,hnh]
    width:       width of printout
    ----------------------------------------------------------------------
    returns:
    --------
    None
    ======================================================================
    '''
    title='Active Set Algorithm'
    print('='*width)
    fmt='{:^'+str(width)+'}'
    print(fmt.format(title))
    print('-'*width)
    print(' minimize objective function')
    print(' ---------------------------')
    print(' f = %s' %fsymb)
    print('-'*width)
    print(' with respect to:')
    print(' ----------------')
    i=0
    for g in Gsymb:
        print(' g%d: %s < 0' %(i,g))
        i+=1
    i=0
    for h in Hsymb:
        print(' h%d: %s = 0' %(i,h))
        i+=1
    print('='*width)
#
# print iteration
#
def printIter(i,X,fX):
    '''
    ======================================================================
    print iteration
    ----------------------------------------------------------------------
    arguments:
    ----------
    i:  iteration number
    X:  point in iteration i
    fX: f(X)
    ----------------------------------------------------------------------
    returns:
    --------
    None
    ======================================================================
    '''
    print(' active set iteration %d' %(i+1))
    print(' ----------------------')
    parstring=' X=['
    for x in X:
        parstring+=' %g' %(x)
    parstring+=' ]'
    parstring+=' f(X)=%g' %(fX)
    print(parstring)
#
# plot function
#
def plotFunc(fsymb,opt=[0,0],n=[101,101,11],xlim=[-4,4],ylim=[-4,4],zlim=[0,100],levels=None,fmt='%.0f',title=r'$f(x)$',figsize=(8,8),dpi=120):
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
    # lamdify
    #
    f=sp.lambdify([x0,x1],fsymb)
    #
    x=np.linspace(xlim[0],xlim[1],n[0])
    y=np.linspace(ylim[0],ylim[1],n[1])
    X,Y=np.meshgrid(x,y)
    Z=np.zeros((n[0],n[1]))
    #
    for i in range(n[0]):
        for j in range(n[1]):
            Z[i,j]=f(X[i,j],Y[i,j])
    #
    if not levels:
        levels=np.linspace(zlim[0],zlim[1],n[2])
    plt.figure(figsize=figsize,dpi=dpi)
    CP=plt.contour(X,Y,Z,cmap=tubaf,levels=levels)
    LP=plt.plot([opt[0]],[opt[1]],color='red',marker='x',linewidth=2,markersize=12,markeredgewidth=2)
    plt.clabel(CP,inline=True,fmt=fmt)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(r'$x_0$')
    plt.ylabel(r'$x_1$')
    plt.title(title+'='+str(fsymb))
    plt.grid()
#
# plot equation
#
def plotEquation(E,x,linestyle='r-',label='g0',nx=101,ny=101,xlim=[-4,4],ylim=[-4,4]):
    '''
    ======================================================================
    plot equation
    ----------------------------------------------------------------------
    arguments:
    ----------
    E: symbolic equation
    x: variable to solve for
    ----------------------------------------------------------------------
    returns:
    --------
    None
    ======================================================================
    '''
    eqn=sp.Eq(E,0)                   # eqn: x0-2 == 0
    xsym=sp.solve(eqn,x)               # x0 = 2
    if x==x0:
        xplot=sp.lambdify(x1,xsym,'numpy')
        XP=[]
        YP=np.linspace(ylim[0],ylim[1],ny)
        for y in YP:
            XP.append(xplot(y))
    elif x==x1:
        xplot=sp.lambdify(x0,xsym,'numpy')
        XP=np.linspace(xlim[0],xlim[1],nx)
        YP=[]
        for x in XP:
            YP.append(xplot(x))
    if label[0]=='g':
        plt.plot(XP,YP,linestyle,label=label+': '+str(E)+'<0')
    else:
        plt.plot(XP,YP,linestyle,label=label+': '+str(E)+'=0')
#
# active set algorithm
#
def activeSetAlgo(X,fsymb,Gsymb,Hsymb,xtol=1e-12,max_setiter=10,verbose=True,width=80):
    '''
    ======================================================================
    active set algorithm using a Newton scheme
    ----------------------------------------------------------------------
    arguments:
    ----------
    X:           start vector [x0,...,xn]
    fsymb:       symbolic objective function y=f(x0,...,xn)
    Gsymb:       list of ng inequality constraints G=[g0,...,gng]
    Hsymb:       list of nh equality constraints G=[h0,...,hnh]
    xtol:        convergence criteria for Newton-Raphson iterations
    max_setiter: maximum number of active set iterations
    verbose:     if True print informations
    width:       width for printout
    ----------------------------------------------------------------------
    returns:
    --------
    Xopt: argmin f(X)
    ======================================================================
    '''
    #
    if verbose:
        printHeader(fsymb,Gsymb,Hsymb)
    #
    # dimensions
    #
    ndim=len(X)     # number of parameters
    ng=len(Gsymb)   # number of inequality constraints
    nh=len(Hsymb)   # number of equality constraints
    #
    # list of points for plotting
    #
    Xlist=[]
    Xlist.append(np.array(X))
    #
    # define an empty active set
    # for adding a set use: activeSet.add(element)
    # for removing a set use: activeSet.discard(element)
    #
    activeSet=set()
    #
    # Lagrange multipliers
    #
    Lg=np.zeros(ng) # for inequality equations
    Lh=np.zeros(nh) # for equality equations
    #
    # gradient of f (symbolic)
    #
    gradfsymb=[sp.diff(fsymb,x0),sp.diff(fsymb,x1)]
    #
    # start active set iterations ##########################################
    #
    i=0
    r=1
    for iset in range(max_setiter):
        #
        # set flags for constraints and Lagrange multipliers
        #
        constraintsOK=False
        lagrangeOK=False
        #
        # evaluate objective function
        #
        fX=fsymb.subs([(x0,X[0]),(x1,X[1])])
        #
        # print information about current iteration
        #
        if verbose:
            printIter(iset,X,fX)
        #
        # evaluate constraints and extend active set
        #
        fgimax=-1.
        gimax=0
        if verbose:
            print(' -----------------------------------')
            print(' check constraints for inactive sets')
            print(' -----------------------------------')
        for gi in range(ng):
            #
            # for all inactive sets
            #
            if gi not in sorted(activeSet):
                #
                # evalute constraints
                #
                fgi=Gsymb[gi].subs([(x0,X[0]),(x1,X[1])])
                if verbose:
                    print(' g%d(X)=%g' %(gi,fgi))
                if fgi > fgimax:
                    fgimax=fgi
                    gimax=gi
        if fgimax > 0:
            activeSet.add(gimax)
            if verbose:
                print(' extending active set to A=%s' %(str(activeSet)))
        else:
            if verbose:
                print(' all constraints fullfilled, keeping active set A=%s' %(str(activeSet)))
            constraintsOK=True
        #
        # evaluate Lagrange multipliers and reduce active set
        #
        lgimin=np.inf
        limin=0
        if verbose:
            print(' ----------------------------------------')
            print(' check Lagrange multipliers of active set')
            print(' ----------------------------------------')
        for gi in range(ng):
            #
            # for all active sets
            #
            if gi in sorted(activeSet):
                if verbose:
                    print(' Lg%d=%g' %(gi,Lg[gi]))
                if Lg[gi] < lgimin:
                    lgimin=Lg[gi]
                    limin=gi
        if lgimin < 0:
            activeSet.discard(limin)
            if verbose:
                print(' reduce active set to A=%s' %(str(activeSet)))
        else:
            if verbose:
                print(' no Lagrange multipliers < 0, keeping active set A=%s' %(str(activeSet)))
            lagrangeOK=True
        #
        # check termination criteria
        #
        if constraintsOK and lagrangeOK and r < xtol:
            if verbose:
                print(' convergence criteria fullfilled => Exit')
                print('-'*width)
                parstring=' Xopt=['
                for x in X:
                    parstring+=' %g' %(x)
                parstring+=' ]  f(Xopt)=%g' %(fsymb.subs([(x0,X[0]),(x1,X[0])]))
                
                print(parstring)
                print('='*width)
            return Xlist
        #
        # Newton-Raphson iteration
        #
        nri=0
        r=1
        while r > xtol:
            nri+=1
            if verbose:
                print(' ---------------------------')
                print(' Newton-Raphson iteration %2d' %(nri))
                print(' ---------------------------')
            #
            # gradients of active constraints
            #
            nga=len(activeSet)          # number of active inequality constraints
            dgA=np.zeros([nga+nh,ndim]) # initialize dgA
            dX=[x0,x1]
            for i in range(nga):
                for j in range(ndim):
                    gi=sorted(activeSet)[i]
                    dgAsymb=sp.diff(Gsymb[gi],dX[j])
                    dgA[i,j]=dgAsymb.subs([(x0,X[0]),(x1,X[1])])
            for i in range(nh):
                for j in range(ndim):
                    dgAsymb=sp.diff(Hsymb[i],dX[j])
                    dgA[nga+i,j]=dgAsymb.subs([(x0,X[0]),(x1,X[1])])
            if verbose:
                print(' dgA=\n',dgA)
            #
            # assemble Lagrange functional
            #
            Lf=fsymb
            for gi in sorted(activeSet):
                Lf=Lf+Lg[gi]*Gsymb[gi]
            for hi in range(nh):
                Lf=Lf+Lh[hi]*Hsymb[hi]
            if verbose:
                print(' Lf=\n',Lf)
            #
            # define Hessian
            #
            Hess=np.zeros([ndim,ndim])
            for i in range(ndim):
                for j in range(ndim):
                    Hsymb_ij=sp.diff(Lf,dX[i],dX[j])
                    Hess[i,j]=Hsymb_ij.subs([(x0,X[0]),(x1,X[1])])
            Hinv=np.linalg.inv(Hess)
            if verbose:
                print(' Hess=\n',Hess)
                print(' Hinv=\n',Hinv)
            #
            # gradf
            #
            gradf=np.zeros(ndim)
            for i in range(ndim):
                gradf[i]=gradfsymb[i].subs([(x0,X[0]),(x1,X[1])])
            if verbose:
                print(' gradf=\n',gradf)
            #
            # update
            #
            if nga+nh > 0:
                #
                # compute A
                #
                A=np.einsum('ki,ij,lj->kl',dgA,Hinv,dgA)    # dgA^T Hinv dgA
                if verbose:
                    print(' A=\n',A)
                #
                # gA
                #
                gA=np.zeros(nga+nh)
                for i in range(nga):
                    gi=sorted(activeSet)[i]
                    gA[i]=Gsymb[gi].subs([(x0,X[0]),(x1,X[1])])
                for i in range(nh):
                    gA[nga+i]=Hsymb[i].subs([(x0,X[0]),(x1,X[1])])
                if verbose:
                    print(' gA=\n',gA)
                #
                # compute b
                #
                deltagA=np.einsum('kj,ij,i->k',dgA,Hinv,gradf)
                b=gA-deltagA
                if verbose:
                    print(' b=\n',b)
                #
                # compute active Lagrange multpliers
                #
                Ainv=np.linalg.inv(A)
                Ltemp=np.einsum('ij,j->i',Ainv,b)
                if verbose:
                    print(' Ainv=\n', Ainv)
                    print(' Ltemp=\n', Ltemp)
                #
                # sort into all Lagrange multipliers
                #
                for i in range(nga):
                    gi=sorted(activeSet)[i]
                    Lg[gi]=Ltemp[i]
                for i in range(nh):
                    Lh[i]=Ltemp[nga+i]
                if verbose:
                    print(' Lg=\n',Lg)
                    print(' Lh=\n',Lh)
                #
                # update x
                #
                deltaX=np.einsum('ij,j->i',Hinv,gradf+np.einsum('ij,i->j',dgA,Ltemp))
                Xnew=X-deltaX
            else:
                deltaX=np.inner(Hinv,gradf)
                Xnew=X-deltaX
            if verbose:
                print(' Xnew=\n',Xnew)
            #
            # residuum
            #
            r=np.linalg.norm(Xnew-X)
            if verbose:
                print(' residuum: r=%g' %(r))
            X=Xnew
            Xlist.append(X)
        if verbose:
            print('-'*width)

if __name__ == '__main__':
    #
    # convergence criteria
    #
    max_setiter=10  # maximum number of active set iterations
    xtol=1e-8       # convergence criterion for residuum of Newton-Raphson iteration
    #
    # define symbolic variables
    #
    x0,x1,g0,g1,g2,h0=sp.symbols('x0 x1 g0 g1 g2 h0')
    #
    # define objective function (symbolic)
    #
    fsymb=(x0+1)**2+(x1-2)**2               # function 1
    #fsymb=sp.exp(x0*x0)*sp.exp(x1*x1)      # function 2
    #fsymb=(1-x0)**2 + 100*(x1-x0**2)**2    # function 3
    #
    # define inequality constraints (symbolic)
    #
    g0=x0-3
    g1=-x1-2
    g2=x1-x0-1
    Gsymb=[g0,g1,g2]
    #
    # define equality constraints (symbolic)
    #
    h0=x0+x1
    Hsymb=[]  # empty -> no equality constraints active
    #
    # start vector
    #
    X=[-3,-3]
    #
    Xlist=activeSetAlgo(X,fsymb,Gsymb,Hsymb,verbose=True)
    #
    # plot
    #
    plotFunc(fsymb,opt=[-1,2])
    plotEquation(Gsymb[0],x0,linestyle='r-',label='g0')
    plotEquation(Gsymb[1],x1,linestyle='g-',label='g1')
    plotEquation(Gsymb[2],x0,linestyle='b-',label='g2')
    #plotEquation(Hsymb[0],x1,linestyle='m-',label='h0')
    X=np.array(Xlist)[:,0]
    Y=np.array(Xlist)[:,1]
    plt.plot(X,Y,'b-o')
    plt.legend()
    plt.show()
