'''
========================================================================
Nonlinear least square solvers
------------------------------------------------------------------------
solves nonlinear least square problems
========================================================================
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as col

np.set_printoptions(precision=6,linewidth=160)

geo=np.array([224,134,3])/255   # geo (orange)
ing=np.array([35,186,226])/255  # ing (blue)
ene=np.array([181,18,62])/255   # ener (red)
mat=np.array([26,150,43])/255   # math (green)
tubaf = col.LinearSegmentedColormap.from_list('tubaf', [ing,mat,geo,ene],N=256)

def quadraticModel(dx,f0,g0,h0,xs=None):
    '''
    ======================================================================
    quadratic model
    ----------------------------------------------------------------------
    dx: step
    f0: function value f(dx=0)
    g0: gradient g(dx=0)
    h0: Hessian h(dx=0)
    xs: scaling matrix
    ----------------------------------------------------------------------
    return
    f1: function f(dx)
    ======================================================================
    '''
    #xs=np.sqrt(xs/xs.min())
    #dx=dx/xs
    f1=f0+np.dot(g0,dx)+0.5*np.dot(dx,np.dot(h0,dx))
    return f1

def lm(x,func,jac,t,y,l=0.001,ftol=1e-6,eta=0.125,max_iter=100,lambda_iter=10,disp=True):
    '''
    ======================================================================
    Levenberg-Marquardt method for solving 
    nonlinear least square problems
    ----------------------------------------------------------------------
    x: start parameter np.array([x1,...,xn])
    func: function for for computing residuals r=func(x,t)-y
    jac: function for jacobian which returns np.array([[dri/dxj,...,drm/dxn)
    t: data points np.array([t1,...,tm])
    y: data values np.array([y1,...,ym])
    ftol: termination criterion
    eta: accuracy to accept step
    max_iter: maximum_number of iterations
    lambda_iter: maximum_number of lambda adjustions
    disp: if True print informations about each iteration
    ======================================================================
    '''
    #
    if disp:
        print('='*80)
        print('Nonlinear Least Squares Levenberg-Marquardt Algorithm')
    #
    m=len(t)  # number of data points
    n=len(x)  # number of parameters
    #
    xs=x                        # save xs as scale values
    #
    # start
    #
    std=np.std(y)               # standard deviation (used for normalization)
    x0=x
    r0=(func(x0,t)-y)/std       # compute normalized residuals
    f0=0.5*(r0**2).sum()        # compute sum of squares
    J0=jac(x0,t)/std            # get the normalized jacobian
    g0=np.inner(J0,r0)          # get the gradient
    normg=np.linalg.norm(g0)    # norm of the gradient
    #
    i=1
    j=1
    df=1
    while abs(df) > ftol and i <= max_iter and j <= lambda_iter: # !!! instead of ftol we could also use gtol (np.linalg.norm(g0) > gtol)
        #
        # try to compute a full step
        # by solving (J.T*J + l*I)*dx=-J.T*res
        # which is equivalent to solve the linear least square problem
        # min_x norm([J sqrt(l*I)].T*dx + [res 0].T)**2
        #
        # for bad scaled problems replace the unit matrix I (np.eye(n)) by a diagonal matrix Dn, 
        # where the diagonal elements reflects typical parameter values 
        #        
        sql=np.sqrt(l)
        sqlI=sql*np.eye(n)
        #
        # scale diagonal matrix
        #
        #sqlI=sql*np.inner(J0,J0)*np.eye(n) 
        #for k in range(n):
        #    sqlI[k,k]=sqlI[k,k]/xs[k]*max(xs)
        #
        JsqlI=np.hstack([J0,sqlI])  # left hand side
        #
        z=np.zeros(n)
        rz=np.hstack([r0,z])  # right hand side
        #
        # call lstsq (linear least squares)
        #
        (dx,resi,rank,sv)=np.linalg.lstsq(JsqlI.T,-rz,rcond=-1) 
        normdx=np.linalg.norm(dx)
        #
        # update point, residuals, objective function, Hessian and model
        #
        x1=x0+dx
        r1=(func(x1,t)-y)/std
        f1=0.5*(r1**2).sum()
        J1=jac(x1,t)/std
        g1=np.inner(J1,r1)
        normg=np.linalg.norm(g1)
        #
        # model
        #
        g0=np.inner(J0,r0)  # gradient for least squares
        h0=np.inner(J0,J0)  # hessian approximation for least squares
        m0=f0
        m1=quadraticModel(dx,f0,g0,h0,xs=xs)
        #
        # accuracy
        #
        a=(f0-f1)/(m0-m1)
        df=f1-f0
        #
        # print
        #
        if disp:
            print('='*80)
            print('LM iteration %d/%d' %(i,j))
            print('-'*16)
            print('       l=%g' %(l))
            print('-----------------------------------------------------------')
            print('      x0=%s' %(str(x0)))
            print('      f0=%g' %(f0))
            print('      m0=%g' %(m0))
            print('-----------------------------------------------------------')
            print('      x1=%s' %(str(x1)))
            print('      f1=%g' %(f1))
            print('      m1=%g' %(m1))
            print('-----------------------------------------------------------')
            print('       a=%g' %(a))
            print('-----------------------------------------------------------')
            print('      dx=%s' %(str(dx)))
            print('  ||dx||=%g' %(normdx))
            print('      df=%g' %(df))
            print('   ||g||=%g' %(normg))
            print('-'*80)
        #
        # adjust lambda (like in the trust region algorithm)
        #
        if a < 0.25:
            l=10.*l
            if disp:
                print('increase lambda to l=%g' %(l))
        else:
            if a > 0.75:
                l=l/10.
                if disp:
                    print('decrease lambda to l=%g' %(l))
            else:
                if disp:
                    print('keep lambda to l=%g' %(l))
        if a > eta:
            if disp:
                print('step accepted')
            x0=x1
            r0=r1
            f0=f1
            J0=J1
            j=1
            i+=1
        else:
            if disp:
                print('step rejected')
            df=1
            j+=1
        #
        # warnings 
        # 
        if i > max_iter: 
            if disp:
                print('maximum number of LM iterations %d/%d exceeded' %(i-1,max_iter))
        if j > lambda_iter: 
            if disp:
                print('maximum number of lambda iterations %d/%d exceeded' %(j-1,lambda_iter))
    #
    # end
    #
    if disp:
        print('='*80)
        print('xopt=%s' %(str(x1)))
        print('='*80)
    return x1

def gn(x,func,jac,t,y,ftol=1e-6,max_iter=100,disp=True):
    '''
    ======================================================================
    Gauss-Newton method for solving 
    nonlinear least square problems
    ----------------------------------------------------------------------
    x: start parameter np.array([x1,...,xn])
    func: function for for computing residuals r=func(x,t)-y
    jac: function for jacobian which returns np.array([[dri/dxj,...,drm/dxn)
    t: data points np.array([t1,...,tm])
    y: data values np.array([y1,...,ym])
    ftol: termination criterion
    max_iter: maximum_number of iterations
    disp: if True print informations about each iteration
    ======================================================================
    '''
    if disp:
        print('='*80)
        print('Nonlinear Least Squares Gauss-Newton Algorithm')
    #
    m=len(t)  # number of data points
    n=len(x)  # number of parameters
    #
    # start
    #
    std=np.std(y)               # variance in our data (used for normalization)
    x0=x
    r0=(func(x0,t)-y)/std       # compute normalized residuals
    f0=0.5*(r0**2).sum()        # compute sum of squares
    J0=jac(x0,t)/std            # get the normalized Jacobian
    #
    i=1
    df=1
    while abs(df) > ftol and i < max_iter:
        (dx,resi,rank,sv)=np.linalg.lstsq(J0.T,-r0,rcond=-1)
        normdx=np.linalg.norm(dx)
        #
        # compute step and new values for residuals and objective function
        #
        x1=x0+dx
        r1=(func(x1,t)-y)/std
        f1=0.5*(r1**2).sum()
        df=f1-f0
        #
        # print
        #
        if disp:
            print('='*80)
            print('GN iteration %d' %(i))
            print('-'*16)
            print('      x0=%s' %(str(x0)))
            print('      f0=%g' %(f0))
            print('-----------------------------------------------------------')
            print('      x1=%s' %(str(x1)))
            print('      f1=%g' %(f1))
            print('-----------------------------------------------------------')
            print('      dx=%s' %(str(dx)))
            print('  ||dx||=%g' %(normdx))
            print('      df=%g' %(df))
        #
        # update
        #
        x0=x1
        r0=r1
        f0=f1
        J0=jac(x0,t)/std
        i+=1
        #
        # warnings 
        # 
        if i > max_iter: 
            if disp:
                print('maximum number of GN iterations %d > %d exceeded' %(i,max_iter))
    #
    # end
    #
    if disp:
        print('='*80)
        print('xopt=%s' %(str(x1)))
        print('='*80)
    return x1
  
if __name__ == '__main__':
    #
    # voce hardening #####################################################
    #
    def func_voce(x,epl):
        '''
        ====================================================================
        voce hardening function
        --------------------------------------------------------------------
        x: parameter array [s0,s1,s2,n]
        epl: array of plastic strains [epl_0,...,epl_m]
        --------------------------------------------------------------------
        return
        s: vector of yield stresses [s_0,...,s_m]
        ====================================================================
        '''
        # internal parameter scaling
        # 0 <= x <= 1
        s0=1000 # scale factor for s0
        s1=500  # scale factor for s1
        s2=500  # scale factor for s2
        n=50    # scale factor for n
        s=x[0]*s0+x[1]*s1*epl+x[2]*s2*(1-np.exp(-x[3]*n*epl)) # !!! check for large n !!! 
        return s
    #
    # Jacobian of voce hardening #########################################
    #
    def jac_voce(x,t):
        '''
        ====================================================================
        jacobian for voce hardening function
        --------------------------------------------------------------------
        x: parameter array [s0,s1,s2,n] !!! 0 <= x <= 1
        epl: array of plastic strains [epl_0,...,epl_m]
        --------------------------------------------------------------------
        return
        J: Jacobian [dr_0/dx_1,...,dr_0/dx_n]
                    [ ...           ...     ]
                    [dr_m/dx_1,...,dr_m/dx_n]
        ====================================================================
        '''
        # internal parameter scaling
        # 0 <= x <= 1
        s0=1000 # scale factor for s0
        s1=500  # scale factor for s1
        s2=500  # scale factor for s2
        n=50    # scale factor for n
        m=len(t)
        dsdx0=s0*np.ones(m)
        dsdx1=s1*t
        dsdx2=s2*(1-np.exp(-n*t*x[3]))        # !!! check for large n !!! 
        dsdx3=n*s2*t*x[2]*np.exp(-n*t*x[3])   # !!! check for large n !!! 
        J=np.array([dsdx0,dsdx1,dsdx2,dsdx3])
        return J
    #
    # generate data y ####################################################
    #
    m=101                                         # number of data points
    nl=5                                          # noise level
    t=np.geomspace(0.01,1.,m)                     # data points
    t=np.linspace(0.,1.,m)                      # data points
    x=np.array([0.5,0.5,0.5,0.5])                 # parameters
    noise=np.random.normal(loc=0,scale=nl,size=m) # generate noise
    ynoise=func_voce(x,t)+noise                   # generate data
    #
    # initial guess for both methods
    # (use scaled values between 0...1)
    #
    x0=np.array([0.8,0.7,0.8,0.4])  # initial guess (bad guess is x0=np.array([0.8,0.7,0.8,0.1]))
    #x0=np.array([0.8,0.7,0.8,0.1])
    #
    # Levenberg-Marquardt method #########################################
    #
    xopt=lm(x0,func_voce,jac_voce,t,ynoise,l=0.0001,ftol=1e-6,max_iter=1000,lambda_iter=100,disp=True)
    #
    # generate data for plots
    #
    mplot=101
    tplot=np.linspace(0.,1.,mplot)
    yexact=func_voce(x,tplot)
    yopt=func_voce(xopt,tplot)
    #
    # plot
    #
    plt.title('Nonlinear least squares fit with Levenberg-Marquardt method')
    plt.plot(t,ynoise,'r.',label='data')
    plt.plot(tplot,yexact,'g-',label='exact')
    plt.plot(tplot,yopt,'b-',label='fit')
    plt.grid()
    plt.legend()
    plt.show()
    #
    # Gauss-Newton method ################################################
    #
    xopt=gn(x0,func_voce,jac_voce,t,ynoise,ftol=1e-8,max_iter=1000,disp=True)
    #
    # generate data for plots
    #
    mplot=101
    tplot=np.linspace(0.,1.,mplot)
    yexact=func_voce(x,tplot)
    yopt=func_voce(xopt,tplot)
    #
    # plot
    #
    plt.title('Nonlinear least squares fit with Gauss-Newton method')
    plt.plot(t,ynoise,'r.',label='data')
    plt.plot(tplot,yexact,'g-',label='exact')
    plt.plot(tplot,yopt,'b-',label='fit')
    plt.grid()
    plt.legend()
    plt.show()
