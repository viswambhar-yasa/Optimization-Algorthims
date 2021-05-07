'''
========================================================================
Golden Section Search
------------------------------------------------------------------------
find the minimum of a 1D-function
========================================================================
'''
import numpy as np
#
# ======================================================================
# golden section search
# ======================================================================
#
def gss(f,x1,x2,xtol=1e-7,verbose=True):
	'''
	======================================================================
	golden section search
	to find the minimum of f in [x1,x2]
	----------------------------------------------------------------------
	parameters:
	f: a strictly unimodal function on [a,b] (callable)
	x1: lower bound
	x2: upper bound
	xtol: termination criteria (accuracy for x)
	verbose: if True print values during iterations
	----------------------------------------------------------------------
	return:
	x: argument where f(x) has its minimum
	----------------------------------------------------------------------
	example:
	--------
	f = lambda x: (x-2)**2
	x = gss(f, 1, 5)
	print(x)
	1.9999999330223426
	======================================================================
	'''
	gr=(np.sqrt(5.0)+1.0)/2.0	# golden ratio
	n=np.log(xtol/(x2-x1))/np.log(1./gr) # number of function evaluations
	if verbose:
		print('='*80)
		print('{:^80}'.format('Golden Section Search'))
		print('{:^80}'.format('need %d function evaluations' %(np.ceil(n))))
		print('='*80)
	x3=x2-(x2-x1)/gr
	x4=x1+(x2-x1)/gr
	f3=f(x3)
	f4=f(x4)
	i=3
	while abs(x4-x3) > xtol:
		if verbose:
			print('i: %2d  x1=%9.6f  x3=%9.6f  x4=%9.6f  x2=%9.6f  f=%9.6f' %(i,x1,x3,x4,x2,(f3+f4)/2))
		if f4 > f3:
			x2=x4
			x4=x3
			f4=f3
			x3=x2-(x2-x1)/gr
			f3=f(x3)
		else:
			x1=x3
			x3=x4
			f3=f4
			x4=x1+(x2-x1)/gr
			f4=f(x4)
		i+=1
	if verbose:
		print('i: %2d  x1=%9.6f  x3=%9.6f  x4=%9.6f  x2=%9.6f  f=%9.6f' %(i,x1,x3,x4,x2,(f3+f4)/2))
		print('='*80)
	return (x1+x2)/2.
#
# ======================================================================
# main
# ======================================================================
#
if __name__ == '__main__':
	#
	# define your objective function
	#
	def f(x):
		y=(x-2)**2
		return y
	#
	# or as lambda function
	#
	# f=lambda x: (x-2)**2
	#
	# initial values for x1 and x2
	#
	x1=1
	x2=5
	#
	# call algorithm
	#
	xopt=gss(f,x1,x2)
	print('xopt=%18.16f f(xopt)=%18.16f' %(xopt,f(xopt)))
	print('='*80)
