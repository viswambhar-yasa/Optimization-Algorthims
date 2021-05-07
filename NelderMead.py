'''
========================================================================
Nelder-Mead optimization method for n-dimensional smooth functions
------------------------------------------------------------------------
see: Nocedal, Wright, Numerical Optimization, 2nd Ed. (2006), Ch. 9.5
========================================================================
'''
#
# ======================================================================
# imports
# ----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as col
# ======================================================================
#
plt.ion()	# interactive plotting on
#
# ======================================================================
# define tubaf color map
# ======================================================================
#
geo=np.array([224,134,3])/255		# geo (orange)
ing=np.array([35,186,226])/255	# ing (blue)
ene=np.array([181,18,62])/255		# ener (red)
mat=np.array([26,150,43])/255		# math (green)
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
def f2d(x):
	'''
	======================================================================
	2d objective function, xopt=[2,1] f(xopt)=0
	----------------------------------------------------------------------
	x: parameter vector x=np.array([x0,x1])
	======================================================================
	'''
	f=(x[0]-2*x[1])**2 + (x[0]-2)**4
	return f
#
# ======================================================================
# 2d test function
# ======================================================================
#
def test(x):
	f=(x[0]-2*x[1])**2 + np.sin(x[0])+np.cos(x[1])
	return f
#
# ======================================================================
# print and plot functions
# ======================================================================
#
def printHeader(x,title,f,disp):
	'''
	======================================================================
	print header and initial value of x and f(x)
	----------------------------------------------------------------------
	x: start parameter np.array([x0,...,xn])
	title: title string for header
	f: function to minimize (callable)
	disp: if True header will be printed
	======================================================================
	'''
	width=4+len(x)*11+23
	if disp:
		print('='*width)
		fmt='{:^'+str(width)+'}'
		print(fmt.format(title))
		print('='*width)
		parstring='{:>4}'.format('i')
		for i in range(len(x)):
			parstring+='{:^11}'.format('   x['+str(i)+']')
		parstring+='{:^23}'.format('f(x)')
		print(parstring)
		print('-'*width)
		parstring=''
		for xi in x:
			parstring+=' %10.6f' %(xi)
		print('%4d%s %22.15e' %(0, parstring, f))
#
def printIter(i,x,f,disp):
	'''
	======================================================================
	print iteration result value of x and f(x)
	----------------------------------------------------------------------
	i: iteration number 
	x: parameter vector np.array([x0,...,xn])
	f: function to minimize (callable)
	disp: if True iteration will be printed
	======================================================================
	'''
	if disp:
		parstring=''
		for xi in x:
			parstring+=' %10.6f' %(xi)
		print('%4d%s %22.15e' %(i, parstring, f))
#
def printFooter(i,x,f,disp):
	'''
	======================================================================
	print final result value of x and f(x)
	----------------------------------------------------------------------
	i: iteration number 
	x: parameter vector np.array([x0,...,xn])
	f: function to minimize (callable)
	disp: if True iteration will be printed
	======================================================================
	'''
	width=4+len(x)*11+23
	if disp:
		print('-'*width)
		printIter(i,x,f,disp)
		print('='*width)
#
def plotIter(xlist,plot,anim):
	'''
	======================================================================
	plot iteration in matplotlib
	works only for 2d functions
	----------------------------------------------------------------------
	xlist: list of x values [[x1[0],x1[1]],...,[xn[0],xn[1]]]
	plot: if True plot is done
	anim: if True plot will be animated (slower)
	======================================================================
	'''
	if plot:
		if len(xlist[0])==2:
			#
			# plot the last line segment
			#
			x1=xlist[-2][0]									# x of second last point
			y1=xlist[-2][1]									# y of second last point
			x2=xlist[-1][0]									# x of last point
			y2=xlist[-1][1]									# y of last point
			plt.plot([x1,x2],[y1,y2],'b-o') # plot blue line with dots
			plt.draw()
			#
			# animated plot (slow)
			#
			if anim:
				plt.pause(0.0000001)
		else:
			print('plot only for 2d problems')
#
# ======================================================================
# plot 2d test function
# ======================================================================
#
def plotf2d(figsize=(8,8),dpi=72):
	'''
	======================================================================
	plot 2d objective function as contour plot
	----------------------------------------------------------------------
	figsize: (xsize,ysize) in inches
	dpi: screen resolution in dots per inches (adapt to your screen) 
	======================================================================
	'''
	#
	levels=[0.04,0.2,1,2,5,10,20,30,40,60,80,100,150,200,250]
	nx=101
	ny=101
	xmin=0
	xmax=6
	ymin=0
	ymax=6
	xopt=2
	yopt=1
	#
	x=np.linspace(xmin,xmax,nx)
	y=np.linspace(ymin,ymax,ny)
	X,Y=np.meshgrid(x,y)
	Z=np.zeros(X.shape)
	'''
	for i in range(nx):
		for j in range(ny):
			Z[i,j]=f2d([X[i,j],Y[i,j]])
	'''
	Z=f2d([X,Y])
	#
	plt.figure(figsize=figsize,dpi=dpi)
	CP=plt.contour(X,Y,Z,cmap=tubaf,levels=levels)
	LP=plt.plot([xopt],[yopt],color='red',marker='x',linewidth=2,markersize=12,markeredgewidth=2)
	plt.clabel(CP,inline=True,fmt='%g')
	plt.xlabel(r'$x_0$', fontsize=0.2*dpi)
	plt.ylabel(r'$x_1$', fontsize=0.2*dpi)
	plt.show()
#
# ======================================================================
# plot simplex
# ======================================================================
#
def plotSimplex(XS,plotSim=True,ask=True):
	'''
	======================================================================
	plot simplex (only 2d functions)
	----------------------------------------------------------------------
	XS: sorted coordinates of simplex corners
	plotSim: [True, False]
	======================================================================
	'''
	n=len(XS)-1
	if n != 2:
		print('plotSimplex works only for 2d functions')
		return 0
	#
	if plotSim:
		#
		# simplex lines
		#
		X=list(XS[:,0])
		Y=list(XS[:,1])
		X.append(XS[0,0])
		Y.append(XS[0,1])
		plt.plot(X,Y,'b-o')
		plt.plot([XS[-1,0],XS[-1,0]],[XS[-1,1],XS[-1,1]],'ro')
		plt.draw()
		plt.pause(0.0001)
		if ask:
			inp=input('press 0 to supress plotting simplex, any other key to continue\n')
			if inp=='0':
				plotSim=False
		return plotSim
#
# ======================================================================
# Nelder-Mead
# ======================================================================
#
def simplex(x,f,dx=1.0,max_iter=100,tol=1e-6,disp=True,plot=True,anim=True,plotSim=True):
	'''
	======================================================================
	Nelder-Mead Simplex Algorithm
	find minimum of function f
	----------------------------------------------------------------------
	x: start parameter np.array([x0,...,xn])
	f: function to minimize (callable)
	dx: size for initial simplex
	max_iter: maximum number of iterations
	tol: stop criteria dx < tol
	disp: if True display each iteration in console
	plot: if True show each iteration in matplotlib (only 2D)
	anim: if True matplotlib is animated (only 2D)
	======================================================================
	'''
	#
	# parametric
	#
	def ft(t,f,xm,xn):
		'''
		=============================================================
		parametric objective function along direction f(xm+t*(xn-xm))
		-------------------------------------------------------------
		t: parameter 
		f: function (callable)
		xm: mean point of x
		xn: worst point
		-------------------------------------------------------------
		return:
		xt=xm+t*(xn-xm)
		f(xt)
		=============================================================
		'''
		xt=xm+t*(xn-xm)
		return xt,f(xt)
	#
	# update simplex
	#
	def updateSimplex(XS,FS,xa,fa,n):
		'''
		=========================================
		update simplex
		-----------------------------------------
		XS: sorted simplex corners
		FS: sorted function values
		xa: corner to updated
		fa: function value to be updated
		n: index of corner to be updated
		-----------------------------------------
		return:
		df: maximum difference of function values
		dl: maximum length of simplex
		=========================================
		'''
		XS[n]=xa	# replace last corner by xa
		FS[n]=fa	# replace last function value by fa
		df=FS.max()-FS.min()
		dl=np.linalg.norm((XS[1:n]-XS[0]),axis=0).max()
		return df,dl
	#
	# initial simplex
	#
	n=len(x)
	X=np.zeros((n+1,n))
	F=np.zeros(n+1)
	for i in range(n+1):
		X[i]=x
		if i > 0:
			X[i,i-1]+=dx
		#
		# evalute objective function for simplex corners
		#
		F[i]=f(X[i])
	#
	# sort initial simplex
	#
	IS=np.argsort(F)		# return idizes for sorted function values
	FS=np.sort(F)				# return sorted function values
	XS=np.zeros((n+1,n))	# sort parameter vectors
	XS=X[IS]					# new sorted simplex
	xlist=[]				# list of parameter vectors (for plot)
	xlist.append(np.copy(XS[0]))
	#
	# df - max. difference for objective function
	# dl - max. length of simplex edges
	#
	df=FS.max()-FS.min()
	dl=np.linalg.norm((XS[1:n]-XS[0]),axis=0).max()
	#
	# print header
	#
	printHeader(X[IS[0]],'Nelder-Mead',FS[0],disp)
	#
	# iterate
	#
	j=0
	while dl > tol:
		j+=1
		if j > max_iter:
			print('maximum number of simplex iterations %d exceeded' %(max_iter))
			break				# exit while loop
		#
		# sort F,X
		#
		IS=np.argsort(FS)	# sorted indizes
		FS=np.sort(FS)		# sorted function values
		X=np.copy(XS)			# !!! we need a copy !!!
		XS=X[IS]					# new sorted simplex
		#
		# mean point
		#
		xm=XS[0:n].sum(axis=0)/n
		#
		# best point
		#
		x0=XS[0]
		xlist.append(np.copy(XS[0]))
		#
		# worst point
		#
		xn=XS[n]
		#
		# print and plot
		#
		printIter(j,XS[0],FS[0],disp)
		plotIter(xlist,plot,anim)
		plotSim=plotSimplex(XS,plotSim=plotSim)
		#
		# a) t=-1
		#
		t=-1
		xa,fa=ft(t,f,xm,xn)
		if  FS[0] <= fa and fa < FS[n-1]:
			#
			# update
			#
			df,dl=updateSimplex(XS,FS,xa,fa,n)
			continue
		#
		elif fa < FS[0]:
			#
			# b) t=-2
			#
			t=-2
			xb,fb=ft(t,f,xm,xn)
			if fb < fa:
				#
				# update
				#
				df,dl=updateSimplex(XS,FS,xb,fb,n)
				continue
			else:
				#
				# update
				#
				df,dl=updateSimplex(XS,FS,xa,fa,n)
				continue
		#
		elif fa >= FS[n-1]:
			if FS[n-1] <= fa and fa < FS[n]:
				#
				# c) t=-0.5
				#
				t=-0.5
				xc,fc=ft(t,f,xm,xn)
				if fc <= fa:	
					#
					# update
					#
					df,dl=updateSimplex(XS,FS,xc,fc,n)
					continue
			else:
				#
				# d) t=0.5
				#
				t=0.5
				xd,fd=ft(t,f,xm,xn)
				if fd < FS[n]:
					#
					# update
					#
					df,dl=updateSimplex(XS,FS,xd,fd,n)
					continue
			#
			# e) shrink
			#
			for i in range(1,n+1):
				XS[i]=0.5*(XS[0]+XS[i])
				FS[i]=f(XS[i])
				df=FS.max()-FS.min()
				dl=np.linalg.norm((XS[1:n]-XS[0]),axis=0).max()
	#
	# final value
	#
	j+=1
	xlist.append(np.copy(XS[0]))
	printFooter(j,XS[0],FS[0],disp)
	plotIter(xlist,plot,anim)
#
# ======================================================================
# main
# ======================================================================
#
if __name__ == '__main__':
	#
	# 2d test function
	#
	plotf2d(dpi=120)
	x0=[4,4]
	simplex(x0,f2d,dx=1.0,max_iter=120,anim=0,plotSim=1)
	input('\nPress Key to Finish\n')
