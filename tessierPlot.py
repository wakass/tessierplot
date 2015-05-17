#tessier.py
#tools for plotting all kinds of files, with fiddle control etc

# data = loadFile(...)
# a = plot3DSlices(data,)

import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.use('Qt4Agg')


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.widgets import Button

import pandas as pd
import numpy as np
import math

import re
import os
_moduledir = os.path.dirname(os.path.realpath(__file__))
import tessierStyles as tstyle
from imp import reload #DEBUG
reload(tstyle) #DEBUG

import IPython  
ipy=IPython.get_ipython()
ipy.magic("pylab qt")



from mpl_toolkits.axes_grid1 import make_axes_locatable

class Fiddle:

	def __init__(self,fig):
		self.fig = fig
		self.press = None
		self.Fiddle = False
		self.fiddleOn = False

	def turnon(self, event, current_ax):
		self.Fiddle = True
		toggle_selector.RS = RectangleSelector(current_ax, self.line_select_callback,
									   drawtype='box', useblit=True,
									   button=[3], # don't use middle button
									   minspanx=5, minspany=5,
									   spancoords='pixels')
		plt.connect('key_press_event', toggle_selector)

	def line_select_callback(eclick, erelease):
		'eclick and erelease are the press and release events'
		x1, y1 = eclick.xdata, eclick.ydata
		x2, y2 = erelease.xdata, erelease.ydata
		print("({:3.2f}, {:3.2f}) --> ({:3.2f}, {:3.2f})".format(x1, y1, x2, y2))
		print(" The button you used were: {:s} {:s}".format(eclick.button, erelease.button))

	def connect(self,event):
		'(dis-)connect to all the events we need'
		if(not self.fiddleOn):
			self.cidpress = self.fig.canvas.mpl_connect(
					'button_press_event', self.on_press)
			self.cidrelease = self.fig.canvas.mpl_connect(
					'button_release_event', self.on_release)
			self.cidmotion = self.fig.canvas.mpl_connect(
					'motion_notify_event', self.on_motion)
			self.fiddleOn = True
		else:
			self.fig.canvas.mpl_disconnect(self.cidpress)
			self.fig.canvas.mpl_disconnect(self.cidrelease)
			self.fig.canvas.mpl_disconnect(self.cidmotion)
			self.fiddleOn = False

	def disconnect(self,event):
		'disconnect all the events we needed'
		self.fig.canvas.mpl_disconnect(self.cidpress)
		self.fig.canvas.mpl_disconnect(self.cidrelease)
		self.fig.canvas.mpl_disconnect(self.cidmotion)
		self.fiddleOn = False

	def on_press(self, event):
		'on button press we will see if the mouse is over us and store some data'

		contains, attrd = self.fig.contains(event)
		if not contains: return
		if(event.xdata==None): return
		self.press = event.xdata, event.ydata
		a = event.inaxes.images
		for i in a:
			self.clim = i.get_clim()

	def on_motion(self, event):
		'on motion we will move the rect if the mouse is over us'
		if self.press is None: return
		if(event.xdata==None): return
		#if math.isnan(float(event.xdata)): return
		xpress, ypress = self.press
		dx = event.xdata - xpress
		dy = event.ydata - ypress
		#print('xp={:f}, ypress={:f}, event.xdata={:f},event.ydata={:f}, dx={:f}, dy={:f}'.format(xpress,ypress,event.xdata, event.ydata,dx, dy))
		#self.rect.set_x(x0+dx)
		#self.rect.set_y(y0+dy)


		#dx controls limits width
		#dy controls limits center
		a = event.inaxes.images

		for i in a:
			xmin,xmax,ymin,ymax = i.get_extent()

			vmin,vmax=self.clim
			xrel=abs(xmin - xmax)
			yrel=abs(ymin - ymax)

			newvmin = dy/yrel*(vmin-vmax) + vmin + dx/xrel*(vmin-vmax)
			newvmax = dy/yrel*(vmin-vmax) + vmax - dx/xrel*(vmin-vmax)

			if (newvmax > newvmin):
				i.set_clim(vmax=newvmax, vmin=newvmin)




		self.fig.canvas.draw()


	def on_release(self, event):
		'on release we reset the press data'
		self.press = None

def parseheader(file):
	names = []
	types= []
	all = []
	skipindex = 0
	with open(file) as f:	

		colname = re.compile(r'^#.*?name\:{1}\s(.*?)\r?$')
		type    = re.compile(r'^#.*?type\:{1}\s(.*?)\r?$')
		for i, line in enumerate(f):
			if i < 3:
				continue

			a = colname.findall(line)
			b = type.findall(line)
			if len(a) >= 1:
				names.append(a[0])
			if i > 5:
				if line[0] != '#': #find the skiprows accounting for the first linebreak in the header
					skipindex = i
					break
			if i > 300:
				break
	#nog een keer dunnetjes overdoen met read()
	f = open(file)
	alltext= f.read(skipindex)
	
	with open(file) as myfile:
		alltext = [next(myfile) for x in xrange(skipindex)]
		
	alltext= ''.join(alltext)

	#doregex
	coord_expression = re.compile(r"""
								^\#\s*Column\s(.*?)\:
								[\r\n]{0,2}
								\#\s*end\:\s(.*?)
								[\r\n]{0,2}
								\#\s*name\:\s(.*?)
								[\r\n]{0,2}
								\#\s*size\:\s(.*?)
								[\r\n]{0,2}
								\#\s*start\:\s(.*?)
								[\r\n]{0,2}
								\#\s*type\:\s(.*?)[\r\n]{0,2}$ 
								
								"""#annoying \r's...
								,re.VERBOSE |re.MULTILINE)
	
	val_expression = re.compile(r"""
								^\#\s*Column\s(.*?)\:
								[\r\n]{0,2}
								\#\s*name\:\s(.*?)
								[\r\n]{0,2}
								\#\s*type\:\s(.*?)[\r\n]{0,2}$
								"""
								,re.VERBOSE |re.MULTILINE)
	coord=  coord_expression.findall(alltext) 
	val  = val_expression.findall(alltext)
	coord = [ zip(('column','end','name','size','start','type'),x) for x in coord]
	coord = [dict(x) for x in coord]
	val = [ zip(('column','name','type'),x) for x in val]
	val = [dict(x) for x in val]
	
	all=coord+val
	return names, skipindex#,all

def quickplot(file,**kwargs):
	names,skipindex = parseheader(file)
	data = loadFile(file,names=names,skiprows=skipindex)


	p = plot3DSlices(data,**kwargs)
	return p

def parseUnitAndNameFromColumnName(input):
	reg = re.compile(r'\{(.*?)\}')
	z = reg.findall(input)
	return z

def getnDimFromData(file,data):

		nDim = 0
		names,skiprows,header=parseheader(file)

		cols = data.columns.tolist()
		filterdata = data.sort(cols[:-1])
		filterdata = filterdata.dropna(how='any')
		#first determine the columns belong to the axes (not measure) coordinates
		cols = [i for i in header if (i['type'] == 'coordinate')]

		for i in cols:
			col = getattr(filterdata,i['name'])
			if len(col.unique()) > 1: #do not count 0d axes..(i.e. with only 1 unique value)
				nDim += 1
# 			self.uniques_per_col.append(list(col.unique()))
		return nDim
	
	
def starplot(file,fig=None):
	names,skiprows=parseheader(file)
	dat = loadFile(file,names,skiprows)
	# dat=dat.dropna(how='any') #not doing this somehow prevent a line drawn from the beginning to the end of the line?

	if fig == None:
		fig = plt.figure()
	else:
		fig = plt.figure(fig.number)

	#basically project the first and final dimension
	for i in range(0,len(dat.keys())-1):
		title = parseUnitAndNameFromColumnName(dat.keys()[i])
		ax = plt.subplot(len(dat.keys())+1,1,i+1)
		plt.plot(dat[dat.keys()[i]],(dat[dat.keys()[-1]]),label=title)
		plt.legend(loc=9, bbox_to_anchor=(-0.5, 0.7), ncol=3)
	return fig
	
def scanplot(file,fig=None,n_index=None,style=[],data=None,**kwargs):
	#kwargs go into matplotlib/pyplot plot command
	if not fig:
		fig = plt.figure()
	names,skip = parseheader(file)

	if data is None:
		print('loading')
		data = loadFile(file,names=names,skiprows=skip)

	uniques_col = []

	#scanplot assumes 2d plots with data in the two last columns
	uniques_col_str = names[:-2]

	reg = re.compile(r'\{(.*?)\}')
	parsedcols = []
	#do some filtering of the colstr to get seperate name and unit of said name
	for a in uniques_col_str:
		z = reg.findall(a)
		if len(z) > 0:
			parsedcols.append(z[0])
		else:
			parsedcols.append('')
		#name is first


	for i in uniques_col_str:
		col = getattr(data,i)
		uniques_col.append(col)

	if n_index != None:
		n_index = np.array(n_index)
		nplots = len(n_index)

	for i,j in enumerate(buildLogicals(uniques_col)):
		if n_index != None:
				if i not in n_index:
					continue
		filtereddata = data.loc[j]
		title =''
		for i,z in enumerate(uniques_col_str):
			title = '\n'.join([title, '{:s}: {:g} (mV)'.format(parsedcols[i],getattr(filtereddata,z).iloc[0])])

		measAxisDesignation = parseUnitAndNameFromColumnName(filtereddata.keys()[-1])
		
		x =  np.array(filtereddata.iloc[:,-2])
		y =  np.array(filtereddata.iloc[:,-1])

		wrap = tstyle.getPopulatedWrap(style)
		wrap['XX'] = y
		tstyle.processStyle(style,wrap)
		#print np.shape(x)
		#print np.shape(wrap['XX'])
		p = plt.plot(x,wrap['XX'],label=title,**kwargs)

	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
		   ncol=2, mode="expand", borderaxespad=0.)
		   
	ax = fig.axes[0]
	xaxislabel = parseUnitAndNameFromColumnName(names[-2])
	yaxislabel = parseUnitAndNameFromColumnName(names[-1])
		
	ax.set_xlabel(xaxislabel[0]+'(' + xaxislabel[1] +')')
	ax.set_ylabel(yaxislabel[0]+'(' + yaxislabel[1] +')')
	from IPython.core import display
	#display.display(fig)
	return fig

def loadFile(file,names=['L','B1','B2','vsd','zz'],skiprows=25):
	#print('loading...')
	data = pd.read_csv(file, sep='\t', comment='#',skiprows=skiprows,names=names)
	data.name = file
	return data

def loadCustomColormap(file='./cube1.txt'):
#	xl = pd.ExcelFile(file)
#
#	dfs = {sheet: xl.parse(sheet) for sheet in xl.sheet_names}
#	for i in dfs.keys():
#		r = dfs[i]
#		ls = [r.iloc[:,0],r.iloc[:,1],r.iloc[:,2]]
#		do = zip(*ls)

	do = np.loadtxt(file)

	ccmap = mpl.colors.LinearSegmentedColormap.from_list('name',do)
	return ccmap

def demoCustomColormap():
	ccmap = loadCustomColormap()
	a = np.linspace(0, 1, 256).reshape(1,-1)
	a = np.vstack((a,a))
	fig = plt.figure()
	plt.imshow(a, aspect='auto', cmap=plt.get_cmap(ccmap), origin='lower')
	plt.show()


def moving_average(a, n=3) :
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n

from scipy.signal import lfilter, lfilter_zi, filtfilt, butter
import math


def buildLogicals(xs):
#combine the logical uniques of each column into boolean index over those columns
#infers that each column has
#like
# 1, 3
# 1, 4
# 1, 5
# 2, 3
# 2, 4
# 2, 5
#uniques of first column [1,2], second column [3,4,5]
#go through list and recursively combine all unique values
	if len(xs) > 1:
		for i in xs[0].unique():
			if math.isnan(i):
				continue
			for j in buildLogicals(xs[1:]):
				yield (xs[0] == i) & j ## boolean and
	elif len(xs) == 1:
		for i in xs[0].unique():
			if (math.isnan(i)):
				#print('NaN found, skipping')
				continue
			yield xs[0] == i
	else:
		#empty list
		yield slice(None) #return a 'semicolon' to select all the values when there's no value to filter on


class plot3DSlices:
	fig = None
	data = []
	uniques_col_str=None
	exportData = []
	exportDataMeta =[]
	def show(self):
		plt.show()

	def exportToMtx(self):

		for j, i in enumerate(self.exportData):

			data = i
			print(j)
			m = self.exportDataMeta[j]

			sz = np.shape(data)
			#write
			try:
				fid = open('{:s}{:d}{:s}'.format(self.data.name, j, '.mtx'),'w+')
			except Exception as e:
				print('Couldnt create file: {:s}'.format(str(e)))
				return

			#example of first two lines
			#Units, Data Value at Z = 0.5 ,X, 0.000000e+000, 1.200000e+003,Y, 0.000000e+000, 7.000000e+002,Nothing, 0, 1
			#850 400 1 8
			str1 = 'Units, Name: {:s}, {:s}, {:f}, {:f}, {:s}, {:f}, {:f}, {:s}, {:f}, {:f}\n'.format(
				m['datasetname'],
				m['xname'],
				m['xlims'][0],
				m['xlims'][1],
				m['yname'],
				m['ylims'][0],
				m['ylims'][1],
				m['zname'],
				m['zlims'][0],
				m['zlims'][1]
				)
			floatsize = 8
			str2 = '{:d} {:d} {:d} {:d}\n'.format(m['xu'],m['yu'],1,floatsize)
			fid.write(str1)
			fid.write(str2)
			#reshaped = np.reshape(data,sz[0]*sz[1],1)
			data.tofile(fid)
			fid.close()



	def __init__(self,data,n_index=None,meshgrid=False,hilbert=False,didv=False,fiddle=True,uniques_col_str=[],style='normal',clim=(0,0),aspect='auto',interpolation='none'):
		#uniques_col_str, array of names of the columns that are e.g. the slices of the
		#style, 'normal,didv,didv2,log'
		#clim, limits of the colorplot c axis

		self.exportData =[]
		self.data = data
		self.uniques_col_str=uniques_col_str

		#n_index determines which plot to plot,
		# 0 value for plotting all
		
		

		print('sorting...')
		cols = data.columns.tolist()
		filterdata = data.sort(cols[:-1])
		filterdata = filterdata.dropna(how='any')

		uniques_col = []
		self.uniques_per_col=[]
		
		
		sweepdirection = data[cols[-1]][0] > data[cols[-1]][1] #True is sweep neg to pos


		uniques_col_str = list(uniques_col_str)
		for i in uniques_col_str:
			col = getattr(filterdata,i)
			uniques_col.append(col)
			self.uniques_per_col.append(list(col.unique()))

		self.ccmap = loadCustomColormap()


		#fig,axs = plt.subplots(1,1,sharex=True)
		self.fig = plt.figure()
		self.fig.subplots_adjust(top=0.96, bottom=0.07, left=0.07, right=0.9,hspace=0.0)


		nplots = 1
		for i in self.uniques_per_col:
			nplots *= len(i)

		if n_index != None:
			n_index = np.array(n_index)
			nplots = len(n_index)
			

		cnt=0
		#enumerate over the generated list of unique values specified in the uniques columns
		for j,ind in enumerate(buildLogicals(uniques_col)):
			if n_index != None:
				if j not in n_index:
					continue

			slicy = filterdata.loc[ind]
			#get all the last columns, that we assume contains the to be plotted data
			x=slicy.iloc[:,-3]
			y=slicy.iloc[:,-2]
			z=slicy.iloc[:,-1]

			xu = np.size(x.unique())
			yu = np.size(y.unique())


			## if the measurement is not complete this will probably fail so trim of the final sweep?
			print('xu: {:d}, yu: {:d}, lenz: {:d}'.format(xu,yu,len(z)))

			if xu*yu != len(z):
				xu = (len(z) / yu) #dividing integers so should automatically floor the value

			#trim the first part of the sweep, for different min max, better to trim last part?
			#or the first since there has been sorting
			#this doesnt work for e.g. a hilbert measurement

			print('xu: {:d}, yu: {:d}, lenz: {:d}'.format(xu,yu,len(z)))
			if hilbert:

				Z = np.zeros((xu,yu))

				#make a meshgrid for indexing
				xs = np.linspace(x.min(),x.max(),xu)
				ys = np.linspace(y.min(),y.max(),yu)
				xv,yv = np.meshgrid(xs,ys,sparse=True)
				#evaluate all datapoints
				for i,k in enumerate(xs):
					print(i,k)
					for j,l in enumerate(ys):
						ind = (k == x) & (l == y)
						#print(z[ind.index[0]])
						Z[i,j] = z[ind.index[0]]
				#keep a z array, index with datapoints from meshgrid+eval
				XX = Z
			else:
				#sorting sorts negative to positive, so beware:
				#sweep direction determines which part of array should be cut off
				if sweepdirection:
					z = z[-xu*yu:]
					x = x[-xu*yu:]
					y = y[-xu*yu:]
				else:
					z = z[:xu*yu]
					x = x[:xu*yu]
					y = y[:xu*yu]

				XX = np.reshape(z,(xu,yu))

			self.x = x
			self.y = y
			self.z = z
			#now set the lims
			xlims = (x.min(),x.max())
			ylims = (y.min(),y.max())

			#determine stepsize for di/dv, inprincipe only y step is used (ie. the diff is also taken in this direction and the measurement swept..)
			xstep = (xlims[0] - xlims[1])/xu
			ystep = (ylims[0] - ylims[1])/yu
			ext = xlims+ylims


#             if meshgrid:
#                 X, Y = np.meshgrid(xi, yi)
#                 scipy.interpolate.griddata((xs, ys), Z, X, Y)
#                 Z = griddata(x,y,Z,xi,yi)

			self.XX = XX

			self.exportData.append(XX)
			try:
				m={
					'xu':xu,
					'yu':yu,
					'xlims':xlims,
					'ylims':ylims,
					'zlims':(0,0),
					'xname':cols[-3],
					'yname':cols[-2],
					'zname':'unused',
					'datasetname':data.name}
				self.exportDataMeta = np.append(self.exportDataMeta,m)
			except:
				pass
			print('plotting...')

			ax = plt.subplot(nplots, 1, cnt+1)
			cbar_title = ''

			if didv: #some backwards compatibility
			   style = 'didv'

			if type(style) != list:
				style = list([style])
				#print(style)

			#smooth the datayesplz
			#import scipy.ndimage as ndimage
			#XX = ndimage.gaussian_filter(XX,sigma=1.0,order=0)


			measAxisDesignation = parseUnitAndNameFromColumnName(self.data.keys()[-1])
			#wrap all needed arguments in a datastructure
			cbar_quantity = measAxisDesignation[0]
			cbar_unit = measAxisDesignation[1]
			cbar_trans = [] #trascendental tracer :P For keeping track of logs and stuff

			w = tstyle.getPopulatedWrap(style)
			w2 = {'ext':ext, 'ystep':ystep,'XX': XX, 'cbar_quantity': cbar_quantity, 'cbar_unit': cbar_unit, 'cbar_trans':cbar_trans}
			for k in w2:
				w[k] = w2[k]
			tstyle.processStyle(style, w)

			#unwrap
			ext = w['ext']
			XX = w['XX']
			cbar_trans_formatted = ''.join([''.join(s+'(') for s in w['cbar_trans']])
			cbar_title = cbar_trans_formatted + w['cbar_quantity'] + ' (' + w['cbar_unit'] + ')'
			if len(w['cbar_trans']) is not 0:
				cbar_title = cbar_title + ')'

			#postrotate np.rot90
			XX = np.rot90(XX)

			if 'deinterlace' in style:
				self.fig = plt.figure()
				ax_deinter_odd  = plt.subplot(2, 1, 1)
				w['deinterXXodd'] = np.rot90(w['deinterXXodd'])
				ax_deinter_odd.imshow(w['deinterXXodd'],extent=ext, cmap=plt.get_cmap(self.ccmap),aspect=aspect,interpolation=interpolation)

				ax_deinter_even = plt.subplot(2, 1, 2)
				w['deinterXXeven'] = np.rot90(w['deinterXXeven'])
				ax_deinter_even.imshow(w['deinterXXeven'],extent=ext, cmap=plt.get_cmap(self.ccmap),aspect=aspect,interpolation=interpolation)

			self.im = ax.imshow(XX,extent=ext, cmap=plt.get_cmap(self.ccmap),aspect=aspect,interpolation=interpolation, norm=w['imshow_norm'])
			if clim != (0,0):
			   self.im.set_clim(clim)

			if 'flipaxes' in style:
				ax.set_xlabel(cols[-2])
				ax.set_ylabel(cols[-3])
			else:
				ax.set_xlabel(cols[-3])
				ax.set_ylabel(cols[-2])


			title = ''
			for i in uniques_col_str:
				title = '\n'.join([title, '{:s}: {:g} (mV)'.format(i,getattr(slicy,i).iloc[0])])
			print(title)
			if 'notitle' not in style:
				ax.set_title(title)
			# create an axes on the right side of ax. The width of cax will be 5%
			# of ax and the padding between cax and ax will be fixed at 0.05 inch.
			divider = make_axes_locatable(ax)
			cax = divider.append_axes("right", size="5%", pad=0.05)

			pos = list(ax.get_position().bounds)

			self.cbar = plt.colorbar(self.im, cax=cax)
			cbar = self.cbar

			cbar.set_label(cbar_title)

			cnt+=1 #counter for subplots


		from IPython.core import display
		if mpl.get_backend() == 'Qt4Agg':
			display.display(self.fig)

		if fiddle and (mpl.get_backend() == 'Qt4Agg'):
			self.fiddle = Fiddle(self.fig)
			axFiddle = plt.axes([0.1, 0.85, 0.15, 0.075])


			self.bnext = Button(axFiddle, 'Fiddle')
			self.bnext.on_clicked(self.fiddle.connect)

			#attach to the relevant figure to make sure the object does not go out of scope
			self.fig.fiddle = self.fiddle
			self.fig.bnext = self.bnext
		#plt.show()



class plotR:
	def __init__(self,file,fiddle=True):
		self.header = None
		self.names = None
		self.fig = None
	
		self.exportData =[]		
		self.file = file

		self.data  = self.loadFile(file) 
		self.filterdata = None
		
		self.ndim = self.getnDimFromData()
		self.dims = self.getdimFromData()
		self.bControls = True #boolean controlling state of plot manipulation buttons
		
		
	def quickplot(self,**kwargs):
		nDim = self.ndim
		#if the uniques of a dimension is less than x, plot in consequential 2d, otherwise 3d

		#maybe put logic here to plot some uniques as well from nonsequential axes?
		filter = self.dims < 5
		filter_neg = np.array([not x for x in filter])

		coords = np.array([x['name'] for x in self.header if x['type']=='coordinate'])
		
		uniques_col_str = coords[filter]

		if len(coords[filter_neg]) > 1: 
			print 'plot3d'
			self.plot3d(uniques_col_str=uniques_col_str,**kwargs)
		else:
			print 'plot2d'
			self.plot2d(**kwargs)		

	def plot3d(self,fiddle=True,uniques_col_str=[],n_index=None,style='normal',clim=(0,0),aspect='auto',interpolation='none',**kwargs): #previously plot3dslices
		if not self.fig:
			self.fig = plt.figure()
		
		if self.data is None:
			print('loading')
			self.loadFile(file)
			
		cols = self.data.columns.tolist()
		filterdata = self.sortdata()

		uniques_col = []
		self.uniques_per_col=[]
		
		
		sweepdirection = self.data[cols[-1]][0] > self.data[cols[-1]][1] #True is sweep neg to pos
		
				

		for i in uniques_col_str:
			col = getattr(filterdata,i)
			uniques_col.append(col)
			self.uniques_per_col.append(list(col.unique()))

		self.ccmap = loadCustomColormap()


		self.fig.subplots_adjust(top=0.96, bottom=0.07, left=0.07, right=0.9,hspace=0.0)


		nplots = 1
		for i in self.uniques_per_col:
			nplots *= len(i)

		if n_index != None:
			n_index = np.array(n_index)
			nplots = len(n_index)
			

		cnt=0
		#enumerate over the generated list of unique values specified in the uniques columns
		for j,ind in enumerate(buildLogicals(uniques_col)):
			if n_index != None:
				if j not in n_index:
					continue

			slicy = filterdata.loc[ind]
			#get all the last columns, that we assume contains the to be plotted data
			x=slicy.iloc[:,-3]
			y=slicy.iloc[:,-2]
			z=slicy.iloc[:,-1]

			xu = np.size(x.unique())
			yu = np.size(y.unique())


			## if the measurement is not complete this will probably fail so trim of the final sweep?
			print('xu: {:d}, yu: {:d}, lenz: {:d}'.format(xu,yu,len(z)))

			if xu*yu != len(z):
				xu = (len(z) / yu) #dividing integers so should automatically floor the value

			#trim the first part of the sweep, for different min max, better to trim last part?
			#or the first since there has been sorting
			#this doesnt work for e.g. a hilbert measurement

			print('xu: {:d}, yu: {:d}, lenz: {:d}'.format(xu,yu,len(z)))

			#sorting sorts negative to positive, so beware:
			#sweep direction determines which part of array should be cut off
			if sweepdirection:
				z = z[-xu*yu:]
				x = x[-xu*yu:]
				y = y[-xu*yu:]
			else:
				z = z[:xu*yu]
				x = x[:xu*yu]
				y = y[:xu*yu]

			XX = np.reshape(z,(xu,yu))

			self.x = x
			self.y = y
			self.z = z
			#now set the lims
			xlims = (x.min(),x.max())
			ylims = (y.min(),y.max())

			#determine stepsize for di/dv, inprincipe only y step is used (ie. the diff is also taken in this direction and the measurement swept..)
			xstep = (xlims[0] - xlims[1])/xu
			ystep = (ylims[0] - ylims[1])/yu
			ext = xlims+ylims


			self.XX = XX
			
			self.exportData.append(XX)
			try:
				m={
					'xu':xu,
					'yu':yu,
					'xlims':xlims,
					'ylims':ylims,
					'zlims':(0,0),
					'xname':cols[-3],
					'yname':cols[-2],
					'zname':'unused',
					'datasetname':data.name}
				self.exportDataMeta = np.append(self.exportDataMeta,m)
			except:
				pass

			ax = plt.subplot(nplots, 1, cnt+1)
			cbar_title = ''
			
			
			if type(style) != list:
				style = list([style])

			measAxisDesignation = parseUnitAndNameFromColumnName(self.data.keys()[-1])
			#wrap all needed arguments in a datastructure
			cbar_quantity = measAxisDesignation[0]
			cbar_unit = measAxisDesignation[1]
			cbar_trans = [] #trascendental tracer :P For keeping track of logs and stuff

			w = tstyle.getPopulatedWrap(style)
			w2 = {'ext':ext, 'ystep':ystep,'XX': XX, 'cbar_quantity': cbar_quantity, 'cbar_unit': cbar_unit, 'cbar_trans':cbar_trans}
			for k in w2:
				w[k] = w2[k]
			tstyle.processStyle(style, w)

			#unwrap
			ext = w['ext']
			XX = w['XX']
			cbar_trans_formatted = ''.join([''.join(s+'(') for s in w['cbar_trans']])
			cbar_title = cbar_trans_formatted + w['cbar_quantity'] + ' (' + w['cbar_unit'] + ')'
			if len(w['cbar_trans']) is not 0:
				cbar_title = cbar_title + ')'

			#postrotate np.rot90
			XX = np.rot90(XX)

			if 'deinterlace' in style:
				self.fig = plt.figure()
				ax_deinter_odd  = plt.subplot(2, 1, 1)
				w['deinterXXodd'] = np.rot90(w['deinterXXodd'])
				ax_deinter_odd.imshow(w['deinterXXodd'],extent=ext, cmap=plt.get_cmap(self.ccmap),aspect=aspect,interpolation=interpolation)

				ax_deinter_even = plt.subplot(2, 1, 2)
				w['deinterXXeven'] = np.rot90(w['deinterXXeven'])
				ax_deinter_even.imshow(w['deinterXXeven'],extent=ext, cmap=plt.get_cmap(self.ccmap),aspect=aspect,interpolation=interpolation)

			self.im = ax.imshow(XX,extent=ext, cmap=plt.get_cmap(self.ccmap),aspect=aspect,interpolation=interpolation, norm=w['imshow_norm'])
			if clim != (0,0):
			   self.im.set_clim(clim)

			if 'flipaxes' in style:
				ax.set_xlabel(cols[-2])
				ax.set_ylabel(cols[-3])
			else:
				ax.set_xlabel(cols[-3])
				ax.set_ylabel(cols[-2])


			title = ''
			for i in uniques_col_str:
				title = '\n'.join([title, '{:s}: {:g} (mV)'.format(i,getattr(slicy,i).iloc[0])])
			print(title)
			if 'notitle' not in style:
				ax.set_title(title)
			# create an axes on the right side of ax. The width of cax will be 5%
			# of ax and the padding between cax and ax will be fixed at 0.05 inch.
			divider = make_axes_locatable(ax)
			cax = divider.append_axes("right", size="5%", pad=0.05)

			pos = list(ax.get_position().bounds)

			self.cbar = plt.colorbar(self.im, cax=cax)
			cbar = self.cbar

			cbar.set_label(cbar_title)

			cnt+=1 #counter for subplots
		self.toggleFiddle()
	
	def plot2d(self,n_index=None,style=['normal'],**kwargs): #previously scanplot
# 		scanplot(file,fig=None,n_index=None,style=[],data=None,**kwargs):
		#kwargs go into matplotlib/pyplot plot command
		
		if not self.fig:
			self.fig = plt.figure()
		
		if self.data is None:
			print('loading')
			self.loadFile(file)
	
		uniques_col = []
	
		#scanplot assumes 2d plots with data in the two last columns
		uniques_col_str = [i['name'] for i in self.header if i['type']=='coordinate' ][:-1]    
	
		reg = re.compile(r'\{(.*?)\}')
		parsedcols = []
		#do some filtering of the colstr to get seperate name and unit of said name
		for a in uniques_col_str:
			z = reg.findall(a)
			if len(z) > 0:
				parsedcols.append(z[0])
			else:
				parsedcols.append('')
			#name is first

		
		for i in uniques_col_str:
			col = getattr(self.data,i)
			uniques_col.append(col)

		if n_index != None:
			n_index = np.array(n_index)
			nplots = len(n_index)
		
		for i,j in enumerate(buildLogicals(uniques_col)):
			if n_index != None:
					if i not in n_index:
						continue
			filtereddata = self.data.loc[j]
			title =''
			for i,z in enumerate(uniques_col_str):
				title = '\n'.join([title, '{:s}: {:g} (mV)'.format(parsedcols[i],getattr(filtereddata,z).iloc[0])])

			x =  np.array(filtereddata.iloc[:,-2])
			y =  np.array(filtereddata.iloc[:,-1])
	
			wrap = tstyle.getPopulatedWrap(style)
			wrap['XX'] = y
			tstyle.processStyle(style,wrap)
			ax = plt.plot(x,wrap['XX'],label=title,**kwargs)
				
			
		plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
			   ncol=2, mode="expand", borderaxespad=0.)
		ax = self.fig.axes[0]
		xaxislabel = parseUnitAndNameFromColumnName(self.header[-2]['name'])
		yaxislabel = parseUnitAndNameFromColumnName(self.header[-1]['name'])
		
		ax.set_xlabel(xaxislabel[0]+'(' + xaxislabel[1] +')')
		ax.set_ylabel(yaxislabel[0]+'(' + yaxislabel[1] +')')
		return self.fig

	def sortdata(self,refresh=False):
		#some light caching
		if ((self.filterdata) is None) or refresh:
			cols = self.data.columns.tolist()
			self.filterdata = self.data.sort(cols[:-1])
			self.filterdata = self.filterdata.dropna(how='any')		
		return self.filterdata
		
	def getnDimFromData(self):
		dims = np.array(self.getdimFromData())
		nDim = len(dims[dims > 1])
		return nDim
	
	def getdimFromData(self):
		dims = np.array([],dtype='int')
		filterdata = self.sortdata()
		#first determine the columns belong to the axes (not measure) coordinates
		cols = [i for i in self.header if (i['type'] == 'coordinate')]
		
		for i in cols:
			col = getattr(filterdata,i['name'])
			dims = np.hstack( ( dims ,len(col.unique())  ) )
		return dims
	
	def loadFile(self,file):

		self.header,self.skiprows = self.parseheader(file)
		data = pd.read_csv(file, sep='\t', comment='#',skiprows=self.skiprows,names=[i['name'] for i in self.header])
		data.name = file
		return data
	
	def toggleFiddle(self):
		from IPython.core import display
		if mpl.get_backend() == 'Qt4Agg':
			display.display(self.fig)
		
		if (mpl.get_backend() == 'Qt4Agg'):
			self.fiddle = Fiddle(self.fig)
			axFiddle = plt.axes([0.1, 0.85, 0.15, 0.075])


			self.bnext = Button(axFiddle, 'Fiddle')
			self.bnext.on_clicked(self.fiddle.connect)

			#attach to the relevant figure to make sure the object does not go out of scope
			self.fig.fiddle = self.fiddle
			self.fig.bnext = self.bnext
	
	def toggleControls(self,state=None):
		self.bControls = not self.bControls
		if state == None:
			toggleFiddle()
		
	def parseheader(self,file):
		skipindex = 0
		with open(file) as f:	
			for i, line in enumerate(f):
				if i < 3:
					continue
				if i > 5:
					if line[0] != '#': #find the skiprows accounting for the first linebreak in the header
						skipindex = i
						break
				if i > 300:
					break
		#nog een keer dunnetjes overdoen met read()
		f = open(file)
		alltext= f.read(skipindex)		
		with open(file) as myfile:
			alltext = [next(myfile) for x in xrange(skipindex)]
		alltext= ''.join(alltext)
	
		#doregex
		coord_expression = re.compile(r"""
									^\#\s*Column\s(.*?)\:
									[\r\n]{0,2}
									\#\s*end\:\s(.*?)
									[\r\n]{0,2}
									\#\s*name\:\s(.*?)
									[\r\n]{0,2}
									\#\s*size\:\s(.*?)
									[\r\n]{0,2}
									\#\s*start\:\s(.*?)
									[\r\n]{0,2}
									\#\s*type\:\s(.*?)[\r\n]{0,2}$ 
									
									"""#annoying \r's...
									,re.VERBOSE |re.MULTILINE)
		
		val_expression = re.compile(r"""
									^\#\s*Column\s(.*?)\:
									[\r\n]{0,2}
									\#\s*name\:\s(.*?)
									[\r\n]{0,2}
									\#\s*type\:\s(.*?)[\r\n]{0,2}$
									"""
									,re.VERBOSE |re.MULTILINE)
		coord=  coord_expression.findall(alltext) 
		val  = val_expression.findall(alltext)
		coord = [ zip(('column','end','name','size','start','type'),x) for x in coord]
		coord = [dict(x) for x in coord]
		val = [ zip(('column','name','type'),x) for x in val]
		val = [dict(x) for x in val]
		
		header=coord+val
		self.skipindex = skipindex
		self.header = header
		
		return np.array(header),skipindex
	
	def exportToMtx(self):

		for j, i in enumerate(self.exportData):

			data = i
			print(j)
			m = self.exportDataMeta[j]

			sz = np.shape(data)
			#write
			try:
				fid = open('{:s}{:d}{:s}'.format(self.data.name, j, '.mtx'),'w+')
			except Exception as e:
				print('Couldnt create file: {:s}'.format(str(e)))
				return

			#example of first two lines
			#Units, Data Value at Z = 0.5 ,X, 0.000000e+000, 1.200000e+003,Y, 0.000000e+000, 7.000000e+002,Nothing, 0, 1
			#850 400 1 8
			str1 = 'Units, Name: {:s}, {:s}, {:f}, {:f}, {:s}, {:f}, {:f}, {:s}, {:f}, {:f}\n'.format(
				m['datasetname'],
				m['xname'],
				m['xlims'][0],
				m['xlims'][1],
				m['yname'],
				m['ylims'][0],
				m['ylims'][1],
				m['zname'],
				m['zlims'][0],
				m['zlims'][1]
				)
			floatsize = 8
			str2 = '{:d} {:d} {:d} {:d}\n'.format(m['xu'],m['yu'],1,floatsize)
			fid.write(str1)
			fid.write(str2)
			#reshaped = np.reshape(data,sz[0]*sz[1],1)
			data.tofile(fid)
			fid.close()