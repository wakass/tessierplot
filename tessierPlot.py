#tessier.py
#tools for plotting all kinds of files, with fiddle control etc

# data = loadFile(...)
# a = plot3DSlices(data,)

import matplotlib.pyplot as plt
import matplotlib as mpl


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.widgets import Button,Line2D
from scipy.signal import argrelmax

import pandas as pd
import numpy as np
import math
import tessierView as tv


import re
import os
_moduledir = os.path.dirname(os.path.realpath(__file__))

from tessierGuiElements import *
import tessierStyles as tstyle
from imp import reload #DEBUG
reload(tstyle) #DEBUG


##Only load this part on first import, calling this on reload has dire consequences
## Note: there is still a bug where closing a previously plotted window and plotting another plot causes the window and the kernel to hang
try:
	magichappened
except:
	import IPython  
	ipy=IPython.get_ipython()
	ipy.magic("pylab qt")
	magichappened = False
else:
	magichappened = True
from mpl_toolkits.axes_grid1 import make_axes_locatable



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
	
	
def starplot(file,fig=None,style=None,n_index=None):
	names,skiprows=parseheader(file)
	dat = loadFile(file,names,skiprows)
	# dat=dat.dropna(how='any') #not doing this somehow prevent a line drawn from the beginning to the end of the line?

	if fig == None:
		fig = plt.figure()
	else:
		fig = plt.figure(fig.number)

	#basically project the first and final dimension
	cnt= 0
	for i in range(0,len(dat.keys())-1):
		n_subplots = len(dat.keys())+1
		if n_index != None:
				n_subplots = len(n_index)
				print n_subplots
				if i not in n_index:
					continue
		title = parseUnitAndNameFromColumnName(dat.keys()[i])
		cnt=cnt+1
		ax = plt.subplot(n_subplots,1,cnt)
		
		y= (dat[dat.keys()[-1]])
		wrap = tstyle.getPopulatedWrap(style)
		wrap['XX'] = y
		tstyle.processStyle(style,wrap)
		
		plt.plot(dat[dat.keys()[i]],wrap['XX'],label=title)
		plt.legend(loc=9, bbox_to_anchor=(-0.5, 0.7), ncol=3)
	return fig
	

def loadCustomColormap(file='./cube1.txt'):
	do = np.loadtxt(file)
	ccmap = mpl.colors.LinearSegmentedColormap.from_list('name',do)

	ccmap.set_under(do[0])
	ccmap.set_over(do[-1])
	return ccmap



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
				continue
			yield xs[0] == i
	else:
		#empty list
		yield slice(None) #return a 'semicolon' to select all the values when there's no value to filter on


class plotR:
	def __init__(self,file):
		self.header = None
		self.names = None
		self.fig = None
	
		self.exportData =[]	
		self.exportDataMeta = []	
		self.file = file

		self.data  = self.loadFile(file) 
		self.name  = file
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
		print uniques_col_str
		if len(coords[filter_neg]) > 1: 
			
			self.plot3d(uniques_col_str=uniques_col_str,**kwargs)
			self.exportToMtx() #do this
		else:
			self.plot2d(**kwargs)		

	def autoColorScale(self,data):
		values, edges = np.histogram(data, 256)
		maxima = edges[argrelmax(values,order=24)]
		if maxima.size>0: 
			cminlim , cmaxlim = maxima[0] , np.max(data)
		else:
			cminlim , cmaxlim = np.min(data) , np.max(data)
		return (cminlim,cmaxlim)

		
	def plot3d(self,fiddle=True,massage_func=None,uniques_col_str=[],n_index=None,style='log',clim='auto',aspect='auto',interpolation='none',**kwargs): #previously plot3dslices
		if not self.fig:
			self.fig = plt.figure()
		
		if self.data is None:
			self.loadFile(file)
			
		cols = self.data.columns.tolist()
		filterdata = self.sortdata()
		
		uniques_col = []
		self.uniques_per_col=[]
		self.data = self.data.dropna(how='any')
		sweepdirection = self.data[cols[-2]].iloc[0] > self.data[cols[-2]].iloc[1] #True is sweep neg to pos

				

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
			sluicy = filterdata.loc[ind]
			slicy = sluicy
			#magic follows:
			#get the last columns, and exclude uniques_cols
			us=uniques_col_str
			if len(uniques_col_str) > 0:
				b=[ slicy.columns!=u for u in us]
				c = reduce(lambda x,y: np.logical_and(x,y), b)
				slicy = slicy[slicy.columns[c]]
					
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
			self.extent = ext
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
					'datasetname':self.name}
				self.exportDataMeta = np.append(self.exportDataMeta,m)
			except Exception,e:
				print e
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
			w['massage_func']=massage_func
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
# 			self.XX = XX
			if 'deinterlace' in style:
				self.fig = plt.figure()
				ax_deinter_odd  = plt.subplot(2, 1, 1)
				w['deinterXXodd'] = np.rot90(w['deinterXXodd'])
				ax_deinter_odd.imshow(w['deinterXXodd'],extent=ext, cmap=plt.get_cmap(self.ccmap),aspect=aspect,interpolation=interpolation)
				self.deinterXXodd_data = w['deinterXXodd']
				
				ax_deinter_even = plt.subplot(2, 1, 2)
				w['deinterXXeven'] = np.rot90(w['deinterXXeven'])
				ax_deinter_even.imshow(w['deinterXXeven'],extent=ext, cmap=plt.get_cmap(self.ccmap),aspect=aspect,interpolation=interpolation)
				self.deinterXXeven_data = w['deinterXXeven']

			else:
# 				norm = mpl.colors.PowerNorm(gamma=.2)
# 				norm.autoscale(self.XX)
				self.im = ax.imshow(XX,extent=ext, cmap=plt.get_cmap(self.ccmap),aspect=aspect,interpolation=interpolation, norm=w['imshow_norm'])

				if  str(clim).lower() == 'auto': ## really not a good way i suppose, clim should be a tuple
					self.im.set_clim(self.autoColorScale(XX.flatten()))
				elif clim != (0,0):
					self.im.set_clim(clim)


			if 'flipaxes' in style:
				ax.set_xlabel(slicy.columns[-2])
				ax.set_ylabel(slicy.columns[-3])
			else:
				ax.set_xlabel(slicy.columns[-3])
				ax.set_ylabel(slicy.columns[-2])


			title = ''
			for i in uniques_col_str:
				title = '\n'.join([title, '{:s}: {:g} (mV)'.format(i,getattr(sluicy,i).iloc[0])])
			print(title)
			if 'notitle' not in style:
				ax.set_title(title)
			# create an axes on the right side of ax. The width of cax will be 5%
			# of ax and the padding between cax and ax will be fixed at 0.05 inch.
			divider = make_axes_locatable(ax)
			cax = divider.append_axes("right", size="5%", pad=0.05)

			pos = list(ax.get_position().bounds)
			if hasattr(self, 'im'):
				self.cbar = plt.colorbar(self.im, cax=cax)
				cbar = self.cbar
	
				cbar.set_label(cbar_title)

			cnt+=1 #counter for subplots

		if fiddle: self.toggleFiddle()
		self.toggleLinedraw()
		return self.fig
	
	def plot2d(self,fiddle=False,n_index=None,style=['normal'],**kwargs): #previously scanplot
		# scanplot(file,fig=None,n_index=None,style=[],data=None,**kwargs):
		# kwargs go into matplotlib/pyplot plot command
		# fiddle for compatibility with tessierView only
		
		if not self.fig:
			self.fig = plt.figure()
		
		if self.data is None:
			self.loadFile(file)
	
		uniques_col = []
	
		#assume 2d plots with data in the two last columns
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
	def guessStyle(self):
		style=[]
		#autodeinterlace function
		#	if y[yu-1]==y[yu]: style.append('deinterlace0')

		#autodidv function
		y=self.filterdata.iloc[:,-2]
 		if (max(y) == -1*min(y) and max(y) <= 150):
			style.insert(0,'sgdidv')
 			style.insert(1,'log')
		return style
		
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
		self.data = pd.read_csv(file, sep='\t', comment='#',skiprows=self.skiprows,names=[i['name'] for i in self.header])
		self.data.name = file
		return self.data
	def toggleLinedraw(self):
		self.linedraw=Linedraw(self.fig)

		self.fig.drawbutton = toggleButton('draw', self.linedraw.connect)
		topwidget = self.fig.canvas.topLevelWidget()
		toolbar = topwidget.children()[2]
		action = toolbar.addWidget(self.fig.drawbutton)
		
		self.fig.linedraw = self.linedraw

		
	def toggleFiddle(self):
		from IPython.core import display
		
		if (mpl.get_backend() == 'Qt4Agg' or 'nbAgg'):
			self.fiddle = Fiddle(self.fig)

			self.fig.fiddlebutton = toggleButton('fiddle', self.fiddle.connect)
			topwidget = self.fig.canvas.topLevelWidget()
			toolbar = topwidget.children()[2]
			action = toolbar.addWidget(self.fig.fiddlebutton)

			#attach to the relevant figure to make sure the object does not go out of scope
			self.fig.fiddle = self.fiddle

	
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
				fid = open('{:s}{:d}{:s}'.format(self.name, j, '.mtx'),'w+')
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

