#tessier.py
#tools for plotting all kinds of files, with fiddle control etc

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.signal import argrelmax

import numpy as np
import math
import re

#all tessier related imports
from gui import *
import styles
from data import Data
import helpers


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



def parseUnitAndNameFromColumnName(input):
	reg = re.compile(r'\{(.*?)\}')
	z = reg.findall(input)
	return z


def loadCustomColormap(file=helpers.get_asset('cube1.txt')):
	do = np.loadtxt(file)
	ccmap = mpl.colors.LinearSegmentedColormap.from_list('name',do)

	ccmap.set_under(do[0])
	ccmap.set_over(do[-1])
	return ccmap


class plotR(object):
	def __init__(self,file):
		self.fig = None
		self.file = file

		self.data  = Data.from_file(filepath=file)
		self.name  = file
		self.exportData = []
		self.exportDataMeta = []
		self.bControls = True #boolean controlling state of plot manipulation buttons


	def is2d(self,**kwargs):
		nDim = self.data.ndim_sparse
		#if the uniques of a dimension is less than x, plot in sequential 2d, otherwise 3d

		#maybe put logic here to plot some uniques as well from nonsequential axes?
		filter = self.data.dims < 5
		filter_neg = np.array([not x for x in filter])

		coords = np.array(self.data.coordkeys)

		return len(coords[filter_neg]) < 2

	def quickplot(self,**kwargs):
		nDim = self.data.ndim_sparse

		filter = self.data.dims < 5

		coords = np.array(self.data.coordkeys)

		uniques_col_str = coords[filter]
		print uniques_col_str

		if self.is2d():
			fig = self.plot2d(**kwargs)
		else:
			fig = self.plot3d(uniques_col_str=uniques_col_str,**kwargs)
			self.exportToMtx()

		return fig

	def autoColorScale(self,data):
		#filter out NaNs or infinities, should any have crept in
		data = data[np.isfinite(data)]
		values, edges = np.histogram(data, 256)
		maxima = edges[argrelmax(values,order=24)]
		if maxima.size>0:
			cminlim , cmaxlim = maxima[0] , np.max(data)
		else:
			cminlim , cmaxlim = np.min(data) , np.max(data)
		return (cminlim,cmaxlim)


	def plot3d(self,    massage_func=None,
						uniques_col_str=[],
						drawCbar=True,
						subplots_args={'top':0.96, 'bottom':0.17, 'left':0.14, 'right':0.85,'hspace':0.0},
						ax_destination=None,
						n_index=None,
						style='log',
						clim=None,
						aspect='auto',
						interpolation='none',
						value_axis=-1,
						sweepoverride=False,
						cbar_orientation='vertical',
						cbar_location ='normal',
						**kwargs):
		#some housekeeping
		if not self.fig and not ax_destination:
			self.fig = plt.figure()
			self.fig.subplots_adjust(**subplots_args)
			
		self.ccmap = loadCustomColormap()

		#determine how many subplots we need
		n_subplots = 1

		#make a list of uniques per column associated with column name
		uniques_by_column = dict(zip(self.data.coordkeys + self.data.valuekeys, self.data.dims))

		#by this array
		for i in uniques_col_str:
			n_subplots *= uniques_by_column[i]

		if n_index is not None:
			n_index = np.array(n_index)
			n_subplots = len(n_index)

		if n_subplots > 1:
			width = 2
		else:
			width = 1
		n_valueaxes = len(self.data.valuekeys)
		width = n_valueaxes
		n_subplots = n_subplots * n_valueaxes
		gs = gridspec.GridSpec(int(n_subplots/width)+n_subplots%width, width)

		cnt=0 #subplot counter

		#enumerate over the generated list of unique values specified in the uniques columns
		for j,ind in enumerate(self.data.make_filter_from_uniques_in_columns(uniques_col_str)):
			#each value axis needs a plot
			for value_axis in range(n_valueaxes):

				#plot only if number of the plot is indicated
				if n_index is not None:
					if j not in n_index:
						continue
				data_byuniques = self.data.sorted_data.loc[ind]
				data_slice = data_byuniques

				#get the columns /not/ corresponding to uniques_cols
				#find the coord_keys in the header
				coord_keys = self.data.coordkeys

				#filter out the keys corresponding to unique value columns
				us=uniques_col_str
				coord_keys = [key for key in coord_keys if key not in uniques_col_str ]
				#now find out if there are multiple value axes
				value_keys = self.data.valuekeys


				x=data_slice.loc[:,coord_keys[-2]]
				y=data_slice.loc[:,coord_keys[-1]]
				z=data_slice.loc[:,value_keys[value_axis]]

				xu = np.size(x.unique())
				yu = np.size(y.unique())


				## if the measurement is not complete this will probably fail so trim off the final sweep?
				print('xu: {:d}, yu: {:d}, lenz: {:d}'.format(xu,yu,len(z)))

				if xu*yu != len(z):
					xu = (len(z) / yu) #dividing integers so should automatically floor the value

				#trim the first part of the sweep, for different min max, better to trim last part?
				#or the first since there has been sorting
				#this doesnt work for e.g. a hilbert measurement

				print('xu: {:d}, yu: {:d}, lenz: {:d}'.format(xu,yu,len(z)))

				#sorting sorts negative to positive, so beware:
				#sweep direction determines which part of array should be cut off
				if sweepoverride: ##if sweep True, override the detect value
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
						'xname':coord_keys[-2],
						'yname':coord_keys[-1],
						'zname':'unused',
						'datasetname':self.name}
					self.exportDataMeta = np.append(self.exportDataMeta,m)
				except Exception,e:
					print e
					pass
				if ax_destination is None:
					ax = plt.subplot(gs[cnt])
				else:
					ax = ax_destination
				cbar_title = ''



				if type(style) != list:
					style = list([style])


				measAxisDesignation = parseUnitAndNameFromColumnName(value_keys[value_axis])
				#wrap all needed arguments in a datastructure
				cbar_quantity = measAxisDesignation[0]
				cbar_unit = measAxisDesignation[1]
				cbar_trans = [] #trascendental tracer :P For keeping track of logs and stuff

				w = styles.getPopulatedWrap(style)
				w2 = {'ext':ext, 'ystep':ystep,'XX': XX, 'cbar_quantity': cbar_quantity, 'cbar_unit': cbar_unit, 'cbar_trans':cbar_trans}
				for k in w2:
					w[k] = w2[k]
				w['massage_func']=massage_func
				styles.processStyle(style, w)

				#unwrap
				ext = w['ext']
				XX = w['XX']
				cbar_trans_formatted = ''.join([''.join(s+'(') for s in w['cbar_trans']])
				cbar_title = cbar_trans_formatted + w['cbar_quantity'] + ' (' + w['cbar_unit'] + ')'
				if len(w['cbar_trans']) is not 0:
					cbar_title = cbar_title + ')'

				XX = np.rot90(XX)

				if 'deinterlace' in style:
					self.fig = plt.figure()
					ax_deinter_odd  = plt.subplot(2, 1, 1)
					xx_odd = np.rot90(w['deinterXXodd'])
					ax_deinter_odd.imshow(xx_odd,extent=ext, cmap=plt.get_cmap(self.ccmap),aspect=aspect,interpolation=interpolation)
					self.deinterXXodd_data = xx_odd

					ax_deinter_even = plt.subplot(2, 1, 2)
					xx_even = np.rot90(w['deinterXXeven'])
					ax_deinter_even.imshow(xx_even,extent=ext, cmap=plt.get_cmap(self.ccmap),aspect=aspect,interpolation=interpolation)
					self.deinterXXeven_data = xx_even
				else:
					self.im = ax.imshow(XX,extent=ext, cmap=plt.get_cmap(self.ccmap),aspect=aspect,interpolation=interpolation, norm=w['imshow_norm'],clim=clim)
					if not clim:
						self.im.set_clim(self.autoColorScale(XX.flatten()))

				if 'flipaxes' in style:
					ax.set_xlabel(coord_keys[-1])
					ax.set_ylabel(coord_keys[-2])
				else:
					ax.set_xlabel(coord_keys[-2])
					ax.set_ylabel(coord_keys[-1])


				title = ''
				for i in uniques_col_str:
					title = '\n'.join([title, '{:s}: {:g} (mV)'.format(i,getattr(data_byuniques,i).iloc[0])])
				print(title)
				if 'notitle' not in style:
					ax.set_title(title)
				# create an axes on the right side of ax. The width of cax will be 5%
				# of ax and the padding between cax and ax will be fixed at 0.05 inch.
				if drawCbar:
					from mpl_toolkits.axes_grid.inset_locator import inset_axes
					if cbar_location == 'inset':
						if cbar_orientation == 'horizontal':
							cax = inset_axes(ax,width='30%',height='10%',loc=2,borderpad=1)
						else:
							cax = inset_axes(ax,width='30%',height='10%',loc=1)
					else:
						divider = make_axes_locatable(ax)
						if cbar_orientation == 'horizontal':
							cax = divider.append_axes("top", size="10%", pad=0.05)
						else:
							cax = divider.append_axes("right", size="2.5%", pad=0.05)
						pos = list(ax.get_position().bounds)
					if hasattr(self, 'im'):
						self.cbar = plt.colorbar(self.im, cax=cax,orientation=cbar_orientation)
						cbar = self.cbar

						if cbar_orientation == 'horizontal':
							cbar.set_label(cbar_title,labelpad=-20, x=1.35)
	# 						cbar.ax.xaxis.set_label_position('top')
	# 						cbar.ax.yaxis.set_label_position('left')
						else:
							cbar.set_label(cbar_title)#,labelpad=-19, x=1.32)
				cnt+=1 #counter for subplots
		if self.fig:
			self.toggleFiddle()
			self.toggleLinedraw()
			self.toggleLinecut()
		return self.fig

	def plot2d(self,fiddle=False,
					n_index=None,
					value_axis = -1,
					style=['normal'],
					uniques_col_str=None,
					legend=True,
					ax_destination=None,
					subplots_args={'top':0.96, 'bottom':0.17, 'left':0.14, 'right':0.85,'hspace':0.0},
					massage_func=None,
					**kwargs):
					
		if not self.fig and not ax_destination:
			self.fig = plt.figure()
			self.fig.subplots_adjust(**subplots_args)

		n_valueaxes = len(self.data.valuekeys)
		width = n_valueaxes
		n_subplots = 1 #keep at 1 for now
		n_subplots = n_subplots * n_valueaxes
		gs = gridspec.GridSpec(int(n_subplots/width)+n_subplots%width, width)


		coord_keys = self.data.coordkeys
		value_keys = self.data.valuekeys
		
		#assume 2d plots with data in the two last columns
		if uniques_col_str is None:
			uniques_col_str = coord_keys[:-1]

		uniques_axis_designations = []
		#do some filtering of the colstr to get seperate name and unit of said name
		for a in uniques_col_str:
			uniques_axis_designations.append(parseUnitAndNameFromColumnName(a))

		uniques_col = self.data[uniques_col_str]
		if n_index is not None:
			n_index = np.array(n_index)
			n_subplots = len(n_index)

		ax = None
		for i,j in enumerate(self.data.make_filter_from_uniques_in_columns(uniques_col_str)):
			if n_index is not None:
					if i not in n_index:
						continue
			data = self.data.sorted_data[j]
			#get the columns /not/ corresponding to uniques_cols
			#find the coord_keys in the header
			coord_keys = self.data.coordkeys

			#filter out the keys corresponding to unique value columns
			us=uniques_col_str
			coord_keys = [key for key in coord_keys if key not in uniques_col_str ]
			#now find out if there are multiple value axes
			value_keys = self.data.valuekeys

			x=data.loc[:,coord_keys[-1]]
			y=data.loc[:,value_keys[value_axis]]

						
			title =''
			for i,z in enumerate(uniques_col_str):
				title = '\n'.join([title, '{:s}: {:g}'.format(uniques_axis_designations[i],data[z].iloc[0])])

			wrap = styles.getPopulatedWrap(style)
			wrap['XX'] = y
			wrap['X']  = x
			wrap['massage_func'] = massage_func
			styles.processStyle(style,wrap)
			if ax_destination:
				ax = ax_destination
			else:
				cnt =0
				ax = plt.subplot(gs[cnt])
			ax.plot(wrap['X'],wrap['XX'],label=title,**kwargs)

		if legend:
			plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
				   ncol=2, mode="expand", borderaxespad=0.)
# 		ax = self.fig.axes[0]
		xaxislabel = parseUnitAndNameFromColumnName(coord_keys[-1])
		yaxislabel = parseUnitAndNameFromColumnName(value_keys[value_axis])
		if ax:
			ax.set_xlabel(xaxislabel[0]+'(' + xaxislabel[1] +')')
			ax.set_ylabel(yaxislabel[0]+'(' + yaxislabel[1] +')')
		return self.fig


	def starplot(self,style=[]):
		if not self.fig:
			self.fig = plt.figure()

		data=self.data
		coordkeys = self.data.coordkeys
		valuekeys = self.data.valuekeys

		coordkeys_notempty=[k for k in coordkeys if len(data[k].unique()) > 1]
		n_subplots = len(coordkeys_notempty)
		width = 2
		import matplotlib.gridspec as gridspec
		gs = gridspec.GridSpec(int(n_subplots/width)+n_subplots%width, width)


		for n,k in enumerate(coordkeys_notempty):
			ax = plt.subplot(gs[n])
			for v in valuekeys:
				y= data[v]

				wrap = styles.getPopulatedWrap(style)
				wrap['XX'] = y
				styles.processStyle(style,wrap)

				ax.plot(data[k], wrap['XX'])
			ax.set_title(k)
		return self.fig

	def guessStyle(self):
		style=[]
		#autodeinterlace function
		#	if y[yu-1]==y[yu]: style.append('deinterlace0')

		#autodidv function
		y=self.data.sorted_data.iloc[:,-2]
 		if (max(y) == -1*min(y) and max(y) <= 150):
 			style.extend(['mov_avg(m=1,n=10)','didv','mov_avg(m=1,n=5)','abs'])

		#default style is 'log'
		style.append('log')

		return style



	def toggleLinedraw(self):
		self.linedraw=Linedraw(self.fig)

		self.fig.drawbutton = toggleButton('draw', self.linedraw.connect)
		topwidget = self.fig.canvas.topLevelWidget()
		toolbar = topwidget.children()[2]
		action = toolbar.addWidget(self.fig.drawbutton)

		self.fig.linedraw = self.linedraw
	def toggleLinecut(self):
		self.linecut=Linecut(self.fig,self)

		self.fig.cutbutton = toggleButton('cut', self.linecut.connect)
		topwidget = self.fig.canvas.topLevelWidget()
		toolbar = topwidget.children()[2]
		action = toolbar.addWidget(self.fig.cutbutton)

		self.fig.linecut = self.linecut

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



	def exportToMtx(self):

		for j, i in enumerate(self.exportData):

			data = i
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
