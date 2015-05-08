import numpy as np
from scipy import signal
import re
import collections
import matplotlib.colors as mplc

REGEX_STYLE_WITH_PARAMS = re.compile('(.+)\((.+)\)')
REGEX_VARVALPAIR = re.compile('(\w+)=(\w+)')

def nonzeromin(x):
	'''
	Get the smallest non-zero value from an array
	Also works recursively for multi-dimensional arrays
	Returns None if no non-zero values are found
	'''
	x = numpy.array(x)
	nzm = None
	if len(x.shape) > 1:
		for i in range(x.shape[0]):
			xnow = nonzeromin(x[i])
			if xnow != 0 and xnow is not None and (nzm is None or nzm > xnow):
				nzm = xnow
	else:
		for i in range(x.shape[0]):
			if x[i] != 0 and (nzm is None or nzm > x[i]):
				nzm = x[i]
	return nzm

def helper_deinterlace(w):
	w['deinterXXodd'] = w['XX'][1::2,1:] #take every other column in a sweepback measurement, offset 1
	w['deinterXXeven'] = w['XX'][::2,:] #offset 0
	#w.deinterXXodd  = w.deinterXXodd
	#w.deinterXXeven = w.deinterXXeven

def helper_mov_avg(w):
	(m, n) = (int(w['mov_avg_m']), int(w['mov_avg_n']))     # The shape of the window array
	win = np.ones((m, n))
	#win = signal.kaiser(m,8.6,sym=False)
	w['XX'] = moving_average_2d(w['XX'], win)

def helper_savgol(w):
	'''Perform Savitzky-Golay smoothing'''
	w['XX'] = signal.savgol_filter(
			w['XX'], int(w['savgol_samples']), int(w['savgol_order']))

def helper_didv(w):
	a=(w['XX'])

# 	if len(a.shape) == a.ndim: #if 1D 
# 		print '1d'
# 		w['XX'] = np.diff(w['XX'])
# 		print np.shape(w['XX'])
# 		w['XX'] = np.append(w['XX'],w['XX'][-1])
# 
# 		print np.shape(w['XX'])
# 	else:
	w['XX'] = np.diff(w['XX'],axis=1)
	#1 nA / 1 mV = 0.0129064037 conductance quanta
	w['XX'] = w['XX'] / w['ystep'] * 0.0129064037
	w['cbar_quantity'] = 'dI/dV'
	w['cbar_unit'] = '$\mu$Siemens'
	w['cbar_unit'] = r'$\mathrm{e}^2/\mathrm{h}$'

def helper_log(w):
	w['XX'] = np.log10(np.abs(w['XX']))
	w['cbar_trans'] = ['log$_{10}$','abs'] + w['cbar_trans']
	w['cbar_quantity'] = w['cbar_quantity']
	w['cbar_unit'] = w['cbar_unit']

def helper_logy(w):
	w['XX'] = np.log10(np.abs(w['XX']))
	w['cbar_trans'] = ['log$_{10}$','abs'] + w['cbar_trans']
	w['cbar_quantity'] = w['cbar_quantity']
	w['cbar_unit'] = w['cbar_unit']

def helper_fancylog(w):
	'''
	Use a logarithmic normalising function for plotting.
	This might be incompatible with Fiddle.
	'''
	w['XX'] = abs(w['XX'])
	(cmin, cmax) = (w['fancylog_cmin'], w['fancylog_cmax'])
	if type(cmin) is str:
		cmin = float(cmin)
	if type(cmax) is str:
		cmax = float(cmax)
	if cmin is None:
		cmin = w['XX'].min()
		if cmin == 0:
			cmin = 0.1 * nonzeromin(w['XX'])
	if cmax is None:
		cmax = w['XX'].max()
	w['imshow_norm'] = mplc.LogNorm(vmin=cmin, vmax=cmax)

def helper_normal(w):
	w['XX'] = w['XX']
	
def helper_minsubtract(w):
	w['XX'] = w['XX'] - np.min(w['XX'])


def helper_abs(w):
	w['XX'] = np.abs(w['XX'])
	w['cbar_trans'] = ['abs'] + w['cbar_trans']
	w['cbar_quantity'] = w['cbar_quantity']
	w['cbar_unit'] = w['cbar_unit']

def helper_flipaxes(w):
	w['XX'] = np.transpose( w['XX'])
	w['ext'] = (w['ext'][2],w['ext'][3],w['ext'][0],w['ext'][1])

STYLE_FUNCS = {
	'deinterlace': helper_deinterlace,
	'didv': helper_didv,
	'log': helper_log,
	'normal': helper_normal,
	'flipaxes': helper_flipaxes,
	'mov_avg': helper_mov_avg,
	'abs': helper_abs,
	'savgol': helper_savgol,
	'fancylog': helper_fancylog,
	'minsubtract': helper_minsubtract
}

'''
Specification of styles with arguments
Format:
	{'<style_name>': {'<param_name>': <default_value>, 'param_order': order}}
Multiple styles can be specified, multiple parameters (name-defaultvalue pairs)
can be specified, and param_order decides the order in which they can be given
as non-keyword arguments.
'''
STYLE_SPECS = {
	'deinterlace': {'param_order': []},
	'didv': {'param_order': []},
	'log': {'param_order': []},
	'normal': {'param_order': []},
	'flipaxes': {'param_order': []},
	'mov_avg': {'m': 1, 'n': 5, 'win': None, 'param_order': ['m', 'n', 'win']},
	'abs': {'param_order': []},
	'savgol': {'samples': 3, 'order': 1, 'param_order': ['samples', 'order']},
	'fancylog': {'cmin': None, 'cmax': None, 'param_order': ['cmin', 'cmax']},
	'minsubtract': {'param_order': []}
}

#Backward compatibility
styles = STYLE_FUNCS

def getEmptyWrap():
	'''Get empty wrap with default parameter values'''
	w = {'ext':(0,0,0,0), 'ystep':1, 'XX': [], 'cbar_quantity': '', 'cbar_unit': 'a.u.', 'cbar_trans': [], 'imshow_norm': None}
	return w

def getPopulatedWrap(style=[]):
	'''
	Get wrap with populated values specified in the style list
	For example, if styles is:
		['deinterlace', 'mov_avg(n=5, 1)']
	This will add the following to the wrap:
		{'mov_avg_n': '5', 'mov_avg_m': '1'}
	'''
	w = getEmptyWrap()
	if style is None:
		return w
	elif type(style) is not list:
		style = list([style])
	for s in style:
		try:
			# Parse, process keyword arguments and collect non-kw arguments
			sregex = re.match(REGEX_STYLE_WITH_PARAMS, s)
			spar = []
			if sregex is not None:
				(s, sparamstr) = sregex.group(1,2)
				sparams = (
						sparamstr.replace(';',',').replace(':','=')
						.split(','))
				for i in range(len(sparams)):
					sparam = sparams[i].strip()
					spregex = re.match(REGEX_VARVALPAIR, sparam)
					if spregex is None:
						spar.append(sparam)
					else:
						w['{:s}_{:s}'.format(s, spregex.group(1))] = spregex.group(2)
			# Process non-keyword arguments and default values
			(i, j) = (0, 0)
			pnames = STYLE_SPECS[s]['param_order']
			while i < len(pnames):
				key = '{:s}_{:s}'.format(s, pnames[i])
				if key not in w:
					if j < len(spar):
						w[key] = spar[j]
						j += 1
					else:
						w[key] = STYLE_SPECS[s][pnames[i]]
				i += 1
		except Exception as e:
			print('getPolulatedWrap(): Style {:s} does not exist ({:s})'.format(s, str(e)))
			pass
	return w

def processStyle(style, wrap):
	for s in style:
		try:
# 			print(s)
			func = STYLE_FUNCS[s.split('(')[0]]
			func(wrap)
		except Exception as e:
			print('processStyle(): Style {:s} does not exist ({:s})'.format(s, str(e)))
			pass


def moving_average_2d(data, window):
    """Moving average on two-dimensional data.
    """
    # Makes sure that the window function is normalized.
    window /= window.sum()
    # Makes sure data array is a numpy array or masked array.
    if type(data).__name__ not in ['ndarray', 'MaskedArray']:
        data = np.asarray(data)

    # The output array has the same dimensions as the input data
    # (mode='same') and symmetrical boundary conditions are assumed
    # (boundary='symm').
    return signal.convolve2d(data, window, mode='same', boundary='symm')
