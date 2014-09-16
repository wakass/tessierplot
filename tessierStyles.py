import numpy as np
from scipy import signal

def helper_deinterlace(w):
	w['deinterXXodd'] = w['XX'][1::2,1:] #take every other column in a sweepback measurement, offset 1
	w['deinterXXeven'] = w['XX'][::2,:] #offset 0            
	#w.deinterXXodd  = w.deinterXXodd
	#w.deinterXXeven = w.deinterXXeven
  
def helper_mov_avg(w):  
	m, n = 1, 5     # The shape of the window array
	win = np.ones((m, n))
	#win = signal.kaiser(m,8.6,sym=False)
	w['XX'] = moving_average_2d(w['XX'], win)
	
def helper_didv(w):
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
   
def helper_normal(w):
	w['XX'] = w['XX']

def helper_abs(w):
	w['XX'] = np.abs(w['XX'])
	w['cbar_trans'] = ['abs'] + w['cbar_trans'] 
	w['cbar_quantity'] = w['cbar_quantity'] 
	w['cbar_unit'] = w['cbar_unit']


def helper_flipaxes(w):
	w['XX'] = np.transpose( w['XX'])
	w['ext'] = (w['ext'][2],w['ext'][3],w['ext'][0],w['ext'][1])
	
styles = {
	'deinterlace': helper_deinterlace,
	'didv': helper_didv,
	'log': helper_log,
	'normal': helper_normal,
	'flipaxes': helper_flipaxes,
	'mov_avg': helper_mov_avg,
	'abs': helper_abs
	
	}
	
def getEmptyWrap():
	w = {'ext':(0,0,0,0), 'ystep':1,'XX': [], 'cbar_quantity': '', 'cbar_unit': 'a.u.', 'cbar_trans':[]}
	return w

def processStyle(style,wrap):
	for s in style:
		try:
			#print s
			styles[s](wrap)
		except Exception, e:
			print 'Style %s does not exist (%s)' % (s,e)
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
