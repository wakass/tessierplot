#tessierMagic, where the real % happens
#tessierView for couch potato convenience...

import tessierPlot as ts
import jinja2 as jj
import matplotlib.pyplot as plt
import pylab
import os
from itertools import chain
import numpy as np
import re
from IPython.display import VimeoVideo
from IPython.display import display, HTML

reload(ts)

_plot_width = 4. # in inch (ffing inches eh)
_plot_height = 3. # in inch

_fontsize_plot_title = 10
_fontsize_axis_labels = 10
_fontsize_axis_tick_labels = 10

plotstyle = 'normal'

pylab.rcParams['figure.figsize'] = (_plot_width, _plot_height) #(width in inch, height in inch)
pylab.rcParams['axes.labelsize'] = _fontsize_axis_labels
pylab.rcParams['xtick.labelsize'] = _fontsize_axis_tick_labels
pylab.rcParams['ytick.labelsize'] = _fontsize_axis_tick_labels

class tessierView:
    def __init__(self, rootdir='/Users/waka/phd/pynotes/data/dipstick/data/20150430/', filemask='.*\.dat$',filterstring=''):
        self._root = rootdir
        self._filemask = filemask
        self._filterstring = filterstring
        self._allthumbs = []
    def on(self):   
        print 'You are now watching through the glasses of ideology'
        display(VimeoVideo('106036638'))
        
    def getthumbcachepath(self,file):
        oneupdir = os.path.abspath(os.path.join(os.path.dirname(file),os.pardir))
        datedir = os.path.split(oneupdir)[1] #directory name should be datedir, if not 
        if re.match('[0-9]{8}',datedir):
            preid= datedir
        else:
            preid = ''
        
        #relative to project/working directory
        cachepath = os.path.join(os.getcwd(),'./thumbnails', preid + '_'+os.path.splitext(os.path.split(file)[1])[0] + '_thumb.png')
        return cachepath
        
    def getthumbdatapath(self,file):
        thumbdatapath = os.path.splitext(file)[0] + '_thumb.png'
        return thumbdatapath
    
    def getsetfilepath(self,file):
        return (os.path.splitext(file)[0] + '.set')
    
    def makethumbnail(self,file,override=False):
        #create a thumbnail and store it in the same directory and in the thumbnails dir for local file serving, override options for if file already exists
        thumbfile = self.getthumbcachepath(file)
        thumbfile_datadir =  self.getthumbdatapath(file)
        if (os.path.exists(thumbfile) and override) or (not os.path.exists(thumbfile)):
            try:
                p = ts.plotR(file)
                p.quickplot() #make quickplot more intelligent so it detect dimensionality from uniques
                p.fig.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.85,hspace=0.0)
                p.fig.savefig(thumbfile)
                p.fig.savefig(thumbfile_datadir)
                plt.close(p.fig)
            except Exception,e:
                thumbfile = None #if fail no thumbfile was created
                print 'Error {:s}'.format(e)
                pass
        #do nothing if thumb exists
        
        
        return thumbfile
    
    
    def walk(self,filemask,filterstring,**kwargs):
        paths = (self._root,)
        images = 0
        self.allthumbs = []
    
        reg = re.compile(self._filemask) #get only files determined by filemask
    
        for dirname,dirnames,filenames in chain.from_iterable(os.walk(path) for path in paths):
            for filename in filenames:
                fullpath = os.path.join(dirname,filename)
                res = reg.findall(filename)
                if res: #found a file that matches the filemask
                    if filterstring in open(self.getsetfilepath(fullpath)).read():   #liable for improvement
                    #check for certain parameters with filterstring in the set file: e.g. 'dac4: 1337.0'
                        thumbpath = self.makethumbnail(fullpath)
                        if thumbpath:
                            self._allthumbs.append({'datapath':fullpath,'thumbpath':thumbpath})
                            images += 1
                            
        return self._allthumbs
    
    def htmlout(self,refresh=False):
        if refresh:
            self.walk(self._filemask,'dac')
        
        #unobfuscate the file relative to the working directory
        #since files are served from ipyhton notebook from ./files/
        all_relative = [{ 'thumbpath':'./files/'+os.path.relpath(k['thumbpath'],start=os.getcwd()),'datapath':k['datapath'] } for k in self._allthumbs]
    
        # print all_relative
        out=u"""
        <div>
    {% for item in items %}

    {% if loop.index % 3 == 1 %}
        <div class='row'>
    {% endif %}

        <div class='col'>
            <div class='thumb'>
            <img src="{{ item.thumbpath }}"/> 
            </div>
            <div class='controls'>
            <button id='{{ item.datapath }}' onClick='plot(this.id,"normal","dsf")'>Normal</button>
            </div>
        </div>
    {% if loop.index % 3 == 0 %}
        </div>
    {% endif %}
    {% endfor %}    
    </div>
        <script type="text/Javascript">
            function handle_output(out){
                    
            }
            function plot(id,x,y){
                dir = id.split('/');
                
                exec = '{{ plotcommand }}';
                
                var kernel = IPython.notebook.kernel;
                var callbacks = { 'iopub' : {'output' : handle_output}};
                var msg_id = kernel.execute(exec, callbacks, {silent:false});
            }
        </script>
        
        <style type="text/css">
        @media (min-width: 30em) {
            .row { width: 100%; display: table; table-layout: fixed; }
            .col { display: table-cell;  width: 100%;  }
        }
        </style>
        """
        temp = jj.Template(out)
                 
        display(HTML(temp.render(items=all_relative,plotcommand='a=1')))