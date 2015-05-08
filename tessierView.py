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
from IPython.display import display, HTML, display_html


reload(ts)

_plot_width = 4. # in inch (ffing inches eh)
_plot_height = 3. # in inch

_fontsize_plot_title = 10
_fontsize_axis_labels = 10
_fontsize_axis_tick_labels = 10

plotstyle = 'normal'

rcP = {  'figure.figsize': (_plot_width, _plot_height), #(width in inch, height in inch)
        'axes.labelsize':  _fontsize_axis_labels,
        'xtick.labelsize': _fontsize_axis_tick_labels,
        'ytick.labelsize': _fontsize_axis_tick_labels,
        'legend.fontsize': 5.
        }
        
rcP_old = pylab.rcParams.copy()

# pylab.rcParams['legend.linewidth'] = 10

class tessierView(object):
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
    ##Todo, copy thumbs on the fly from their working directories, thus negating having to recheck..the..data..file?
    
    
        #create a thumbnail and store it in the same directory and in the thumbnails dir for local file serving, override options for if file already exists
        thumbfile = self.getthumbcachepath(file)
        thumbfile_datadir =  self.getthumbdatapath(file)
        try:
            if ((not os.path.exists(thumbfile)) or override):
                #now make thumbnail because it doesnt exist or if u need to refresh
                pylab.rcParams.update(rcP)

                p = ts.plotR(file)
                if len(p.data) > 20: ##just make sure really unfinished measurements are thrown out
                    p.quickplot() #make quickplot more intelligent so it detect dimensionality from uniques
#                     p.fig.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.85,hspace=0.0)
                    p.fig.savefig(thumbfile,bbox_inches='tight' )
                    p.fig.savefig(thumbfile_datadir,bbox_inches='tight' )
                    plt.close(p.fig)
                else:
                    thumbfile = None
        except Exception,e:
            thumbfile = None #if fail no thumbfile was created
            print 'Error {:s} for file {:s}'.format(e,file)
            pass
            
        #put back the old settings
        pylab.rcParams = rcP_old
   
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
                        thumbpath = self.makethumbnail(fullpath,**kwargs)
                        if thumbpath:
                            self._allthumbs.append({'datapath':fullpath,'thumbpath':thumbpath})
                            images += 1
                            
        return self._allthumbs
    def _ipython_display_(self):
        display_html(HTML(self.genhtml(refresh=False)))
    
    def htmlout(self,refresh=False):
        display(HTML(self.genhtml(refresh=refresh)))
        
    def genhtml(self,refresh=False):
        self.walk(self._filemask,'dac',override=refresh)
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
            <button id='{{ item.datapath }}' onClick='plot(this.id,"&#91;&#93;","dsf")'>Normal</button>
            <button id='{{ item.datapath }}' onClick='plot(this.id,"{{"[\\'abs\\',\\'log\\']"|e}}","dsf")'>Log</button>
            <button id='{{ item.datapath }}' onClick='plot(this.id,"{{"[\\'savgol\\',\\'didv\\',\\'abs\\']"|e}}","dsf")'>dIdV</button>
            <button id='{{ item.datapath }}' onClick='plot(this.id,"{{" [\\'savgol\\',\\'didv\\',\\'log\\']"|e}} ","dsf")'>Log(dIdV)</button>

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
                
                exec = 'file \= \"' + id + '\"; {{ plotcommand }}';
                exec = exec.printf(x)

                var kernel = IPython.notebook.kernel;
                var callbacks = { 'iopub' : {'output' : handle_output}};
                var msg_id = kernel.execute(exec, callbacks, {silent:false});
            }
            String.prototype.printf = function (obj) {
                var useArguments = false;
                var _arguments = arguments;
                var i = -1;
                if (typeof _arguments[0] == "string") {
                useArguments = true;
                }
                if (obj instanceof Array || useArguments) {
                return this.replace(/\%s/g,
                function (a, b) {
                  i++;
                  if (useArguments) {
                    if (typeof _arguments[i] == 'string') {
                      return _arguments[i];
                    }
                    else {
                      throw new Error("Arguments element is an invalid type");
                    }
                  }
                  return obj[i];
                });
                }
                else {
                return this.replace(/{([^{}]*)}/g,
                function (a, b) {
                  var r = obj[b];
                  return typeof r === 'string' || typeof r === 'number' ? r : a;
                });
                }
            };

        </script>
        
        <style type="text/css">
        @media (min-width: 30em) {
            .row { width: 100%; display: table; table-layout: fixed; }
            .col { display: table-cell;  width: 100%;  }
        }
        </style>
        """
        temp = jj.Template(out)
        plotcommand = """import tessierPlot as ts; p = ts.plotR(file); p.quickplot(style=%s) """      
        return temp.render(items=all_relative,plotcommand=plotcommand)