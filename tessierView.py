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
def getthumbcachepath(file):
    oneupdir = os.path.abspath(os.path.join(os.path.dirname(file),os.pardir))
    datedir = os.path.split(oneupdir)[1] #directory name should be datedir, if not 
    if re.match('[0-9]{8}',datedir):
        preid= datedir
    else:
        preid = ''
    
    #relative to project/working directory
    cachepath = os.path.normpath(os.path.join(os.getcwd(),'thumbnails', preid + '_'+os.path.splitext(os.path.split(file)[1])[0] + '_thumb.png'))
    return cachepath
    
def getthumbdatapath(file):
    thumbdatapath = os.path.splitext(file)[0] + '_thumb.png'
    return thumbdatapath


def overridethumbnail(file, fig):
##Todo, copy thumbs on the fly from their working directories, thus negating having to recheck..the..data..file?
    #create a thumbnail and store it in the same directory and in the thumbnails dir for local file serving, override options for if file already exists
    thumbfile = getthumbcachepath(file)
    thumbfile_datadir =  getthumbdatapath(file)
    try:
        pylab.rcParams.update(rcP)
        fig.savefig(thumbfile,bbox_inches='tight' )
        fig.savefig(thumbfile_datadir,bbox_inches='tight' )
        plt.close(fig)
    except Exception,e:
        thumbfile = None #if fail no thumbfile was created
        print 'Error {:s} for file {:s}'.format(e,file)
        pass
        
    #put back the old settings
    pylab.rcParams = rcP_old

    return thumbfile


class tessierView(object):
    def __init__(self, rootdir='./', filemask='.*\.dat(?:\.gz)?$',filterstring=''):
        self._root = rootdir
        self._filemask = filemask
        self._filterstring = filterstring
        self._allthumbs = []
    def on(self):   
        print 'You are now watching through the glasses of ideology'
        display(VimeoVideo('106036638'))
          
#     def getsetfilepath(self,file):
#         return (os.path.splitext(file)[0] + '.set')
    def getsetfilepath(self,file):
        file_Path, file_Extension = os.path.splitext(file)
        if   file_Extension ==  '.gz':
            file_Path = os.path.splitext(file_Path)[0]
        elif file_Extension != '.dat':
            print 'Wrong file extension'
#         print file_Path
        return (file_Path + '.set')

    def makethumbnail(self, file,override=False,style=[]):
        #create a thumbnail and store it in the same directory and in the thumbnails dir for local file serving, override options for if file already exists
        thumbfile = getthumbcachepath(file)
        thumbfile_datadir =  getthumbdatapath(file)

        try:
            if os.path.exists(thumbfile):
                thumbnailStale = os.path.getmtime(thumbfile) < os.path.getmtime(file)   #check modified date with file modified date, if file is newer than thumbfile refresh the thumbnail
            if ((not os.path.exists(thumbfile)) or override or thumbnailStale):
                #now make thumbnail because it doesnt exist or if u need to refresh
                pylab.rcParams.update(rcP)

                p = ts.plotR(file)
                if len(p.data) > 20: ##just make sure really unfinished measurements are thrown out
                    guessStyle = p.guessStyle()
                    p.quickplot(style=guessStyle + style)
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
                #print fullpath
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
        #print all_relative
        # print all_relative
        out=u"""

        <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
        <meta http-equiv="Pragma" content="no-cache"> 
        <meta http-equiv="Expires" content="0">
        
        <div>
    {% for item in items %}

    {% if loop.index % 3 == 1 %}
        <div class='row'>
    {% endif %}

        <div class='col'>
<!-- 
        <div class="output_area">
            <div class="output_subarea output_png output_result">
 -->
     <div class='thumb'>
                <img  class="ui-resizable" src="{{ item.thumbpath }}"/> 


                <div class="ui-resizable-handle ui-resizable-e" style="z-index: 90; display: none;"></div>
                <div class="ui-resizable-handle ui-resizable-s" style="z-index: 90; display: none;"></div>  
                <div class="ui-resizable-handle ui-resizable-se ui-icon ui-icon-gripsmall-diagonal-se" style="z-index: 90; display: none;"></div>
     </div>
<!-- 
</div>
</div>
 -->
            <div class='controls'>
    
            <button id='{{ item.datapath }}' onClick='toclipboard(this.id)'>Filename to clipboard</button>
            <br/>
            <button id='{{ item.datapath }}' onClick='refresh(this.id)'>Refresh</button>

            <br/>
            <button id='{{ item.datapath }}' onClick='plotwithStyle(this.id)' class='plotStyleSelect'>Plot with</button>
            <form name='{{ item.datapath }}'>
            <select name="selector">
                <option value="{{"[\\'\\']"|e}}">normal</option>
                <option value="{{"[\\'log\\']"|e}}">log</option>
                <option value="{{"[\\'savgol\\',\\'log\\']"|e}}">savgol,log</option>
                <option value="{{"[\\'sgdidv\\']"|e}}">sgdidv</option>
                <option value="{{" [\\'sgdidv\\',\\'log\\']"|e}} ">sgdidv,log</option>
                <option value="{{" [\\'mov_avg\\',\\'didv\\',\\'abs\\']"|e}} ">mov_avg,didv,abs</option>
                <option value="{{" [\\'mov_avg\\',\\'didv\\',\\'abs\\',\\'log\\']"|e}} ">mov_avg,didv,abs,log</option>
                <option value="{{" [\\'deinterlace\\']"|e}} ">deinterlace</option>
                <option value="{{" [\\'crosscorr\\']"|e}} ">Crosscorr</option>
            </select>
            
            </form>            
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
            function pycommand(exec){
                exec = exec.replace(/\\\\/g,"\\\\\\\\");
                var kernel = IPython.notebook.kernel;
                var callbacks = { 'iopub' : {'output' : handle_output}};
                var msg_id = kernel.execute(exec, callbacks, {silent:false});
            }
            function tovar(id) {
                exec =' file \= \"' + id + '\"';
                pycommand(exec);
            }
            function toclipboard(id) {
                exec ='import pyperclip; pyperclip.copy(\"' + id + '\");pyperclip.paste()';
                pycommand(exec);
            }
            function refresh(id) {
                exec =' import tessierView as tv;  a=tv.tessierView();a.makethumbnail(\"' + id + '\",override=True)';
                pycommand(exec);
            }
            function getStyle(id) {
                id = id.replace(/\\\\/g,"\\\\\\\\");
                var x = document.querySelectorAll('form[name=\\"'+id+'\\"]')
                form = x[0]; //should be only one form
                selector = form.selector;
                var style = selector.options[selector.selectedIndex].value;
                return style
            }
            function plotwithStyle(id) {
                var style = getStyle(id);
                plot(id,style);
            }
            
            function plot(id,x,y){
                dir = id.split('/');
                
                exec = 'file \= \"' + id + '\"; {{ plotcommand }}';
                exec = exec.printf(x)

                pycommand(exec);
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
            .row { width: auto; display: table; table-layout: fixed; }
            .col { display: table-cell;  width: auto;  }
        }
        img{
            width:100%;
            height:auto;
            }
        </style>
        """
        temp = jj.Template(out)

        plotcommand = """import tessierPlot as ts; reload(ts); p = ts.plotR(file); p.quickplot(style= %s) """      
        return temp.render(items=all_relative,plotcommand=plotcommand)
