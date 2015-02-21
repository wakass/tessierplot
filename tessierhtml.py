#tessierMagic, where the real % happens

import pandas as pd
import pylab
import numpy as np
import scipy.interpolate as spinterp
import os, ntpath
import re as regex

import matplotlib.pyplot as plt

import tessierPlot as ts
from IPython.display import Image
from IPython.display import display, HTML



mainfolder = os.path.normpath('../data/heliox/data/')
devicename = 'T5_2_NR_Postanneal'

htmlthumbnailpath = './files/thumbnails'
thumbnailpath = './thumbnails'

htmlout = """<table>
"""
nImages = 0 #image counter

plot_width = 4. # in inch (ffing inches eh)
plot_height = 3. # in inch

fontsize_plot_title = 10
fontsize_axis_labels = 10
fontsize_axis_tick_labels = 10

plotstyle = 'normal'
fiddleButton = False

filebuff = ''

pylab.rcParams['figure.figsize'] = (plot_width, plot_height) #(width in inch, height in inch)
pylab.rcParams['axes.labelsize'] = fontsize_axis_labels
pylab.rcParams['xtick.labelsize'] = fontsize_axis_tick_labels
pylab.rcParams['ytick.labelsize'] = fontsize_axis_tick_labels

reload(ts)

class plot2dstuff:
    data = []
    sweepfig = None
    
    def __init__(self,data,axes):
        self.data = data
        cols = data.columns.tolist()
        ax = data.plot(x=cols[0],y=cols[1])

        self.sweepfig = ax.get_figure()
        ax.set_xlabel(axes[0])
        ax.set_ylabel(axes[1])
        ax.locator_params(nbins=4)

for root, dirs, files in os.walk(mainfolder):
    
    folder = ntpath.basename(root)
    #first check out if we're in a 'date' folder (i.e. 19821105)
    if regex.match('[0-9]{8}',folder):
        currentdate = folder
        foldercell = "<tr><td style='font-weight:bold'>" + folder + "</td></tr>\n"
        htmlout = htmlout + foldercell
        nImages = 0
        
    if regex.match('[0-9]{6}_',folder):
        if (nImages % 3) == 0:
            htmlout = htmlout + "</tr>\n<tr>"
        
        path = root + '/' + folder + '.dat'
            
        columnNames,nCommentLines = ts.parseheader(path)
        nDim = len(columnNames)
    	
    	axesNames = []
    	unitNames = []
    	for i in columnNames:
    		axisName, unitName = ts.parseUnitAndNameFromColumnName(i)
    		axesNames = axesNames + [axisName]
    		unitNames = unitNames + [unitName]
    	
        labellist = regex.findall('(?<=\{).*?(?=\})',s)
        
        htmlout = htmlout + "<td>\n" + folder
        
        thumbnailfile = devicename + '_' + currentdate + '_' + folder[:6] + '.png'
        
        #let's just assume for now that a file with only two axes is a sweep fig
        if nDim==2:
            axesnames = [axesNames[0]+' ['+unitNames[0]+']',axesNames[1]+' ['+unitNames[1]+']']
            if not ('thumbnail.png' in files):
                names,skiprows = ts.parseheader(path) 
                data = ts.loadFile(path,names=names,skiprows=skiprows)
                
                p=plot2dstuff(data,axesnames)
                p.sweepfig.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.85,hspace=0.0)
                p.sweepfig.savefig(os.path.join('./thumbnails/',thumbnailfile))
                p.sweepfig.savefig(os.path.join(root,'thumbnail.png'))
                plt.close(p.sweepfig)
                
            imagefile = os.path.join(htmlthumbnailpath, thumbnailfile)
            imagefile = imagefile.replace("\\", "/")
            jslabels = "'\\\'" + labellist[0] + "\\\''" + ", '\\\'" + labellist[2] + "\\\''"
            htmlroot = root.replace("\\","/")
            
            htmlout = htmlout + """
<br />
<img src=\"""" + imagefile + """\"/> 
<button id='""" + htmlroot + """' onClick=\"plot2d(this.id,""" + jslabels + """)\">Plot</button>
<br />
"""
        
        if nDim >= 3: #3 axes: gate1, gate2, i_sd
            thumbnailfile = devicename + '_' + currentdate + '_' + folder[:6] + '.png'
            axesnames = [labellist[0],labellist[2],labellist[4]]
            
            if not ('thumbnail.png' in files):          
                names,skiprows = ts.parseheader(path) 
                data = ts.loadFile(path,names=names,skiprows=skiprows)
                
                # set uniques_col_str to remainder rest of parameters?
                print names,skiprows
                try:
                    p = ts.plot3DSlices(data,fiddle=fiddleButton,style=plotstyle,uniques_col_str=[])
                
                    p.fig.subplots_adjust(top=0.9, bottom=0.15, left=0.2, right=0.85,hspace=0.0)
                    p.fig.get_axes()[0].locator_params(nbins=4)
                    p.fig.savefig(os.path.join('./thumbnails/',thumbnailfile))
                    p.fig.savefig(os.path.join(root,'thumbnail.png'))
                    plt.close(p.fig)
                except:
                    pass
                
            imagefile = os.path.join(htmlthumbnailpath, thumbnailfile)
            imagefile = imagefile.replace("\\", "/")
            jslabels = "'\\\'" + labellist[0] + "\\\''" + ", '\\\'" + labellist[2] + "\\\'', '\\\'" + labellist[4] + "\\\''"
            htmlroot = root.replace("\\","/")
            htmlout = htmlout + """
<br />
<img src=\"""" + imagefile + """\"/> 
<br />
<div style='font-size:120%'>""" + labellist[0] + " vs. " + labellist[2] + """</div>
<br />
<button id='""" + htmlroot + """' onClick=\"plot3d(this.id,\'normal\',""" + jslabels + """)\">Normal</button>
<button id='""" + htmlroot + """' onClick=\"plot3d(this.id,\'log\',""" + jslabels + """)\">Log</button>
<button id='""" + htmlroot + """' onClick=\"plot3d(this.id,\'didv\',""" + jslabels + """)\">dIdV</button>
<button id='""" + htmlroot + """' onClick=\"plot3d(this.id,\'didv2\',""" + jslabels + """)\">Log(dIdV)</button>
<button id='""" + htmlroot + """' onClick=\"dirtovar(this.id,""" + jslabels + """)\">Save dir to var</button>
</td>"""
        
        nImages = nImages + 1

htmlout = htmlout + """
</table>

<script type="text/Javascript">
    function dirtovar(id,x,y,z) {
        dir = id.split('/')
        datafile = id + '/' + dir[dir.length-1] + '.dat'
        exe1 = "filebuff = '" + datafile + "'"
        //exe2 = "data = ts.loadFile(file,names=" + axesnames + ",skiprows=19)"
        //exe3 = "p= ts.plot3DSlices(data,fiddle=True,style='" + type + "',uniques_col_str=[])"
        //exe4 = "p.fig.suptitle(data.name+'', fontsize=fontsize_plot_title)\\np.fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9,hspace=0.0)\\nts.plt.show()\\nprint file"
        
        exec = exe1// + '\\n' + exe2 + '\\n' + exe3 + '\\n' + exe4
                
        var kernel = IPython.notebook.kernel;
        var callbacks = { 'iopub' : {'output' : handle_output}};
        var msg_id = kernel.execute(exec, callbacks, {silent:false});
    }
    function handle_output(out){
                
    }
    
    function plot3d(id,type,x,y,z) {
        dir = id.split('/')
        datafile = id + '/' + dir[dir.length-1] + '.dat'
        //alert(type) 
        
        exe1 = "file = '" + datafile + "'"
        exe2 = "names,skiprows=ts.parseheader(file)
        exe3 = "data = ts.loadFile(file,names=names,skiprows=skiprows)"
        exe4 = "p= ts.plot3DSlices(data,fiddle=True,style='" + type + "',uniques_col_str=[])"
        exe5 = "p.fig.suptitle(data.name+'', fontsize=fontsize_plot_title)\\np.fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9,hspace=0.0)\\nts.plt.show()\\nprint file"
        
        exec = exe1 + '\\n' + exe2 + '\\n' + exe3 + '\\n' + exe4 + '\\n' + exe5
                
        var kernel = IPython.notebook.kernel;
        var callbacks = { 'iopub' : {'output' : handle_output}};
        var msg_id = kernel.execute(exec, callbacks, {silent:false});
        //alert(exec)
        
    }
    function plot2d(id,x,y) {
        dir = id.split('/')
        datafile = id + '/' + dir[dir.length-1] + '.dat'
        //alert(id) 
        
        
        exe1 = "file = '" + datafile + "'"
        exe2 = "names,skiprows=ts.parseheader(file)
        exe3 = "data = ts.loadFile(file,names=names,skiprows=skiprows)"
        exe4 = "p= plot2dstuff(data,names)"
        exe5 = "p.sweepfig.suptitle(data.name+'', fontsize=fontsize_plot_title)\\np.fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9,hspace=0.0)\\nts.plt.show()"
        
        exec = exe1 + '\\n' + exe2 + '\\n' + exe3 + '\\n' + exe4 + '\\n' + exe5
                
        var kernel = IPython.notebook.kernel;
        var callbacks = { 'iopub' : {'output' : handle_output}};
        var msg_id = kernel.execute(exec, callbacks, {silent:false});
    }
</script>
"""

plot_width = 15. # in inch (ffing inches eh)
plot_height = 10. # in inch

fontsize_plot_title = 16
fontsize_axis_labels = 18
fontsize_axis_tick_labels = 18

pylab.rcParams['figure.figsize'] = (plot_width, plot_height) #(width in inch, height in inch)
pylab.rcParams['axes.labelsize'] = fontsize_axis_labels
pylab.rcParams['xtick.labelsize'] = fontsize_axis_tick_labels
pylab.rcParams['ytick.labelsize'] = fontsize_axis_tick_labels

#print htmlout
display(HTML(htmlout))


