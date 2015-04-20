# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:24:16 2015

@author: sara

Draws a scatterplot of a set of random points of variable size and color.
 - This uses the non-standard renderer, VariableSizeScatterPlot
 - Left-drag pans the plot.
 - Mousewheel up and down zooms the plot in and out.
 - Pressing "z" brings up the Zoom Box, and you can click-drag a rectangular
   region to zoom.  If you use a sequence of zoom boxes, pressing control-y and
   control-z  (Meta-y and Meta-z on Mac) moves you forwards and backwards
   through the "zoom history".
"""




# Major library imports
from numpy import *


# Enthought library imports
from enable.api import Component, ComponentEditor
from traits.api import HasTraits, Instance
from traitsui.api import Item, Group, View

# Chaco imports
from chaco.api import ArrayPlotData, Plot, ColormappedScatterPlot, \
        LinearMapper, ArrayDataSource, jet, DataRange1D
from chaco.tools.api import PanTool, ZoomTool

def scatterplot():
    
    print("scatter plot..")
    
    
    
    
def generateColorCodes(n):
    
    print("generating color codes.. ")
    
    
    
def findBoundary():
    
    print("finding boundary..")
    
    
    
def convertDistanceToColor():
    
    print("converting distance to color ...")
    
    
    
def measureDistanceFromBoundary():
    
    print("calculating distance...")
    
    
def mapColorToImage():
    
    print("mapping colormap back to image ...")
    
    
    
    
def _create_plot_component():     
    numpts = 50
    x = random.rand(numpts)
    y = random.rand(numpts)
    cls = random.rand(numpts)
    colors = [1.0 if i>0.5 else 0.0  for i in cls]
    marker = [1.0 if i>0.5 else 0.0  for i in cls]
    #plt.figure()
    #plt.scatter(x,y,c=colors)
    #plt.show()
    
     # Create some data
  
#    marker_size = numpy.random.normal(4.0, 4.0, numpts)
   

    # Create a plot data object and give it this data
    pd = ArrayPlotData()
    pd.set_data("index", x)
    pd.set_data("value", y)

    # Because this is a non-standard renderer, we can't call plot.plot, which
    # sets up the array data sources, mappers and default index/value ranges.
    # So, its gotta be done manually for now.

    index_ds = ArrayDataSource(x)
    value_ds = ArrayDataSource(y)
    color_ds = ArrayDataSource(colors)

    # Create the plot
    plot = Plot(pd)
    plot.index_range.add(index_ds)
    plot.value_range.add(value_ds)

    # Create the index and value mappers using the plot data ranges
    imapper = LinearMapper(range=plot.index_range)
    vmapper = LinearMapper(range=plot.value_range)

    # Create the scatter renderer
    scatter = ColormappedScatterPlot(
                    index=index_ds,
                    value=value_ds,
                    color_data=color_ds,
                    color_mapper=jet(range=DataRange1D(low=0.0, high=1.0)),
                    fill_alpha=0.4,
                    index_mapper = imapper,
                    value_mapper = vmapper,
                    marker='circle',
                    marker_size = 4)

    # Append the renderer to the list of the plot's plots
    plot.add(scatter)
    plot.plots['var_size_scatter'] = [scatter]

    # Tweak some of the plot properties
    plot.title = "Colored Scatter Plot"
    plot.line_width = 0.5
    plot.padding = 50

    # Attach some tools to the plot
    plot.tools.append(PanTool(plot, constrain_key="shift"))
    zoom = ZoomTool(component=plot, tool_mode="box", always_on=False)
    plot.overlays.append(zoom)

    return plot


#===============================================================================
# Attributes to use for the plot view.
size = (650, 650)
title = "color scatter plot"
bg_color="lightgray"

#===============================================================================
# # Demo class that is used by the demo.py application.
#===============================================================================
class Demo(HasTraits):
    plot = Instance(Component)

    traits_view = View(
                    Group(
                        Item('plot', editor=ComponentEditor(size=size,
                                                            bgcolor=bg_color),
                             show_label=False),
                        orientation = "vertical"),
                    resizable=True, title=title
                    )

    def _plot_default(self):
         return _create_plot_component()

demo = Demo()

if __name__ == "__main__":
    demo.configure_traits()
