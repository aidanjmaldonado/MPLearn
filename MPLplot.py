import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from math import *

## import pezplot as pp

def heights_vs_stride(data, ax):
    # Format data
    heights = np.array(data.Height)
    SL = np.array(data.StrideLength)

    # Label axes
    ax.set_title("Effect of Height (cm) on Stride Length (cm)")
    ax.set_xlabel("Participants")
    ax.set_xticks(range(data.shape[0]))
    ax.set_xticklabels(data.Name)
    ax.set_ylabel("Measurements (cm)")

    # Restrict axes
    ax.set_xlim(-.5, 4.5)
    ax.set_ylim(50, 220)

    # Plot Data
    ax.plot(heights, '-o', markersize=5, label="Height (cm)", color ="teal")
    ax.plot(SL, '-o', markersize=5, label="Stride Length (cm)", color="darkorange")

    # Add Point Labels
    for y in range(data.shape[0]):
        ax.annotate("{} cm".format(heights[y]), xy = (y, heights[y]), xytext=(y, heights[y] + 10), arrowprops=dict(arrowstyle="-", linewidth=0.5, color='teal'), fontsize = 6, fontweight='bold', horizontalalignment='center', color='teal')
        ax.annotate("{} cm".format(SL[y]), xy = (y, SL[y]), xytext=(y, SL[y] + 10), arrowprops=dict(arrowstyle="-", linewidth=0.5, color='darkorange'), fontsize = 6, fontweight='bold', horizontalalignment='center', color='darkorange')

    # Add Legend
    ax.legend(fontsize = 6)

def stride_vs_time(data, ax):
    # Format data
    trials_transposed = np.array(data[['T1Time', 'T2Time', 'T3Time', 'T4Time', 'T5Time']]).T
    x = np.arange(len(trials_transposed))

    # Plot Bar Data
    colors = ['#ff6d01', '#abed9b', '#5546ea', '#fbbd05', '#cc0100']
    for y in range(5):
        ax.bar((x-.237)+y*.12, trials_transposed[y], width=0.10, color=colors[y], label=data.Name[y])

    # Label Axes
    ax.set_title("[5 Trials] Effect of Stride Length (cm) on Travel Time (s)")
    ax.set_xlabel("Participants")
    ax.set_xticks(range(data.shape[0]))
    ax.set_xticklabels(["Trial {} Time".format(y) for y in range(data.shape[0])])
    ax.set_ylabel("Time (seconds)")

    # Add Legend
    ax.legend(fontsize = 8)

def basic_mbar():
    return

def basic_scatter():
    return
    
def basic_line():
    return


class MPLPlot():
    def __init__(self, is_multiplot = False, dim: tuple[int, int] = (1,1)):
        self.is_multiplot = is_multiplot
        self.dim = dim
        self.rows, self.cols = dim[0], dim[1]
        if self.cols < 1: self.cols, self.dim = 1, (self.rows, 1)
        if self.rows < 1: self.rows, self.dim = 1, (1, self.cols)
        self.filled = np.zeros(dim)
        self.index = 1

        if self.is_multiplot: #and self.dim != (1,1)
            self.is_multiplot = True
            
            # Create the fig and establish the axs array
            self.fig, self.axs = plt.subplots(self.dim[0], self.dim[1])
            if self.cols <= 2:
                plt.subplots_adjust(left=0.063, right=0.943, bottom=0.10, top=0.917, wspace=0.156, hspace=0.51)
            else:
                plt.subplots_adjust(left=0.125, right=0.9, bottom=0.04, top=0.965, wspace=0.204, hspace=0.26)

            # Store the axs array in a numpy array
            self.axs = np.array(self.axs)

            # Reshape to eliminate 0 dimension
            if (self.dim[0] == 1):
                self.axs = np.array([self.axs])
                print("Fixed x", self.axs.shape)
            if (self.dim[1] == 1):
                self.axs = self.axs.reshape(-1, 1)
                print("Fixed y", self.axs.shape)
            
        else:
            print("wtf did we go here?\n\n")
            dim = [1,1]
            self.fig, ax = plt.subplots(self.dim[0], self.dim[1])


    # def add_function(type: function, ax)
    def add_plot(self, plot_type: callable, **specifications):

        # Grand filtering/defaulting: 
        specifications = MPLPlot.grand_filtering(specifications)
 
        # Determine which plot to place in ## Need some kind of error handling
        while (True):
            if (self.index > self.cols * self.rows):
                print("multiplot full!")
                return
            columnindex = ((self.index % self.cols) - 1)
            rowindex = ((ceil(self.index / self.cols)) - 1)
            self.index += 1
            if (self.filled[rowindex, columnindex] == 0):
                self.filled[rowindex, columnindex] = 1
                break

        # Plot
        self.plot(plot_type, rowindex, columnindex, specifications)
        return
        
    def insert_plot(self, plot_type: callable, pindex: int, **specifications):
        
        # Out of Bounds Error
        if (pindex < 0 or pindex > self.rows * self.cols):
            print("Error, out of bounds pindex")
            return 

        # Grand filtering/defaulting: 
        specifications = MPLPlot.grand_filtering(specifications)

                # Determine which plot to place in ## Need some kind of error handling
        columnindex = ((pindex % self.cols) - 1)
        rowindex = ((ceil(pindex / self.cols)) - 1)
        if (self.filled[rowindex, columnindex] == 0):
            self.filled[rowindex, columnindex] = 1
            self.plot(plot_type, rowindex, columnindex, specifications)
            return

        while (True):
            if (self.index > self.cols * self.rows):
                print("multiplot full!")
                return
            columnindex = ((self.index % self.cols) - 1)
            rowindex = ((ceil(self.index / self.cols)) - 1)
            self.index += 1
            if (self.filled[rowindex, columnindex] == 0):
                self.filled[rowindex, columnindex] = 1
                break


        # Plot 
        self.plot(plot_type, rowindex, columnindex, specifications)
        return

    def grand_filtering(specifications: dict):
        if 'data' not in specifications:
            specifications['data'] = None
            print("Empty graph created, no data passed in")
            return
        if 'height' not in specifications:
            specifications['height'] = None
        if 'title' not in specifications:
            specifications['title'] = ''
        if 'titlefont' not in specifications:
            specifications['titlefont'] = 8
        if 'color' not in specifications:
            specifications['color'] = 'teal'
        if 'colors' not in specifications:
            specifications['colors'] = ["teal" for x in range(specifications['data'].shape[0])]
        if 'label' not in specifications:
            specifications['label'] = None
        if 'xlabel' not in specifications:
            specifications['xlabel'] = ''
        if 'xlabelfontsize' not in specifications:
            specifications['xlabelfontsize'] = 8
        if 'ylabelfontsize' not in specifications:
            specifications['ylabelfontsize'] = 8
        if 'ylabel' not in specifications:
            specifications['ylabel'] = ''
        if 'xticklabel' not in specifications:
            specifications['xticklabel'] = None
        if 'xtickrange' not in specifications:
            specifications['xtickrange'] = [x for x in range(1, specifications['data'].shape[0] + 1)]
        if 'width' not in specifications:
            specifications['width'] = 0.5
        if 'legend' not in specifications:
            specifications['legend'] = False
        if 'legendsize' not in specifications:
            specifications['legendsize'] = 6
        if 'bottom' not in specifications:
            specifications['bottom'] = 0
        if 'align' not in specifications:
            specifications['align'] = 'center'
        if 'xlim' not in specifications:
            specifications['xlim'] = (None,None)
        if 'ylim' not in specifications:
            specifications['ylim'] = (None,None)
        
        return specifications


    def plot(self, plot_type: callable, row: int, col: int, specifications: dict):
        if (plot_type == MPLPlot.basic_bar):
                self.basic_bar(self.axs[row, col], specifications['data'], specifications['height'], bottom=specifications['bottom'], align=specifications['align'], label=specifications['label'], title=specifications['title'], titlefont=specifications['titlefont'], xlabel=specifications['xlabel'], xlabelfontsize=specifications['xlabelfontsize'], ylabel=specifications['ylabel'], ylabelfontsize=specifications['ylabelfontsize'], xticklabel=specifications['xticklabel'], xtickrange=specifications['xtickrange'], color=specifications['colors'], width=specifications['width'], legend=specifications['legend'], legendsize=specifications['legendsize'], xlim=specifications['xlim'], ylim=specifications['ylim'])
        return



    def basic_bar(self, ax, data, height, bottom, align, label=None, title = '', titlefont = 8, xlabel = '', xlabelfontsize = 8, ylabel = '', ylabelfontsize = 8,  xticklabel = None, xtickrange = None, color = None, width = 0.5, legend = False, legendsize = 6, xlim=(None,None), ylim=(None,None)):
        
        # Transpose height input
        row = height.T

        # Create horizontal axis
        x = np.arange(len(height))

        ## Create Plot

        # Plot Bar Data
        ax.bar(x, row, label=label, color=color, width=width, bottom=bottom, align=align)
        ax.set_title(title, fontsize=titlefont)
        ax.set_xlabel(xlabel, fontsize=xlabelfontsize)
        ax.set_ylabel(ylabel, fontsize=ylabelfontsize)
        ax.set_xlim(xlim[0], xlim[1])
        print("ylim = {}".format(ylim))
        ax.set_ylim(ylim[0], ylim[1])


        #Format p2
        if xtickrange is None:
            xtickrange = xtickrange = range(1, data.shape[0] + 1)
            ax.set_xticks(xtickrange)
            if xticklabel is not None:
                ax.set_xticklabels(xticklabel)
            else:
                ax.set_xticklabels([x for x in range(len(height) + 1)]) # Fix this using https://stackoverflow.com/questions/63723514/userwarning-fixedformatter-should-only-be-used-together-with-fixedlocator


        ## Add Legend
        if legend:
            ax.legend(fontsize = legendsize)
        return


    def show(self):
        plt.show()




    # Nowwwww I will
        # Add axis limiting (y-scale) functionality  
            #   ax.set_xlim(-.5, 4.5)
            #    ax.set_ylim(50, 220)
        # Add centering functionality DONE
        # Add Scatter functionality
        # Add Line graph functionality




def main():
    """Goal
    Create plots of all the data
    Calculate the Mean and Standard Deviation
    Run a Hypothesis Test on Null: There is a correlation between height and speed
    """

    # Import data
    data = pd.read_csv("files/csvsets/Ece30Asgn1Data.csv")

    # Create MPLPlot
    myplot = MPLPlot(is_multiplot=True, dim=(5,1))

    # Add Plots
    myplot.add_plot(MPLPlot.basic_bar, data=data, height=data.AvgTime, title="Average Time Baisc Unformated Plot")
    myplot.add_plot(MPLPlot.basic_bar, data=data, height=data.AvgTime, label=data.Name, title="Average Effect of Stride Length (cm) on Travel Time (s)", xlabel="Participants", ylabel="Time (seconds)", xticklabels=data.Name, colors=['#ff6d01', '#abed9b', '#5546ea', '#fbbd05', '#cc0100'], width=0.5, legend=True, legendsize=6)
    
    myplot.insert_plot(MPLPlot.basic_bar, 5, data=data, height=data.AvgVelocity)
    myplot.insert_plot(MPLPlot.basic_bar, 1, data=data, height=data.AvgVelocity, colors='yellow')
    myplot.insert_plot(MPLPlot.basic_bar, 1, data=data, height=data.AvgVelocity, colors='magenta')
    myplot.insert_plot(MPLPlot.basic_bar, 1, data=data, height=data.AvgVelocity, colors='brown')


    # myplot.add_plot(MPLPlot.basic_bar, data=data, height=data.AvgStrides, title="Average Strides Basic Plot", colors='red')
    # myplot.add_plot(MPLPlot.basic_bar, data=data, height=data.AvgVelocity, title="Average Velocity Basic Plot", colors='purple')
    # myplot.add_plot(MPLPlot.basic_bar, data=data, height=data.T2Velocity, title="Trial 2 Velocity", xlabel="Who's the fastest?", colors='green')



    #MPLPlot.show()
    myplot.show()



if __name__ == "__main__":
    main()