import numpy as np, pandas as pd
import matplotlib.pyplot as plt

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

def basic_bar(data, ax, column, labels = None, title = None, xlabel = None, ylabel = None, xtickrange = None, xticklabels = None, colors = None, width = .3, legend = False, legendsize = 8):

    ## Format data
    # Transpose column data
    row = column.T

    # Create horizontal axis
    x = np.arange(len(column))

    # Set default color to teal if not specified
    if colors is None:
        colors = ["teal" for x in range(data.shape[0])]

    # Set default xticklabels if not specified
    if xticklabels is None:
        xticklabels = [x for x in range(data.shape[0])]

    # Set default xtickrange if not specified
    if xtickrange is None:
        xtickrange = range(data.shape[0])

    # Label Axes
    if title is not None:
        ax.set_title("{}".format(title))

    # Add x labels if specified
    if xlabel is not None:
        ax.set_xlabel("{}".format(xlabel))

    # Add y labels if specified
    if ylabel is not None:
        ax.set_ylabel("{}".format(ylabel))

    ## Create Plot

    # Plot Bar Data
    ax.bar(x, row, width = width, color = colors, label = labels)

    ## Add Legend
    if legend:
        ax.legend(fontsize = legendsize)
    return

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

        if self.is_multiplot or dim != [1,1]:
            self.is_multiplot = True
            self.fig, self.axs = plt.subplots(dim[0], dim[1])
            print("hey! Here are your blank axs")
            # print(len(self.axs))
        else:
            dim = [1,1]
            self.fig, ax = plt.subplots(dim[0], dim[1])

    def show(self):
        plt.show()


def main():
    """Goal
    Create plots of all the data
    Calculate the Mean and Standard Deviation
    Run a Hypothesis Test on Null: There is a correlation between height and speed
    """
    data = pd.read_csv("files/csvsets/Ece30Asgn1Data.csv")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.11, top=0.929, wspace=0.204, hspace=0.249)

    # create Linear Graph of Height vs Stride Length
    heights_vs_stride(data, ax1)

    # Create Multi Bar Graph of Trial Times
    stride_vs_time(data, ax2)

    # Create Basic Bar Graph of Average Time
    basic_bar(data,
              ax3,
              data.AvgTime,
              labels = data.Name,
              title = "Average Effect of Stride Length (cm) on Travel Time (s)",
              xlabel = "Participants",
              ylabel = "Time (seconds)",
              xticklabels = data.Name,
              colors = ['#ff6d01', '#abed9b', '#5546ea', '#fbbd05', '#cc0100'],
              width = 0.5,
              legend = True,
              legendsize = 6)

    # Dummy test basic bar
    basic_bar(data,
              ax4,
              data.AvgStrides)

    ## First: Create a plot of Effect of Height (cm) on Stride Length (cm)
    # Need: Heights - Stride Lengths - People - Line Graph
    plt.show()
    print("hotdog")
    myplot = MPLPlot(is_multiplot=True)






if __name__ == "__main__":
    main()