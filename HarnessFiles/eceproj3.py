## Import all packages
from MPLplot import *

## Import Data
data = pd.read_csv("files/TrishaEce30Proj3Data.csv")

## Create MPLPlot
myplot = MPLPlot(is_multiplot=True, dim=((1, 1)), toolbar=True)

# myplot.add_plot(plot_type=MPLPlot.basic_scatter, 
#                 x=np.array([data.Length[0], data.Length[1], data.Length[2], data.Length[3], data.Length[4], data.Length[5]]), 
#                 y=np.array([data.Resistance[0], data.Resistance[1], data.Resistance[2], data.Resistance[3], data.Resistance[4], data.Resistance[5]]), 
#                 title="Figure 1 - Relation Between Resistor Length and Resistance", 
#                 titlefont=10,
#                 xlabelfontsize=10,
#                 ylabelfontsize=10,
#                 xlabel="Length (cm)", 
#                 ylabel="Resistance (Ohms)", 
#                 label=data.Name,
#                 legend=True,
#                 color=["orange", "red", "teal", "purple", "green", "lime"],
#                 s=35)

# myplot.add_plot(plot_type=MPLPlot.basic_scatter, 
#                 x=np.array([data.Width[0], data.Width[1], data.Width[2], data.Width[3], data.Width[4], data.Width[5]]), 
#                 y=np.array([data.Resistance[0], data.Resistance[1], data.Resistance[2], data.Resistance[3], data.Resistance[4], data.Resistance[5]]), 
#                 title="Figure 2 - Relation Between Resistor Width and Resistance", 
#                 titlefont=10,
#                 xlabelfontsize=10,
#                 ylabelfontsize=10,
#                 xlabel="Width (cm)", 
#                 ylabel="Resistance (Ohms)", 
#                 label=data.Name,
#                 legend=True,
#                 color=["orange", "red", "teal", "green", "purple", "lime"],
#                 s=35)

# myplot.add_plot(plot_type=MPLPlot.basic_plot,
#                 x=data.Width,
#                 y=data.Resistance,
#                 titlefont=10,
#                 xlabelfontsize=10,
#                 ylabelfontsize=10,
#                 title="Figure 3 - Relation Between Width and Resistance",
#                 xlabel="Width(cm)",
#                 ylabel="Resistance (Ohms)",
#                 color="magenta")

# myplot.add_plot(plot_type=MPLPlot.basic_plot,
#                 x=data.Area,
#                 y=data.Resistance,
#                 title="Figure 4 - Relation Between Total Area and Resistance",
#                 titlefont=10,
#                 xlabelfontsize=10,
#                 ylabelfontsize=10,
#                 xlabel="Area (cm^2)",
#                 ylabel="Resistance (Ohms)",
#                 color="brown")

myplot.add_plot(plot_type=MPLPlot.basic_bar,
                height=data.Resistance, 
                xlabel="Width (cm)",
                ylabel="Resistance (Ohms)", 
                title="Figure 2 - Relation Between Resistor Width and Resistance", 
                xticklabels=np.array(["R1 - 2.5cm", "R2 - 2.5cm", "R3 - 2.5cm", "R4 - 5cm", "3x - 7.5cm", "4x - 10cm"]),
                xtickrange=[0,1,2,3,4,5],
                label=data.Name, 
                legend=True, 
                colors=["orange", "red", "teal", "green", "purple", "lime"])



## Show Plot
plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
myplot.show()