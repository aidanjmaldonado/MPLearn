## Import all packages
from MPLplot import *

## Import data
data = pd.read_csv("files/csvsets/Ece30Asgn1Data.csv")

## Create MPLPlot
myplot = MPLPlot(is_multiplot=True, dim=((3, 3)), toolbar=False)

## Add Plots

# Bar Plots
myplot.add_plot(MPLPlot.basic_bar, height=data.AvgTime, label=data.Name, title="Average Effect of Stride Length (cm) on Travel Time (s)", xlabel="Participants", ylabel="Time (seconds)", xticklabels=data.Name, colors=['#ff6d01', '#abed9b', '#5546ea', '#fbbd05', '#cc0100'], width=0.5, legend=True, legendsize=6)
myplot.add_plot(MPLPlot.basic_bar, height=data.AvgStrides, width=0.5, colors='limegreen')
myplot.add_plot(MPLPlot.basic_bar, height=np.array([data.AvgStrides, data.AvgVelocity]), width=0.1, colors=["purple", "salmon"], xticklabels = ["Average Strides", "Average Velocity"])

# Standard Plots
myplot.add_plot(MPLPlot.basic_plot, x=[0,1,2,3,4], y=data.Height, xlabel = "Participant", ylabel = "Height (cm)", title="Height Per Participant", xticklabels=data.Name, label="Height", legend=True, color="Blue")
myplot.add_plot(MPLPlot.basic_plot, x=[0,1,2,3,4], y=data.StrideLength, xlabel = "Participant", ylabel = "Stride Length (cm)", title="Stride Length Per Person", xticklabels=data.Name, label="Stride Length", legend=True, color="Red")
myplot.add_plot(MPLPlot.basic_plot, x=[0,1,2,3,4], y=np.array([data.Height, data.StrideLength]), xlabel = "Participant", ylabel = "Centimeters", title="Height/Stride Length Per Participant", xticklabels=data.Name, label=["Height", "Stride Length"], legend=True, color=["Blue", "Red"], ylim=(50,180))

# Scatter Plots
myplot.add_plot(MPLPlot.basic_scatter, x=[0,1,2,3,4], y=data.Height, xlabel = "Participant", ylabel = "Height (cm)", title="Height Per Participant", xticklabels=data.Name, label="Height", legend=True, color="Blue")
myplot.add_plot(MPLPlot.basic_scatter, x=[0,1,2,3,4], y=data.StrideLength, xlabel = "Participant", ylabel = "Stride Length (cm)", title="Stride Length Per Person", xticklabels=data.Name, label="Stride Length", legend=True, color="Red")
myplot.add_plot(MPLPlot.basic_scatter, x=[0,1,2,3,4], y=np.array([data.Height, data.StrideLength]), xlabel = "Participant", ylabel = "Centimeters", title="Height/Stride Length Per Participant", xticklabels=data.Name, label=["Height", "Stride Length"], legend=True, color=["Blue", "Red"], ylim=(50,180))

# Show the figures
myplot.show()
print("bruv 💀")