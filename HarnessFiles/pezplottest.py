## Import all packages
from MPLplot import *

## Import data
data = pd.read_csv("HarnessFiles/Ece30Asgn1Data.csv")
data2 = pd.read_csv("HarnessFiles/Ece30Asgn1DataExtended.csv")


## Create MPLPlot
myplot = MPLPlot(is_multiplot=True, dim=((3, 3)), toolbar=False)

## Add Plots

# Bar Plots - *For Data*
myplot.add_plot(MPLPlot.basic_bar, height=data.AvgTime, label=data.Name, title="Average Effect of Stride Length (cm) on Travel Time (s)", xlabel="Participants", ylabel="Time (seconds)", xticklabels=data.Name, colors=['#ff6d01', '#abed9b', '#5546ea', '#fbbd05', '#cc0100'], width=0.5, legend=True, legendsize=6)
myplot.add_plot(MPLPlot.basic_bar, height=data.AvgStrides, width=0.5, colors='limegreen')
myplot.add_plot(MPLPlot.basic_bar, height=np.array([data.AvgStrides, data.AvgVelocity]), width=0.1, colors=["purple", "salmon"], xticklabels = ["Average Strides", "Average Velocity"])

# # Standard Plots
# myplot.add_plot(MPLPlot.basic_plot, x=[i for i in range(data.shape[0])], y=data.Height, xlabel = "Participant", ylabel = "Height (cm)", title="Height Per Participant", xticklabels=data.Name, label="Height", legend=True, color="Blue")
# myplot.add_plot(MPLPlot.basic_plot, x=[i for i in range(data.shape[0])], y=data.StrideLength, xlabel = "Participant", ylabel = "Stride Length (cm)", title="Stride Length Per Person", xticklabels=data.Name, label="Stride Length", legend=True, color="Red")
# myplot.add_plot(MPLPlot.basic_plot, x=[i for i in range(data.shape[0])], y=np.array([data.Height, data.StrideLength]), xlabel = "Participant", ylabel = "Centimeters", title="Height/Stride Length Per Participant", xticklabels=data.Name, label=["Height", "Stride Length"], legend=True, color=["Blue", "Red"], ylim=(50,180))

# # Scatter Plots
# myplot.add_plot(MPLPlot.basic_scatter, x=[i for i in range(data.shape[0])], y=data.Height, xlabel = "Participant", ylabel = "Height (cm)", title="Height Per Participant", xticklabels=data.Name, label="Height", legend=True, color="Blue")
# myplot.add_plot(MPLPlot.basic_scatter, x=[i for i in range(data.shape[0])], y=data.StrideLength, xlabel = "Participant", ylabel = "Stride Length (cm)", title="Stride Length Per Person", xticklabels=data.Name, label="Stride Length", legend=True, color="Red")
# myplot.add_plot(MPLPlot.basic_scatter, x=[i for i in range(data.shape[0])], y=np.array([data.Height, data.StrideLength]), xlabel = "Participant", ylabel = "Centimeters", title="Height/Stride Length Per Participant", xticklabels=data.Name, label=["Height", "Stride Length"], legend=True, color=["Blue", "Red"], ylim=(50,180))



# ## Create MPLPlot 2
# # Bar Plots - *For Extended Data*
# myextendedplot = MPLPlot(is_multiplot=True, dim=((3, 3)), toolbar=False)

# myextendedplot.add_plot(MPLPlot.basic_bar, height=data2.AvgTime, label=data2.Name, title="Average Effect of Stride Length (cm) on Travel Time (s)", xlabel="Participants", ylabel="Time (seconds)", colors=['#ff6d01', '#abed9b', '#5546ea', '#fbbd05', '#cc0100'], width=0.5, legend=True, legendsize=6)
# myextendedplot.add_plot(MPLPlot.basic_bar, height=data2.AvgStrides, width=0.5, colors='limegreen')
# myextendedplot.add_plot(MPLPlot.basic_bar, height=np.array([data2.AvgStrides, data2.AvgVelocity]), width=0.1, colors=["purple", "salmon"], xticklabels = ["Average Strides", "Average Velocity"])

# # Standard Plots
# myextendedplot.add_plot(MPLPlot.basic_plot, x=[i for i in range(data2.shape[0])], y=data2.Height, xlabel = "Participant", ylabel = "Height (cm)", title="Height Per Participant", label="Height", legend=True, color="Blue")
# myextendedplot.add_plot(MPLPlot.basic_plot, x=[i for i in range(data2.shape[0])], y=data2.StrideLength, xlabel = "Participant", ylabel = "Stride Length (cm)", title="Stride Length Per Person", xticklabels=data2.Name, label="Stride Length", legend=True, color="Red")
# myextendedplot.add_plot(MPLPlot.basic_plot, x=[i for i in range(data2.shape[0])], y=np.array([data2.Height, data2.StrideLength]), xlabel = "Participant", ylabel = "Centimeters", title="Height/Stride Length Per Participant", xticklabels=data2.Name, label=["Height", "Stride Length"], legend=True, color=["Blue", "Red"], ylim=(50,180))

# # Scatter Plots
# myextendedplot.add_plot(MPLPlot.basic_scatter, x=[i for i in range(data2.shape[0])], y=data2.Height, xlabel = "Participant", ylabel = "Height (cm)", title="Height Per Participant", label="Height", legend=True, color="Blue")
# myextendedplot.add_plot(MPLPlot.basic_scatter, x=[i for i in range(data2.shape[0])], y=data2.StrideLength, xlabel = "Participant", ylabel = "Stride Length (cm)", title="Stride Length Per Person", label="Stride Length", legend=True, color="Red")
# myextendedplot.add_plot(MPLPlot.basic_scatter, x=[i for i in range(data2.shape[0])], y=np.array([data2.Height, data2.StrideLength]), xlabel = "Participant", ylabel = "Centimeters", title="Height/Stride Length Per Participant", label=["Height", "Stride Length"], legend=True, color=["Blue", "Red"], ylim=(50,180), s=10)



# Show the figures
# myplot.show()
# myextendedplot.show()
# MPLPlot.show()
print("bruv 💀")

plt.show()

# plot( x=np.array([df.time for x in cols]) )