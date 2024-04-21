#Import all packages
from MPLplot import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

 # Import data
data = pd.read_csv("files/csvsets/Ece30Asgn1Data.csv")

# Create MPLPlot
myplot = MPLPlot(is_multiplot=True, dim=(2,3))

# Add Plots
myplot.add_plot(MPLPlot.basic_bar, data=data, height=data.AvgTime, title="Average Time Baisc Unformated Plot")
myplot.insert_plot(MPLPlot.basic_bar, 5, data=data, height=data.AvgVelocity) # Spot 5 is free so it inserts there
myplot.add_plot(MPLPlot.basic_bar, data=data, height=data.AvgTime, label=data.Name, title="Average Effect of Stride Length (cm) on Travel Time (s)", xlabel="Participants", ylabel="Time (seconds)", xticklabels=data.Name, colors=['#ff6d01', '#abed9b', '#5546ea', '#fbbd05', '#cc0100'], width=0.5, legend=True, legendsize=6)
myplot.insert_plot(MPLPlot.basic_bar, 1, data=data, height=data.AvgVelocity, colors='yellow', xscale='log') # Spot 1 is taken so it fills to next free
myplot.insert_plot(MPLPlot.basic_bar, 1, data=data, height=data.AvgVelocity, colors='magenta', align='edge', bottom=4, ylim=(0,14)) # Spot 1 is taken so it fills to next free

# Show the figures
myplot.show()