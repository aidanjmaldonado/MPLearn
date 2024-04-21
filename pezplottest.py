#Import all packages
from MPLplot import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

 # Import data
data = pd.read_csv("files/csvsets/Ece30Asgn1Data.csv")

# Create MPLPlot
myplot = MPLPlot(is_multiplot=True, dim=((2, 2)))

# Add Plots

myplot.add_plot(MPLPlot.basic_scatter, x=[data.Height], y=[data.AvgStrides], xlabel = "Participant Height (cm)", ylabel = "Average Number of Strides", title="Height vs Strides", label=["Average Strides"], legend=True, color=["Red"])
myplot.add_plot(MPLPlot.basic_scatter, x=[data.Height], y=[data.AvgTime], xlabel = "Participant Height (cm)", ylabel = "Average Time (s)", title="Height vs Time", label=["Average Time"], legend=True, color=["Blue"])
myplot.add_plot(MPLPlot.basic_scatter, x=[data.Height, data.Height], y=[data.AvgTime, data.AvgStrides], xlabel = "Participant Height (cm)", title="Both", label=["Average Time", "Average Strides"], legend=True, color=["Blue", "Red"])

# Show the figures
myplot.show()
print("bruv 💀")
