from MPLplot import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import Data
data1 = pd.read_csv("files/ShakeAndJumpTest.csv")
data2 = pd.read_csv("files/ShakeAndJumpTest2.csv")
data3 = pd.read_csv("files/0-50cm.csv")
data4 = pd.read_csv("files/1-00cm.csv")
data5 = pd.read_csv("files/2-30cm.csv")
data6 = pd.read_csv("files/2-75cm.csv")
data7 = pd.read_csv("files/ShakeAndJumpTest2.csv")

dataa = pd.read_csv("files/0-25cm.csv")
datab = pd.read_csv("files/0-50cm.csv")
datac = pd.read_csv("files/0-75cm.csv")
datad = pd.read_csv("files/1-00cm.csv")
datae = pd.read_csv("files/1-25cm.csv")
dataf = pd.read_csv("files/1-50cm.csv")
datag = pd.read_csv("files/1-75cm.csv")
datah = pd.read_csv("files/2-00cm.csv")
datai = pd.read_csv("files/2-25cm.csv")
dataj = pd.read_csv("files/2-30cm.csv")
datak = pd.read_csv("files/2-50cm.csv")
datal = pd.read_csv("files/2-75cm.csv")





# Create MPLPlot
# ShakeJump = MPLPlot(is_multiplot=False, dim=((1, 2)), toolbar=True)
PhoneDrop = MPLPlot(is_multiplot=True, dim=(2,2))
PeakChart = MPLPlot(is_multiplot=True, dim=(2,2))

MaxAccel = MPLPlot(is_multiplot=False, dim=(2,1))
Velocity = MPLPlot(is_multiplot=False)

# Plot Shake Jump
# ShakeJump.add_plot(plot_type=MPLPlot.basic_plot, x=data1.time, y=np.array([data1.ax, data1.ay, data1.az, data1.atotal]), color=["black", "orange", "green", "red"], title="Acceleration vs Time", xlabel="Time (s)", ylabel="Acceleration (m/s^2)", label=['Acceleration X-axis','Acceleration Y-Axis','Acceleration Z-Axis','Acceleration Total'], legend=True, linewidth=1)
# ShakeJump.add_plot(plot_type=MPLPlot.basic_plot, x=data2.time, y=np.array([data2.ax, data2.ay, data2.az, data2.atotal]), color=["black", "orange", "green", "red"], title="Acceleration vs Time", xlabel="Time (s)", ylabel="Acceleration (m/s^2)", label=['Acceleration X-Axis','Acceleration Y-Axis','Acceleration Z-Axis','Acceleration Total'], legend=True, linewidth=1)

# plot Phone Drop
PhoneDrop.add_plot(plot_type=MPLPlot.basic_plot,
                   x=data3.time,
                   y=np.array([data3.ax, data3.ay, data3.az, data3.atotal]),
                   xlabel = "Time (s)",
                   ylabel = "Acceleration (m/s^2)",
                   title = "Acceleration vs Time (0.5m Drop)",
                   label=['Acceleration X-Axis','Acceleration Y-Axis','Acceleration Z-Axis','Acceleration Total'],
                   color=["blue", "orange", "red", "green"],
                   linewidth=1,
                   legend=True)

PhoneDrop.add_plot(plot_type=MPLPlot.basic_plot,
                   x=data4.time,
                   y=np.array([data4.ax, data4.ay, data4.az, data4.atotal]),
                   xlabel = "Time (s)",
                   ylabel = "Acceleration (m/s^2)",
                   title = "Acceleration vs Time (1m Drop)",
                   label=['Acceleration X-Axis','Acceleration Y-Axis','Acceleration Z-Axis','Acceleration Total'],
                   color=["blue", "orange", "red", "green"],
                   linewidth=1,
                   legend=True)

PhoneDrop.add_plot(plot_type=MPLPlot.basic_plot,
                   x=data5.time,
                   y=np.array([data5.ax, data5.ay, data5.az, data5.atotal]),
                   xlabel = "Time (s)",
                   ylabel = "Acceleration (m/s^2)",
                   title = "Acceleration vs Time (2.27m Drop - Full Reach)",
                   label=['Acceleration X-Axis','Acceleration Y-Axis','Acceleration Z-Axis','Acceleration Total'],
                   color=["blue", "orange", "red", "green"],
                   linewidth=1,
                   legend=True)

PhoneDrop.add_plot(plot_type=MPLPlot.basic_plot,
                   x=data6.time,
                   y=np.array([data6.ax, data6.ay, data6.az, data6.atotal]),
                   xlabel = "Time (s)",
                   ylabel = "Acceleration (m/s^2)",
                   title = "Acceleration vs Time (2.76m Drop - With Chair)",
                   label=['Acceleration X-Axis','Acceleration Y-Axis','Acceleration Z-Axis','Acceleration Total'],
                   color=["blue", "orange", "red", "green"],
                   linewidth=1,
                   legend=True)

# print(max(data4.atotal))
# PeakChart.add_plot(plot_type=MPLPlot.basic_plot, x=[0.5, 1.0, 2.27, 2.76], y=[max(data3.ax), max(data4.ax), max(data5.ax), max(data6.ax)], xlabel="Drop Height (m)", ylabel="Peak Acceleration (m/s^2)", title="Initial Drop Hieght vs Maximum Acceleration", label="X-Axis Acceleration", legend=True, color="Red")
# PeakChart.add_plot(plot_type=MPLPlot.basic_plot, x=[0.5, 1.0, 2.27, 2.76], y=[max(data3.ay), max(data4.ay), max(data5.ay), max(data6.ay)], xlabel="Drop Height (m)", ylabel="Peak Acceleration (m/s^2)", title="Initial Drop Hieght vs Maximum Acceleration", label="Y-Axis Acceleration", legend=True, color="Blue")
# PeakChart.add_plot(plot_type=MPLPlot.basic_plot, x=[0.5, 1.0, 2.27, 2.76], y=[max(data3.az), max(data4.az), max(data5.az), max(data6.az)], xlabel="Drop Height (m)", ylabel="Peak Acceleration (m/s^2)", title="Initial Drop Hieght vs Maximum Acceleration", label="Z-Axis Acceleration", legend=True, color="Green")
# PeakChart.add_plot(plot_type=MPLPlot.basic_plot, x=[0.5, 1.0, 2.27, 2.76], y=[max(data3.atotal), max(data4.atotal), max(data5.atotal), max(data6.atotal)], xlabel="Drop Height (m)", ylabel="Peak Acceleration (m/s^2)", title="Initial Drop Hieght vs Maximum Acceleration", label="Total Acceleration", legend=True, color="Black")

MaxAccel.add_plot(plot_type=MPLPlot.basic_plot, x=[0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.3, 2.5, 2.75], y=[max(dataa.ax), max(datab.ax), max(datac.ax), max(datad.ax), max(datae.ax), max(dataf.ax), max(datag.ax), max(datah.ax), max(datai.ax), max(dataj.ax), max(datak.ax), max(datal.ax)], xlabel="Drop Height (m)", ylabel="Peak Acceleration (m/s^2)", title="Initial Drop Hieght vs Maximum Acceleration", label="x Acceleration", legend=True, color="Red", linewidth=0.5)
MaxAccel.insert_plot(plot_type=MPLPlot.basic_plot, pindex=1, x=[0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.3, 2.5, 2.75], y=[max(dataa.ay), max(datab.ay), max(datac.ay), max(datad.ay), max(datae.ay), max(dataf.ay), max(datag.ay), max(datah.ay), max(datai.ay), max(dataj.ay), max(datak.ay), max(datal.ay)], xlabel="Drop Height (m)", ylabel="Peak Acceleration (m/s^2)", title="Initial Drop Hieght vs Maximum Acceleration", label="y Acceleration", legend=True, color="Blue", linewidth=0.5, force=(1))
MaxAccel.insert_plot(plot_type=MPLPlot.basic_plot, pindex=1, x=[0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.3, 2.5, 2.75], y=[max(dataa.az), max(datab.az), max(datac.az), max(datad.az), max(datae.az), max(dataf.az), max(datag.az), max(datah.az), max(datai.az), max(dataj.az), max(datak.az), max(datal.az)], xlabel="Drop Height (m)", ylabel="Peak Acceleration (m/s^2)", title="Initial Drop Hieght vs Maximum Acceleration", label="z Acceleration", legend=True, color="Green", linewidth=1.5, force=(1), dots="-o", markersize=4)
MaxAccel.insert_plot(plot_type=MPLPlot.basic_plot, pindex=1, x=[0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.3, 2.5, 2.75], y=[max(dataa.atotal), max(datab.atotal), max(datac.atotal), max(datad.atotal), max(datae.atotal), max(dataf.atotal), max(datag.atotal), max(datah.atotal), max(datai.atotal), max(dataj.atotal), max(datak.atotal), max(datal.atotal)], xlabel="Drop Height (m)", ylabel="Peak Acceleration (m/s^2)", title="Initial Drop Height vs Maximum Acceleration", label="total Acceleration", legend=True, color="Black", xlim=(0,2.5), ylim=(0,500), linewidth=2, force=(1), dots="-o", markersize=6)

loglinex = np.linspace(0, 10, 100)
logliney = np.array([500*(np.sqrt(2 * loglinex / 9.8)) for x in range(100)])
# print(logliney, logliney.shape)
# x = np.linspace(0.1, 10, 100)
# y = np.array([np.log(x) for x in range(100)])
# y=np.log(x)

# Velocity.add_plot(plot_type=MPLPlot.basic_plot, x=loglinex, y=logliney, xlim=(0,  3), ylim=(0, 500))

Height = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.3, 2.5, 2.75])
Terminal = np.array([np.sqrt(2 * Height / 9.8) for x in range(12)])


MaxAccel.insert_plot(plot_type=MPLPlot.basic_plot, pindex=1, x=Height, y=Terminal, force=1, xlim=(0.25,2.75), color='Orange')


# MaxAccel.insert_plot(plot_type=MPLPlot.basic_plot, pindex=1, x=loglinex, y=logliney, force=1, label="papi", legend=True)

# MaxAccel.add_plot(plot_type=MPLPlot.basic_plot, x=[0.5, 1, 2.27, 2.76], y=[max(data3.az), max(data4.az), max(data5.az), max(data6.az)], xlabel="Drop Height (m)", ylabel="Peak Acceleration (m/s^2)", title="Initial Drop Hieght vs Maximum Acceleration", label="z Acceleration", legend=True, color="Red", prev=True)


# Show Plot
plt.show()