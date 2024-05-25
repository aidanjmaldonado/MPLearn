## Import all packages
from MPLplot import *

## Import Data
data = pd.read_csv("Ece30Module4Data.csv")

## Create MPLPlot
myplot = MPLPlot(is_multiplot=True, dim=((2, 1)), toolbar=True)

myplot.add_plot(plot_type=MPLPlot.basic_bar,
                height=data.Celsius, 
                xlabel="Proportion of Hot Water / Cold Water",
                ylabel="Water Temperature °C", 
                title="Effect of Proportion of Hot Water / Cold Water on Water Temperature°C", 
                xticklabels=["0", "1/3", "1/2", "2/3", "1"],
                colors="salmon"
)

myplot.add_plot(plot_type=MPLPlot.basic_plot,
                x=data.Percent_Hot,
                y=data.Celsius, 
                xlabel="Proportion of Hot Water / Cold Water",
                ylabel="Water Temperature °C", 
                title="Effect of Proportion of Hot Water / Cold Water on Water Temperature°C", 
                xticklabels=["0", "1/3", "1/2", "2/3", "1"],
                dots="-o",
                markersize=5,
                color="#961127"
)




## Show Plot
plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
myplot.show()