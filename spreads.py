import pandas
import numpy 
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.api import OLS
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot

def plots(data, label):
    plot_acf(data, zero = False)
    plt.title(label + ' Original')
    plt.show()
    plot_acf(abs(data), zero = False)
    plt.title(label + ' Absolute')
    plt.show()
    qqplot(data, line = 's')
    plt.title(label)
    plt.show()

def analysis(data):
    print('Skewness, kurtosis = ', round(stats.skew(data), 3), round(stats.kurtosis(data), 3))
    
keys = ['AAA', 'BAA']
DF = pandas.read_excel('spreads.xlsx', sheet_name = 'data')
vix = DF['VIX'].values
plt.plot(DF['AAA'].values, label = 'AAA-10YTR')
plt.plot(DF['BAA'].values, label = 'BAA-10YTR')
plt.legend(loc = 'best')
plt.title('Spreads')
plt.show()
N = len(vix)
lvix = numpy.log(vix)
plt.plot(vix, label = 'VIX')
plt.legend(loc = 'best')
plt.title('Volatility')
plt.show()
VIXAR = stats.linregress(lvix[:-1], numpy.diff(lvix))
print('Autoregression VIX')
vixres = numpy.array([lvix[k+1] - lvix[k] * (VIXAR.slope + 1)- VIXAR.intercept for k in range(N-1)])

for key in keys:
    print(key)
    series = DF[key].values
    dseries = numpy.diff(series)
    reg = stats.linregress(series[:-1], dseries)
    res = numpy.array([series[k+1] - series[k] - reg.slope*series[k] - reg.intercept for k in range(N-1)])
    print('Initial')
    analysis(res)
    nres = res/vix[1:]
    print('Normalized')
    analysis(nres)
    RegDF = pandas.DataFrame({'a' : 1/vix[1:], 'b' : series[:-1]/vix[1:], 'c' : 1})
    Reg = OLS(dseries/vix[1:], RegDF).fit()
    rres = Reg.resid
    analysis(rres)
    print('Regression coefficients\n', round(Reg.params, 4))
    print('Regression p-values\n', round(Reg.pvalues, 3))
    print('Full')
    plots(rres, key)
    print('Correlation = ', round(stats.pearsonr(vixres, rres)[0], 2))