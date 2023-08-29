import os
import pandas as pd
import seaborn as sns
import env
import wrangle
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime
import numpy as np

import matplotlib.pyplot as plt
import statsmodels.api as sm

#Plots the resampled averages by the ingervals in the dictionary as well as its corresponding autocorrelation

def plot_average_by_interval(flights_daily_mean):
    
    resample_dict = {'D':'Daily', 'W':'Weekly', '2W':'FORTNIGHTLY', 'M':'Monthly', '3M':'Quarterly', '6M':'Semi-Annually', 'Y':'Yearly'}

    for _ in resample_dict:
        if _ == '2W':
            plt.figure(figsize=(10,6))

            flights_daily_mean.resample(_).mean().average_delay.plot(label='daily', color = '#4daf4a')

            plt.title(f'{resample_dict[_][0:]} Average Delay')
            plt.legend()
            plt.show()

            pd.plotting.autocorrelation_plot(flights_daily_mean.average_delay.resample(_).mean(), color = '#4daf4a')
            plt.show()
        else:
            plt.figure(figsize=(10,6))

            flights_daily_mean.resample(_).mean().average_delay.plot(label='daily')

            plt.title(f'{resample_dict[_][0:]} Average Delay')
            plt.legend()
            plt.show()

            pd.plotting.autocorrelation_plot(flights_daily_mean.average_delay.resample(_).mean())
            plt.show()
#Just Shows the best correlated lag plot
def plot_best_lag_plot(flights_daily_mean):

    #Plots the best correlated lag
    pd.plotting.lag_plot(flights_daily_mean.resample('W').mean(), lag=365)
    plt.title('lag plot');

#Show the seasonal_decomposition_plot by the duration put in
def seasonal_decomposition_plot(df, duration):
    y = df.average_delay.resample(duration).mean()

    result = sm.tsa.seasonal_decompose(y)

    decomposition = pd.DataFrame({
        'y': result.observed,
        'trend': result.trend,
        'seasonal': result.seasonal,
        'resid': result.resid
    })

    decomposition['trend_centered'] = decomposition.trend - decomposition.trend.mean()
    decomposition[['trend_centered', 'seasonal', 'resid']].plot();