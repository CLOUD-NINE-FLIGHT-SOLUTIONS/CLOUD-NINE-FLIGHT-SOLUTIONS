import os
import pandas as pd
import seaborn as sns
import env
# import acquire
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime



# Pulls the airline data for any airline from the 10 csv's of all the flights in north america from 2009 to 2019

def pull_airline_data(airline = 'UA') -> pd.DataFrame:
    filename = f'{airline}.csv'
    
    #if this airline has already been pulled it will be saved as csv
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=[0])
    
    #Pull and Create the df
    else:
        #Create the list of columns # want to pull from the dataframe
        column_list = ['FL_DATE', 'OP_CARRIER', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']

        #Pull the abridged columns in a loop for 2009-2018 creating an empth df outside the loop
        flights = pd.DataFrame()
        for i in range(2009, 2019):
            # read each csv and pull the columns of interest
            flightsi = pd.read_csv(f'airline delay analysis/{i}.csv', usecols=column_list)
            # Mask for specific airline which in this case is United Airlines or UA
            flightsi = flightsi[flightsi['OP_CARRIER'] == airline]
            #links all the df's together vertically into one
            flights = flights.append(flightsi)

        #2019 dataset slightly different with less columns and OP_UNIQUE_CARRIER instead of OP_CARRIER
        column_list = ['FL_DATE', 'OP_UNIQUE_CARRIER', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']    
        #SAME as above except renames the column in order to append (MVP SOLUTION)
        flights2019 = pd.read_csv(f'airline delay analysis/2019.csv', usecols=column_list) 
        flights2019 = flights2019.rename(columns={'OP_UNIQUE_CARRIER':'OP_CARRIER'})
        flights2019 = flights2019[flights2019['OP_CARRIER'] == airline]
        flights = flights.append(flights2019)
        
        #create the csv
        flights.to_csv(f'{airline}.csv')
        
        #return the db
        return flights
    

#Cleans the flights dataframe and creates a daily mean dataframe for delay times imputing values for September 2009 and October 2011


def clean_flight_data_for_average_daily_delay(flights):
    #Fills in nulls as zero as null means no delay
    flights.fillna(0, inplace=True)
    #Makes FL_DATE column a datetime datatype
    flights.FL_DATE = flights.FL_DATE.astype('datetime64')
    #Makes FL_DATE the index
    flights = flights.set_index('FL_DATE')
    #creates a new column to create a total delay for each observation
    flights['total_delays'] = flights.CARRIER_DELAY + flights.WEATHER_DELAY + flights.NAS_DELAY + flights.SECURITY_DELAY + flights.LATE_AIRCRAFT_DELAY
    #Drops the now used columns
    flights = flights.drop(columns=['OP_CARRIER', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'])
    
    #impute by mean delay by averaging the "average delays" from the month before and after
    flights_monthly_mean = flights.resample('M').mean().total_delays
    impute_1 = flights_monthly_mean.loc['2009-09-30'] + flights_monthly_mean.loc['2009-11-30'] / 2
    impute_2 = flights_monthly_mean.loc['2011-06-30'] + flights_monthly_mean.loc['2011-08-31'] / 2
    
    #flights_daily_mean = flights.resample('D').mean().total_delays
    #Chose Dataframe
    flights_daily_mean = pd.DataFrame(flights.resample('D').mean().total_delays)
    
    #First imputation
    start_date = datetime.strptime("2009-10-01", "%Y-%m-%d")
    end_date = datetime.strptime("2009-10-31", "%Y-%m-%d")
    
    #Two lists to zip into tuple
    date_list = pd.date_range(start_date, end_date, freq='D')
    impute_1_list = [impute_1] * 31
    
    #Zip them and df them
    list_tuples = list(zip(date_list, impute_1_list))
    list_tuples = pd.DataFrame(list_tuples, columns=['FL_DATE', 'total_delays'])

    #datetime and index 
    list_tuples.FL_DATE = list_tuples.FL_DATE.astype('datetime64')
    list_tuples = list_tuples.set_index('FL_DATE')

    #Append to the daily mean df
    flights_daily_mean = flights_daily_mean.append(list_tuples)

    #Do the same with the second imputation
    start_date_2 = datetime.strptime("2011-07-01", "%Y-%m-%d")
    end_date_2 = datetime.strptime("2011-07-31", "%Y-%m-%d")
    date_list_2 = pd.date_range(start_date_2, end_date_2, freq='D')
    impute_2_list = [impute_2] * 31

    list_tuples_2 = list(zip(date_list_2, impute_2_list))
    list_tuples_2 = pd.DataFrame(list_tuples_2, columns=['FL_DATE', 'total_delays'])

    list_tuples_2.FL_DATE = list_tuples_2.FL_DATE.astype('datetime64')
    list_tuples_2 = list_tuples_2.set_index('FL_DATE')

    flights_daily_mean = flights_daily_mean.append(list_tuples_2)

    flights_daily_mean = flights_daily_mean[flights_daily_mean['total_delays'].notna()]

    flights_daily_mean = flights_daily_mean.rename(columns={'total_delays':'average_delay'})

    flights_daily_mean = flights_daily_mean.sort_index()
    
    return flights_daily_mean


import os
import pandas as pd
import seaborn as sns

# import acquire
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime

# Pulls the airline data for any airline from the 10 csv's of all the flights in north america from 2009 to 2019

def pull_airline_data(airline = 'AA') -> pd.DataFrame:
    filename = f'{airline}.csv'
    
    #if this airline has already been pulled it will be saved as csv
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=[0])
    
    #Pull and Create the df
    else:
        #Create the list of columns # want to pull from the dataframe
        column_list = ['FL_DATE', 'OP_CARRIER_FL_NUM', 'OP_CARRIER', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'ORIGIN', 'DEST']

        #Pull the abridged columns in a loop for 2009-2018 creating an empth df outside the loop
        flights = pd.DataFrame()
        for i in range(2009, 2019):
            # read each csv and pull the columns of interest
            flightsi = pd.read_csv(f'airline delay analysis/{i}.csv', usecols=column_list)
            # Mask for specific airline which in this case is United Airlines or UA
            flightsi = flightsi[flightsi['OP_CARRIER'] == airline]
            #links all the df's together vertically into one
            flights = flights.append(flightsi)

        #2019 dataset slightly different with less columns and OP_UNIQUE_CARRIER instead of OP_CARRIER
        column_list = ['FL_DATE', 'OP_CARRIER_FL_NUM', 'OP_UNIQUE_CARRIER', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'ORIGIN', 'DEST'] 
        
        #SAME as above except renames the column in order to append (MVP SOLUTION)
        flights2019 = pd.read_csv(f'airline delay analysis/2019.csv', usecols=column_list) 
        flights2019 = flights2019.rename(columns={'OP_UNIQUE_CARRIER':'OP_CARRIER'})
        flights2019 = flights2019[flights2019['OP_CARRIER'] == airline]
        flights = flights.append(flights2019)
        
        #Fills in nulls as zero as null means no delay        
        flights.fillna(0, inplace=True)
        
        #create the csv
        flights.to_csv(f'{airline}.csv')
        
        # List of the top 15 Class B airports    
        top_15_hubs = ['ATL',
                        'DFW',
                        'DEN',
                        'ORD',
                        'LAX',
                        'JFK',
                        'IAH',
                        'PHX',
                        'EWR',
                        'SFO',
                        'SEA',
                        'IAD',
                        'PHL',
                        'CLT',
                        'MIA']
        
        # Filtering the rows with the top 15 airports 
        flights = flights[flights['ORIGIN'].isin(top_15_hubs)]
        
        
        # Suming the rows delay times
        flights['row_sums'] = flights[['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']].sum(axis=1)

        
        # Deleting rows that have no delays
        flights = flights[flights['row_sums']>0]
        
        # Reseting the index
        flights.reset_index(inplace=True, drop=True)

        # Return the db
        return flights
    

#Cleans the flights dataframe and creates a daily mean dataframe for delay times imputing values for September 2009 and October 2011


def clean_flight_data_for_average_daily_delay(flights):


    flights.FL_DATE = flights.FL_DATE.astype('datetime64')
    #Makes FL_DATE the index
    flights = flights.set_index('FL_DATE')
    #creates a new column to create a total delay for each observation
    flights['total_delays'] = flights.CARRIER_DELAY + flights.WEATHER_DELAY + flights.NAS_DELAY + flights.SECURITY_DELAY + flights.LATE_AIRCRAFT_DELAY
    #Drops the now used columns
    flights = flights.drop(columns=['OP_CARRIER', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'])

    #impute by mean delay by averaging the "average delays" from the month before and after
    flights_monthly_mean = flights.resample('M').mean().total_delays
    impute_1 = flights_monthly_mean.loc['2009-09-30'] + flights_monthly_mean.loc['2009-11-30'] / 2
    impute_2 = flights_monthly_mean.loc['2011-06-30'] + flights_monthly_mean.loc['2011-08-31'] / 2
    
    #flights_daily_mean = flights.resample('D').mean().total_delays
    #Chose Dataframe
    flights_daily_mean = pd.DataFrame(flights.resample('D').mean().total_delays)
    
    #First imputation
    start_date = datetime.strptime("2009-10-01", "%Y-%m-%d")
    end_date = datetime.strptime("2009-10-31", "%Y-%m-%d")
    
    #Two lists to zip into tuple
    date_list = pd.date_range(start_date, end_date, freq='D')
    impute_1_list = [impute_1] * 31
    
    #Zip them and df them
    list_tuples = list(zip(date_list, impute_1_list))
    list_tuples = pd.DataFrame(list_tuples, columns=['FL_DATE', 'total_delays'])

    #datetime and index 
    list_tuples.FL_DATE = list_tuples.FL_DATE.astype('datetime64')
    list_tuples = list_tuples.set_index('FL_DATE')

    #Append to the daily mean df
    flights_daily_mean = flights_daily_mean.append(list_tuples)

    #Do the same with the second imputation
    start_date_2 = datetime.strptime("2011-07-01", "%Y-%m-%d")
    end_date_2 = datetime.strptime("2011-07-31", "%Y-%m-%d")
    date_list_2 = pd.date_range(start_date_2, end_date_2, freq='D')
    impute_2_list = [impute_2] * 31

    list_tuples_2 = list(zip(date_list_2, impute_2_list))
    list_tuples_2 = pd.DataFrame(list_tuples_2, columns=['FL_DATE', 'total_delays'])

    list_tuples_2.FL_DATE = list_tuples_2.FL_DATE.astype('datetime64')
    list_tuples_2 = list_tuples_2.set_index('FL_DATE')

    flights_daily_mean = flights_daily_mean.append(list_tuples_2)

    flights_daily_mean = flights_daily_mean[flights_daily_mean['total_delays'].notna()]

    flights_daily_mean = flights_daily_mean.rename(columns={'total_delays':'average_delay'})
    

    # finding only the rows with sum 0
    flights_daily_mean = flights_daily_mean[flights_daily_mean['average_delay']>0]

    flights_daily_mean = flights_daily_mean.sort_index()
    
    return flights_daily_mean
