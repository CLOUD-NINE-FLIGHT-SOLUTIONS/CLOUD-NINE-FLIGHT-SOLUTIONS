# <a name="top"></a>CLOUD-NINE-FLIGHT-SOLUTIONS
![]()

by: Alfred W. Pirovits
    Emanuel Villa
    Miatta Sinayoko

<p>
  <a href="https://github.com/Alfred-W-S-Pirovits-Jr/telco_churn_project#top" target="_blank">
    <img alt="" src="" />
  </a>
</p>


***
[[Executive Summary](#executive_summary)]
[[Project Goals](#project_goals)]
[[Project Description](#project_description)]
[[Project Planning](#planning)]
[[Initial Questions](#initial_questions)]
[[Hypothesis](#hypothesis)]
[[Total Delays resampled to Average Delay](#target)]
[[Key Nice to haves (With more time)](#nice_to_haves)]
[[Key Findings, Recommendations and Takeaways](#findings)]
[[Data Dictionary](#dictionary)]
[[Data Acquire and Prep](#wrangle)]
[[Exploration](#explore)]
[[Conclusion](#conclusion)]
[[Steps to Reproduce](#steps_to_reproduce)]
___

![Cloud_9_Logo](Cloud_9_Logo%20(1).png)

## <a name="exective_summary"></a> Executive Summary:
This project includes data pulled from https://www.kaggle.com/datasets/sherrytp/airline-delay-analysis covering 10 years of flights from 2009-2019.  The data are held in 10 csv's (one for each year).Major airlines United Airlines,American Airlines, Delta and South West in addition to the top 15 Class B airpot hubs are obsrved. The project breaks down total delays for each flight and extracts an average delay over two week intervals from which a time series model is constructed to accurately characterize seasonal variation in the data with regard to delays.  

***

## <a name="project_goals"></a> Project Goals:

The goal is to develop a Machine Learning model that can accuratly decode a decade of flight data to predict and manage airline delays. By harnessing the power of seasonal trends and incorporating comprehensive flight data, we empower airlines to optimize operations,minimize costs, enhance passenger experience, and soar above the competition. Data-driven insights pave the way for stakeholders to make executive decisions based off of actionable analysis.

***

## <a name="project_description"></a>Project Description:
[[Back to top](#top)]
The purpose of this project is to look at all of the massive amounts of data and see if we can garner greneral trends that may prove useful to the mahor airline carrier stakeholders. We suspected that there is a yearly pattern that holds and dictates delays given the four seasons in a year but it would be nice to show that there is a repeatable trend. Additionally, we are wondering if these results will be different by major airlines as they often own different hubs in the transportation network.  Different airports have diffenent airlines operating out of them as main hubs.

***

## <a name="planning"></a>Project Planning:    
[[Back to top](#top)]
The main goal of the project is to explore the data presented and see what we can discover.  Since there is a verbouse aount of data ton observe we want to cut it down into a manageable set of features that I could use to characterize delays.  We are relying on team member and Jr. Data Scientist Alfred W. S. Pirovits Jr. domain knowledge as a holder of a Commercial Pilot's License to choose initial features as appropriate.  After doing this, we selected reputable airlines such as United Airlines,American Airlines, Delta and South West and focused on on Top 15  out of 37 Class B hubs. 
***

## <a name="initial_questions"></a>Initial_questions:
[[Back to top](#top)]
- Do flight delays exhibit a predictable seasonal pattern?
- Is the seasonal variation in delays different for major airlines given their distinct hubs?  
- Is the total delay a viable aggregation of the individual delay types?   
- Does the expected delay based on seasonal trends align with actual delay data?
   


***


## <a name="hypothesis"></a>Hypothesis:
[[Back to top](#top)]
The main hypothesis is that there would be a meaningful seasonal perodicity to the data as the seasons correlate to delays year over year-and that periodicity can be used to build a model that can anticipate delays in the National Airspace System.  I was also suspecting the opposite of what I observed.  I thought that there would be a higher average delay in the winter as opposed to the summer but that is the opposite of what I found.  

Perhaps the airlines have taken weather delays into account in the winter time but have not done the same in the summer.  Or perhaps holts seasonal trend was able to extract more meaningful delays from the summer time but the winter data proved more residual heavy and attributed less of the delays to a seasoal pattern.  This might make sense given the fact that weather phenomena cross the us in timeframes less than two weeks, especially during the winter when systems cross the us much faster than they do in the summer.  More analysis is necessary to really hash out the details here.

***

## <a name="target"></a>Total Delays resampled to Average Delay:
[[Back to top](#top)]
The Target Variable is Total minutes delayed coded as average_delay in the resampled dataframe.  I totaled up the 5 types of delays into one column.  These were Carrier Delay, Weather Delay, NAS Delay (National Airspace System), Security Delay and Late Aircraft Delay which were combined into a total delay column.  Then the dataframe was resampled to mean delay by day and further analyzed.  In the end I did a time series analysis on the delay averaged over two week periods.   

***

## <a name="nice_to_haves"></a>Key Nice to haves (With more time):
[[Back to top](#top)]
With more time I might further divide this models by a given hub.  At this point the notebook can be run for any airline with continuous data.  This will include all the major airlines like American, Southwest, Delta, United and others.  I would like to see if further breaking down airline trends by hub would be useful.  It may or may not as usually airlines dedicate a tail number to a 3 leg trip and it remains consistent. 

Also, I would like to explore if I could use the Holt's Seasonal Trend model as a baseline from which I can and hone in on the errors with more information.  For instance if the Seasonal model shows an average delay of 10 minutes but it was 45, what accounts for the difference.  My inclination is that weather is the biggest factor.  However, even though year over year trends in weather are somewhat predictable, the day to day variation is not.  Christmas one year might be a blizzard in New York, but the next might be clear skies.  Yet it usually snows sometime in the winter.  I would probably feature engineer the difference between the average_delay predicted by this model and the actual delay for each flight and find features that may have an impact and probably run a linear regression on it.  Then I would combine the two to create a better prediction.

I also would like to see if I can get the METAR reports for those 10 years.  Every airport disseminates weather conditions to pilots once an hour.  This data has to be held somewhere.  I would like to see if I could find and append this data to the already large dataset to see if that helps.  That data would be linked by departure and arrival airports and times of departure and arrival.




***

## <a name="findings"></a>Key Findings, Recommendations and Takeaways:
[[Back to top](#top)]
The key finding is that with an RMSE of 3.74 minutes which beat the best baseline of .55 minutes, there isn't much value in the actual model by the fortnight to predict actual delays.  However the seasonal trend is absolutely valuable.  It formalizes what the airlines already know.  That there is a seasonality to the delays.  But interestingly enough, the biggest average delay peaks are happening in the summertime versus the wintertime which is the exact opposite of my expectations.  Perhaps the airlines have accounted for the winter delays but have not sufficiently accounted for the summer delays.  

I recommend putting the seasonal trend data against known delays and see if there is an agreement or an adjustment that needs to be made.  I also note that this is an MVP and though almost useless as a practical model, it may still prove to turn into something promising.  I made the error of combining the delays at the beginning of this project due to time constraints.  If I were to do it again I would do multiple models...one for each target and see if this comes up with more promising results.  

As a takeaway, I learned a bit about the time series analysis and think that I might use more models like FP prophet to get a better handle of this data.  Also since Holt's Seasonal trend with dampening worked best on test and validate, I wonder if its better to use the non dampened data as a basis for the ensemble method that I suggested above.  

***

## <a name="dictionary"></a>Data Dictionary  
[[Back to top](#top)]
FL_DATE OP_CARRIER CARRIER_DELAY WEATHER_DELAY NAS_DELAY SECURITY_DELAY LATE_AIRCRAFT_DELAY
### Data Used
---

| Attribute            | Definition                                                                                     | Data Type       |
|----------------------|------------------------------------------------------------------------------------------------|-----------------|
| FL_DATE              | The date of the flight in question                                                             | datetime64[ns]  |
| OP_CARRIER           | Two letter IATA carrier code for the airline in question                                       | object          |
| OP_CARRIER_FL_NUM    | Flight number associated with the airline                                                      | int64           |
| ORIGIN               | Origin airport for the flight                                                                  | object          |
| DEST                 | Destination airport for the flight                                                             | object          |
| CARRIER_DELAY        | Delay caused by the carrier in minutes                                                         | float64         |
| WEATHER_DELAY        | Delay caused due to weather conditions                                                         | float64         |
| NAS_DELAY            | NAS (national airspace system) delay caused in minutes                                         | float64         |
| SECURITY_DELAY       | Delay caused by security problems in minutes                                                   | float64         |
| LATE_AIRCRAFT_DELAY  | Delay caused by aircraft coming in late from a previous flight in minutes                      | float64         |
| row_sums             | Sum of row values)                                                                             | float64         |
| total_delay          | The total of all the delay columns created during feature engineering in minutes               | float64         |
| average_delay        | Average of the total delay column resampled by duration in minutes                             | float64         |


***


## <a name="explore"></a>Data Exploration:
[[Back to top](#top)]
- Python files used for exploration:
    - wrangle.ipynb
    - wrangle.py
    - explore.py
    - explore.ipynb
    - model.py
    - model.ipynb
    - final_notebook.ipynb

The steps to look through the MVP are in the final notebook.  There are a lot of functions in the preliminary  exploration, the acquire and the prepare files that one can use to explore further but for the purposes of reproducing this mvp all that is needed is in the wrangle.py file and the project_final.ipynb.
### Takeaways from exploration:

The columns are independent and so we can combine them into a total delay colum 
***

## <a name="conclusion"></a>Conclusion:
[[Back to top](#top)]
This is a long way away from a viable real world product.  However it is a very good start.  More work is needed to seperate the causes of delays and group them in a more meaningful way.  This has been an excellent project to learn time series models on.  I would want to create an ensemble model in a later iteration of this project.  All in all

- There is a clear seasonality to the delays
- Our rmse is just over half a minute better than basline but really is not all that useful in and of itself over millions of flights
- The seasonal trend IS useful and can inform expected delays given the time of the year
- There are a lot of residuals...more on that below in next steps
- Even the best models couldn't predict COVID!!!

***

## <a name="Next Steps"></a>Next Steps:
[[Back to top](#top)]
A general game plan for next steps is as follows:
-  See if I can get the historical METAR data and append it to the original dataset by destination and arrival information for every flight
-  Then I would see if I can do a classification based on weather and other columns that are in the dataset focusing on high recall for delays...specifically weather delays.
-  I would then create a new column that interpolates average delays from this model at a daily resolution.
-  From there I would construct a linear regression based on features to be discovered that can accurately predict the deviation from the expected delays based on time of year.  This should catch at least a portion of the residuals that are observed here.
-  I would then construct a seperate model for Carrier Delays and Late Aircraft Delays in order to see if there is anything meaningful there. 

***

## <a name="steps_to_reproduce"></a>Steps to Reproduce:
- Pick an airline and go.
