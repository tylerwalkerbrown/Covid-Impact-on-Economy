# Why did Bitcoin jump during a time of uncertainty ?

This project will be focusing on the impact of free money handouts (stimulus checks) and how the market jumped soon after. My current prediction is that stimulus played a key impact in Bitcoins short run of success and our current economic failures under new power. I will also be looking at the impact on the housing market and the suspected negative correlation between inflation and median home prices. In this analysis I will include:
- EDA
- Time Series Analysis 
- Forcasting 
- Brainstorming a predictive model (havent decided on varible selection/approach

Some of the skills I will be displaying include:
- Data Manipulation 
- Scraping 
- Data Collection
- Hypothesis testing 
- Statisical Modeling ?
- Pandas 
- Matplotlib
- Rolling average plot
- Cleansing

Data Sources:
- https://www.pandemicoversight.gov/data-interactive-tools/data-stories/three-rounds-stimulus-checks-see-how-many-went-out-and-how-much
- https://www.usinflationcalculator.com/inflation/current-inflation-rates/
- https://www.kaggle.com/datasets/robikscube/zillow-home-value-index
- https://data.bls.gov/timeseries/LNS14000000

Data I want for christmas:
- Remote worker month to month changes 
- Stimulus check spending habits 

# Required Packages


```python
# Modeling and Forecasting
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster
```


```python
from bs4 import BeautifulSoup  
import pandas as pd
from urllib.request import urlopen
from os import path
#Visuals
import matplotlib as plt
import matplotlib
from matplotlib.pylab import rcParams
#Stock Portion 
import pandas_datareader.data as web
import pandas as pd
import datetime
from textblob import TextBlob
import re
import numpy as np
import os
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from datetime import date
import requests
import seaborn as sns

#Changing the Figure size
rcParams['figure.figsize'] = 16,6
```

# Unemployment Data Collection/EDA


```python
#Reading in data from bls.gov
unemployment = pd.read_html('https://data.bls.gov/timeseries/LNS14000000')
```


```python
#Indexing proper table for analysis
unemployment=unemployment[1]
```


```python
#Cleaning and reformatting data
unemployment = unemployment.melt(id_vars=["Year"], 
        var_name="Jan")
#Year has to be changed from float to str
unemployment.Year = unemployment.Year.astype(str)
#Concating rows to get a general date column 
unemployment['Date'] = unemployment.Year +' ' + unemployment.Jan
#Changing concatination to datetime
unemployment['Date'] = unemployment['Date'].astype('datetime64')
#Sorting by ascending for time series 
unemployment = unemployment.sort_values(by='Date').reset_index(drop=True)
#Setting index to the date column that is created
unemployment = unemployment.set_index(['Date'])
```


```python
#Plotting inflation rates with monthly rolling average
unemployment['monthly_rolling'] = unemployment['value'].rolling(10).mean()
unemployment['monthly_rolling'].plot(label = 'monthly rolling')
unemployment['value'].plot(figsize = (15,7))
plt.xlabel('Time', fontsize=30)
plt.ylabel('Unemployment Rate', fontsize=30)
plt.xticks(ha='right', rotation=55, fontsize=20, fontname='monospace')
plt.yticks(rotation=55, fontsize=20, fontname='monospace')
plt.title('2010 - 2022 Unemployment Rate', fontsize=30)
plt.legend(loc=2,prop={'size': 20})
plt.show()
```


    
![png](output_9_0.png)
    



```python
#Descriptive Statistics of unemployment
unemployment.groupby(['Year']).value.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012</th>
      <td>12.0</td>
      <td>8.075000</td>
      <td>0.213733</td>
      <td>7.7</td>
      <td>7.875</td>
      <td>8.20</td>
      <td>8.200</td>
      <td>8.3</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>12.0</td>
      <td>7.358333</td>
      <td>0.352803</td>
      <td>6.7</td>
      <td>7.200</td>
      <td>7.40</td>
      <td>7.525</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>12.0</td>
      <td>6.158333</td>
      <td>0.372847</td>
      <td>5.6</td>
      <td>5.875</td>
      <td>6.15</td>
      <td>6.375</td>
      <td>6.7</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>12.0</td>
      <td>5.275000</td>
      <td>0.245412</td>
      <td>5.0</td>
      <td>5.075</td>
      <td>5.25</td>
      <td>5.425</td>
      <td>5.7</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>12.0</td>
      <td>4.875000</td>
      <td>0.121543</td>
      <td>4.7</td>
      <td>4.800</td>
      <td>4.90</td>
      <td>4.925</td>
      <td>5.1</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>12.0</td>
      <td>4.358333</td>
      <td>0.167649</td>
      <td>4.1</td>
      <td>4.275</td>
      <td>4.35</td>
      <td>4.400</td>
      <td>4.7</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>12.0</td>
      <td>3.891667</td>
      <td>0.124011</td>
      <td>3.7</td>
      <td>3.800</td>
      <td>3.85</td>
      <td>4.000</td>
      <td>4.1</td>
    </tr>
    <tr>
      <th>2019</th>
      <td>12.0</td>
      <td>3.675000</td>
      <td>0.135680</td>
      <td>3.5</td>
      <td>3.600</td>
      <td>3.60</td>
      <td>3.725</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2020</th>
      <td>12.0</td>
      <td>8.091667</td>
      <td>3.616743</td>
      <td>3.5</td>
      <td>6.125</td>
      <td>7.40</td>
      <td>10.400</td>
      <td>14.7</td>
    </tr>
    <tr>
      <th>2021</th>
      <td>12.0</td>
      <td>5.358333</td>
      <td>0.831711</td>
      <td>3.9</td>
      <td>4.675</td>
      <td>5.60</td>
      <td>6.000</td>
      <td>6.4</td>
    </tr>
    <tr>
      <th>2022</th>
      <td>10.0</td>
      <td>3.660000</td>
      <td>0.150555</td>
      <td>3.5</td>
      <td>3.600</td>
      <td>3.60</td>
      <td>3.700</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



# Forcasting Unemployment


```python
#Creating train test for forcasting
train = unemployment[unemployment.index < pd.to_datetime("2020-09-01", format='%Y-%m-%d')]
train['train'] = train['value']
test = unemployment[unemployment.index >= pd.to_datetime("2020-09-01", format='%Y-%m-%d')]
test['test'] = test['value']
```

    /var/folders/d5/yv3yty4s3y33ty4r_pc546j80000gn/T/ipykernel_1035/984100058.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      train['train'] = train['value']
    /var/folders/d5/yv3yty4s3y33ty4r_pc546j80000gn/T/ipykernel_1035/984100058.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      test['test'] = test['value']



```python
#Delete other columns
del train['Jan']
del train['monthly_rolling']
del train['value']
del test['Jan']
del train['Year']
del test['monthly_rolling']
del test['value']
del test['Year']
```


```python
#Train test split graph
plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.title("Train/Test split for Passenger Data")
plt.ylabel("Passenger Number")
plt.xlabel('Year-Month')
sns.set() 
plt.show()
```


    
![png](output_14_0.png)
    



```python
#Dropping NA values from test set
test = test.dropna()
```


```python
from pmdarima.arima import auto_arima
model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train)
forecast = model.predict(n_periods=len(test))
forecast = pd.DataFrame(forecast,index = test.index,columns=['Prediction'])
```

    Performing stepwise search to minimize aic
     ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=313.107, Time=0.10 sec
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=311.838, Time=0.01 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=313.838, Time=0.02 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=313.838, Time=0.02 sec
     ARIMA(0,1,0)(0,0,0)[0]             : AIC=309.838, Time=0.01 sec
     ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=311.290, Time=0.05 sec
    
    Best model:  ARIMA(0,1,0)(0,0,0)[0]          
    Total fit time: 0.218 seconds


    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/statsmodels/tsa/statespace/sarimax.py:1899: RuntimeWarning: invalid value encountered in reciprocal
      return np.roots(self.polynomial_reduced_ar)**-1
    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/statsmodels/tsa/statespace/sarimax.py:1906: RuntimeWarning: invalid value encountered in reciprocal
      return np.roots(self.polynomial_reduced_ma)**-1
    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/statsmodels/tsa/statespace/sarimax.py:1899: RuntimeWarning: invalid value encountered in reciprocal
      return np.roots(self.polynomial_reduced_ar)**-1
    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/statsmodels/tsa/statespace/sarimax.py:1906: RuntimeWarning: invalid value encountered in reciprocal
      return np.roots(self.polynomial_reduced_ma)**-1



```python
from math import sqrt
from sklearn.metrics import mean_squared_error
rms = sqrt(mean_squared_error(test,forecast))
print("RMSE: ", rms)
```

    RMSE:  3.6783671126444974


ISSUE WITH CODE

ISSUE WITH CODE


```python
#Train test split graph
plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.plot(forecast, color = "orange")
plt.title("Train/Test split for Passenger Data")
plt.ylabel("Passenger Number")
plt.xlabel('Year-Month')
sns.set() 
plt.show()
```


    
![png](output_20_0.png)
    


# Web Scraping Data Collection (Stim)


```python
get_url = requests.get("https://www.pandemicoversight.gov/data-interactive-tools/data-stories/three-rounds-stimulus-checks-see-how-many-went-out-and-how-much")

get_text = get_url.text

soup = BeautifulSoup(get_text, "html.parser")
```


```python
soup.find('td', {'class' :'pos'}).find('strong').text
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Input In [76], in <cell line: 1>()
    ----> 1 soup.find('td', {'class' :'pos'}).find('strong').text


    AttributeError: 'NoneType' object has no attribute 'find'


# Inflation Rates


```python
#Importing data from usinflationcalculator.com
inflation = pd.read_html('https://www.usinflationcalculator.com/inflation/current-inflation-rates/')
```


```python
df = inflation[0]
header_row = df.iloc[0]
df2 = pd.DataFrame(df.values[1:], columns=header_row)
df2 = df2.replace(['Avail.Dec.13'],[0])
df2 = df2.iloc[: , :-1]
pivot = pd.pivot_table(df2, index = ['Year'])
```


```python
#Re ordering columns in pivoted table 
pivot = pivot.iloc[:,[4,3,7,0,8,6,5,1,11,10,9,2]]
```


```python
#Creating run on data frame for analyzing 
inflation_list = df2.melt(id_vars=["Year"], 
        var_name="Jan")
#Concating year and month 
inflation_list['Date'] = inflation_list.Year +' ' + inflation_list.Jan
```


```python
#Converting date to datetime64
inflation_list['Date'] = inflation_list['Date'].astype('datetime64')
#Ordering by ascending and resetting index
inflation_list = inflation_list.sort_values(by='Date').reset_index(drop=True)
#Setting datetime as index for plotting 
inflation_list = inflation_list.set_index(['Date'])
#Creating a rate column 
inflation_list['rate'] = inflation_list.value.astype(float)
```


```python
#Plotting inflation rates with monthly rolling average
inflation_list['monthly_rolling'] = inflation_list['rate'].rolling(10).mean()
inflation_list['monthly_rolling'].plot(label = 'monthly rolling')
inflation_list['rate'].plot(figsize = (15,7))
plt.xlabel('Time', fontsize=30)
plt.ylabel('Inflation rate', fontsize=30)
plt.xticks(ha='right', rotation=55, fontsize=20, fontname='monospace')
plt.yticks(rotation=55, fontsize=20, fontname='monospace')
plt.title('2010 - 2022 Inflation Rates', fontsize=30)
plt.legend(loc=2,prop={'size': 20})
plt.show()
```


    
![png](output_30_0.png)
    


# Bitcoin Data


```python
#Setting start and end dates
start = datetime.datetime(2000,1,1)
end = datetime.datetime(2023,1,1)
```


```python
#Reading in amazon and tesla history
bitcoin = web.DataReader("BTC-USD", 'yahoo', start, end )
```


```python
#Plitting out a time series of open and close on log scale
bitcoin['Open'].plot(label = 'bitcoin Open price' )
bitcoin['Close'].plot(label = 'bitcoin Close price' )
plt.yscale("log")
plt.legend()
plt.ylabel('Bitcoin Price')
plt.show()
```


    
![png](output_34_0.png)
    



```python
#Linear time series
#Price at open comp
bitcoin['Open'].plot(figsize = (15,7))
plt.title('Price at Open ')
```




    Text(0.5, 1.0, 'Price at Open ')




    
![png](output_35_1.png)
    


# Housing Data


```python
#Setting directory to folder containing housing data
os.chdir("Desktop/Portfolio Project/bitcoinproject")
```


```python
#Reading in the housing data
home_prices=pd.read_csv('ZHVI.csv')
```


```python
#Checking data types
home_prices.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 274 entries, 0 to 273
    Data columns (total 52 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Unnamed: 0                274 non-null    object 
     1   Virginia                  274 non-null    float64
     2   California                274 non-null    float64
     3   Arizona                   272 non-null    float64
     4   Florida                   274 non-null    float64
     5   New Jersey                274 non-null    float64
     6   Texas                     272 non-null    float64
     7   West Virginia             271 non-null    float64
     8   Colorado                  274 non-null    float64
     9   New York                  274 non-null    float64
     10  Georgia                   274 non-null    float64
     11  Nevada                    273 non-null    float64
     12  Massachusetts             274 non-null    float64
     13  Tennessee                 274 non-null    float64
     14  Michigan                  274 non-null    float64
     15  New Mexico                247 non-null    float64
     16  Montana                   212 non-null    float64
     17  Oregon                    274 non-null    float64
     18  Hawaii                    274 non-null    float64
     19  Minnesota                 274 non-null    float64
     20  Utah                      274 non-null    float64
     21  Alaska                    273 non-null    float64
     22  Pennsylvania              272 non-null    float64
     23  South Carolina            274 non-null    float64
     24  the District of Columbia  274 non-null    float64
     25  Maryland                  274 non-null    float64
     26  New Hampshire             274 non-null    float64
     27  Nebraska                  274 non-null    float64
     28  Washington                274 non-null    float64
     29  Iowa                      272 non-null    float64
     30  Missouri                  274 non-null    float64
     31  Rhode Island              274 non-null    float64
     32  Illinois                  271 non-null    float64
     33  Ohio                      274 non-null    float64
     34  Connecticut               272 non-null    float64
     35  North Carolina            273 non-null    float64
     36  Idaho                     270 non-null    float64
     37  Vermont                   273 non-null    float64
     38  North Dakota              166 non-null    float64
     39  Indiana                   274 non-null    float64
     40  Wisconsin                 274 non-null    float64
     41  Kansas                    273 non-null    float64
     42  Louisiana                 273 non-null    float64
     43  Mississippi               274 non-null    float64
     44  Kentucky                  274 non-null    float64
     45  Oklahoma                  274 non-null    float64
     46  Maine                     274 non-null    float64
     47  South Dakota              271 non-null    float64
     48  Wyoming                   246 non-null    float64
     49  Alabama                   274 non-null    float64
     50  Delaware                  274 non-null    float64
     51  Arkansas                  273 non-null    float64
    dtypes: float64(51), object(1)
    memory usage: 111.4+ KB



```python
#Renaming first column 
home_prices = home_prices.rename(columns={'Unnamed: 0': 'Date'})
```


```python
#Converting date to datetime64
home_prices['Date'] = home_prices['Date'].astype('datetime64')
#Ordering by ascending and resetting index
home_prices = home_prices.sort_values(by='Date').reset_index(drop=True)
#Setting datetime as index for plotting 
home_prices = home_prices.set_index(['Date'])
```


```python
home_prices.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Virginia</th>
      <th>California</th>
      <th>Arizona</th>
      <th>Florida</th>
      <th>New Jersey</th>
      <th>Texas</th>
      <th>West Virginia</th>
      <th>Colorado</th>
      <th>New York</th>
      <th>Georgia</th>
      <th>...</th>
      <th>Louisiana</th>
      <th>Mississippi</th>
      <th>Kentucky</th>
      <th>Oklahoma</th>
      <th>Maine</th>
      <th>South Dakota</th>
      <th>Wyoming</th>
      <th>Alabama</th>
      <th>Delaware</th>
      <th>Arkansas</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-01</th>
      <td>136883.0</td>
      <td>199819.0</td>
      <td>134833.0</td>
      <td>115496.0</td>
      <td>184652.0</td>
      <td>114273.0</td>
      <td>70996.0</td>
      <td>184080.0</td>
      <td>150445.0</td>
      <td>131886.0</td>
      <td>...</td>
      <td>106036.0</td>
      <td>85661.0</td>
      <td>93743.0</td>
      <td>80713.0</td>
      <td>115854.0</td>
      <td>119061.0</td>
      <td>NaN</td>
      <td>103693.0</td>
      <td>151890.0</td>
      <td>81581.0</td>
    </tr>
    <tr>
      <th>2000-02-01</th>
      <td>137375.0</td>
      <td>200668.0</td>
      <td>135160.0</td>
      <td>115837.0</td>
      <td>185415.0</td>
      <td>114398.0</td>
      <td>71056.0</td>
      <td>184977.0</td>
      <td>151215.0</td>
      <td>132331.0</td>
      <td>...</td>
      <td>106383.0</td>
      <td>85674.0</td>
      <td>93850.0</td>
      <td>80779.0</td>
      <td>116451.0</td>
      <td>118794.0</td>
      <td>NaN</td>
      <td>103973.0</td>
      <td>152286.0</td>
      <td>81755.0</td>
    </tr>
    <tr>
      <th>2000-03-01</th>
      <td>137833.0</td>
      <td>201736.0</td>
      <td>135544.0</td>
      <td>116198.0</td>
      <td>186054.0</td>
      <td>114437.0</td>
      <td>71130.0</td>
      <td>185827.0</td>
      <td>151858.0</td>
      <td>132774.0</td>
      <td>...</td>
      <td>106590.0</td>
      <td>85720.0</td>
      <td>93986.0</td>
      <td>80997.0</td>
      <td>116894.0</td>
      <td>118421.0</td>
      <td>NaN</td>
      <td>104127.0</td>
      <td>152696.0</td>
      <td>81908.0</td>
    </tr>
    <tr>
      <th>2000-04-01</th>
      <td>138657.0</td>
      <td>203985.0</td>
      <td>136292.0</td>
      <td>116897.0</td>
      <td>187429.0</td>
      <td>114644.0</td>
      <td>71262.0</td>
      <td>187745.0</td>
      <td>153216.0</td>
      <td>133662.0</td>
      <td>...</td>
      <td>107053.0</td>
      <td>85880.0</td>
      <td>94287.0</td>
      <td>81336.0</td>
      <td>117873.0</td>
      <td>118008.0</td>
      <td>NaN</td>
      <td>104476.0</td>
      <td>153562.0</td>
      <td>82218.0</td>
    </tr>
    <tr>
      <th>2000-05-01</th>
      <td>139378.0</td>
      <td>206395.0</td>
      <td>137072.0</td>
      <td>117563.0</td>
      <td>188623.0</td>
      <td>114694.0</td>
      <td>71392.0</td>
      <td>189645.0</td>
      <td>154443.0</td>
      <td>134509.0</td>
      <td>...</td>
      <td>107439.0</td>
      <td>86306.0</td>
      <td>94609.0</td>
      <td>81739.0</td>
      <td>118632.0</td>
      <td>117934.0</td>
      <td>NaN</td>
      <td>104806.0</td>
      <td>154454.0</td>
      <td>82446.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 51 columns</p>
</div>




```python
#Average home prices across the country 
home_prices.plot(figsize = (100,70),linewidth=4)
plt.xlabel('Time', fontsize=90)
plt.ylabel('Median Home Prices', fontsize=90)
plt.xticks(ha='right', rotation=55, fontsize=70, fontname='monospace')
plt.yticks(rotation=55, fontsize=70, fontname='monospace')
plt.title('2010 - 2022 Median Home Prices', fontsize=100)
plt.legend(loc=2,prop={'size': 50})
```




    <matplotlib.legend.Legend at 0x7f8ca4a53460>




    
![png](output_43_1.png)
    



```python
#Top 5 average home prices
pd.DataFrame(home_prices.mean().sort_values(ascending=False).head())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Hawaii</th>
      <td>509024.021898</td>
    </tr>
    <tr>
      <th>the District of Columbia</th>
      <td>453238.240876</td>
    </tr>
    <tr>
      <th>California</th>
      <td>425873.215328</td>
    </tr>
    <tr>
      <th>Massachusetts</th>
      <td>358995.551095</td>
    </tr>
    <tr>
      <th>New Jersey</th>
      <td>320147.357664</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Bottom 5 average home prices 
pd.DataFrame(home_prices.mean().sort_values(ascending=False).tail())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Kentucky</th>
      <td>125567.000000</td>
    </tr>
    <tr>
      <th>Arkansas</th>
      <td>114584.652015</td>
    </tr>
    <tr>
      <th>Mississippi</th>
      <td>114422.693431</td>
    </tr>
    <tr>
      <th>Oklahoma</th>
      <td>113882.649635</td>
    </tr>
    <tr>
      <th>West Virginia</th>
      <td>99073.018450</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Top 5 housing markets
home_prices[['Hawaii','the District of Columbia','Massachusetts','California','New Jersey']].plot()
plt.xlabel('Time', fontsize=20)
plt.ylabel('Median Home Prices', fontsize=20)
plt.xticks(ha='right', rotation=55, fontsize=20, fontname='monospace')
plt.yticks(rotation=55, fontsize=20, fontname='monospace')
plt.title('2010 - 2022 Median Home Prices', fontsize=20)
plt.legend(loc=2,prop={'size': 15})
```




    <matplotlib.legend.Legend at 0x7f8c70666190>




    
![png](output_46_1.png)
    



```python
#Bottom 5 housing markets
home_prices[['Kentucky','Arkansas','Mississippi','Oklahoma','West Virginia']].plot()
plt.xlabel('Time', fontsize=20)
plt.ylabel('Median Home Prices', fontsize=20)
plt.xticks(ha='right', rotation=55, fontsize=20, fontname='monospace')
plt.yticks(rotation=55, fontsize=20, fontname='monospace')
plt.title('2010 - 2022 Median Home Prices', fontsize=20)
plt.legend(loc=2,prop={'size': 15})
```




    <matplotlib.legend.Legend at 0x7f8c7064df10>




    
![png](output_47_1.png)
    



```python
#Aggregating all states to get US averages
us_home_averages = pd.DataFrame(home_prices.agg("mean", axis="columns"))
```


```python
#Changing name of column 
us_home_averages = us_home_averages.rename(columns={0: 'home_price'})
```

# Comparing Metrics


```python
#Getting the percentage change to see how they interact in a chart
#Changed the periods of the bitcoin to be every 30 days 
home_percent_change = us_home_averages.pct_change(periods=-1)
bitcoin_percent = bitcoin.Open.pct_change(periods = -30)
inflation_percent = inflation_list['rate'].pct_change(periods=-1)
```


```python
bitcoin['index_column'] = bitcoin.index
```


```python
bitcoin.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 2998 entries, 2014-09-17 to 2022-12-01
    Data columns (total 7 columns):
     #   Column        Non-Null Count  Dtype         
    ---  ------        --------------  -----         
     0   High          2998 non-null   float64       
     1   Low           2998 non-null   float64       
     2   Open          2998 non-null   float64       
     3   Close         2998 non-null   float64       
     4   Volume        2998 non-null   int64         
     5   Adj Close     2998 non-null   float64       
     6   index_column  2998 non-null   datetime64[ns]
    dtypes: datetime64[ns](1), float64(5), int64(1)
    memory usage: 187.4 KB



```python
#Setting range of the start of bitcoin to present date
left = datetime.date(2015, 3, 15)
right = date.today()

```


```python
#Plotting percentage change over time between homes,bitcoin and inflation
home_percent_change.plot(label = 'home prices')
bitcoin_percent.plot(label = 'bitcoin Open price' )
inflation_percent.plot(label = 'inflation rate')
plt.gca().set_xbound(left, right)
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f8ca479b3d0>




    
![png](output_55_1.png)
    


# Merging Data on Date


```python
#Correlation matrix
import seaborn as sns
```

Main Sets:
- home_prices
 - us_home_averages (aggregated set)
- bitcoin
- inflation_list
- unemployment


```python
inflation_list.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Jan</th>
      <th>value</th>
      <th>rate</th>
      <th>monthly_rolling</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-01</th>
      <td>2000</td>
      <td>Jan</td>
      <td>2.7</td>
      <td>2.7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-02-01</th>
      <td>2000</td>
      <td>Feb</td>
      <td>3.2</td>
      <td>3.2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-03-01</th>
      <td>2000</td>
      <td>Mar</td>
      <td>3.8</td>
      <td>3.8</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-04-01</th>
      <td>2000</td>
      <td>Apr</td>
      <td>3.1</td>
      <td>3.1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-05-01</th>
      <td>2000</td>
      <td>May</td>
      <td>3.2</td>
      <td>3.2</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
us_home_averages.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>home_price</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-01</th>
      <td>133801.000000</td>
    </tr>
    <tr>
      <th>2000-02-01</th>
      <td>134153.361702</td>
    </tr>
    <tr>
      <th>2000-03-01</th>
      <td>134519.063830</td>
    </tr>
    <tr>
      <th>2000-04-01</th>
      <td>135238.659574</td>
    </tr>
    <tr>
      <th>2000-05-01</th>
      <td>135994.042553</td>
    </tr>
  </tbody>
</table>
</div>




```python
bitcoin.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
      <th>index_column</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-09-17</th>
      <td>468.174011</td>
      <td>452.421997</td>
      <td>465.864014</td>
      <td>457.334015</td>
      <td>21056800</td>
      <td>457.334015</td>
      <td>2014-09-17</td>
    </tr>
    <tr>
      <th>2014-09-18</th>
      <td>456.859985</td>
      <td>413.104004</td>
      <td>456.859985</td>
      <td>424.440002</td>
      <td>34483200</td>
      <td>424.440002</td>
      <td>2014-09-18</td>
    </tr>
    <tr>
      <th>2014-09-19</th>
      <td>427.834991</td>
      <td>384.532013</td>
      <td>424.102997</td>
      <td>394.795990</td>
      <td>37919700</td>
      <td>394.795990</td>
      <td>2014-09-19</td>
    </tr>
    <tr>
      <th>2014-09-20</th>
      <td>423.295990</td>
      <td>389.882996</td>
      <td>394.673004</td>
      <td>408.903992</td>
      <td>36863600</td>
      <td>408.903992</td>
      <td>2014-09-20</td>
    </tr>
    <tr>
      <th>2014-09-21</th>
      <td>412.425995</td>
      <td>393.181000</td>
      <td>408.084991</td>
      <td>398.821014</td>
      <td>26580100</td>
      <td>398.821014</td>
      <td>2014-09-21</td>
    </tr>
  </tbody>
</table>
</div>




```python
unemployment.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Jan</th>
      <th>value</th>
      <th>monthly_rolling</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012-01-01</th>
      <td>2012</td>
      <td>Jan</td>
      <td>8.3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2012-02-01</th>
      <td>2012</td>
      <td>Feb</td>
      <td>8.3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2012-03-01</th>
      <td>2012</td>
      <td>Mar</td>
      <td>8.2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2012-04-01</th>
      <td>2012</td>
      <td>Apr</td>
      <td>8.2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2012-05-01</th>
      <td>2012</td>
      <td>May</td>
      <td>8.2</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Merging us homes and inflation
all_data = pd.merge(inflation_list,us_home_averages, on = 'Date')
```


```python
#Mergin in bitcoin data
all_data = pd.merge(all_data,bitcoin, on = 'Date')
```


```python
#taking in unemployment
all_data = pd.merge(all_data,unemployment, on = 'Date')
```


```python
#Creating empty frame to store data
data = pd.DataFrame()
```


```python
#Renaming column 
data[['Year', 'inflation rate', 'home_price','High', 'Low', 'Open', 
          'Close', 'Volume', 'Adj Close', 'unemployment rate']]=all_data[['Year_x', 'rate', 'home_price','High', 'Low', 'Open', 
          'Close', 'Volume', 'Adj Close', 'value_y']]
```


```python
#Correlation Heat Map 
matrix = data.corr().round(2)
sns.heatmap(matrix, annot=True)
plt.show()
```


    
![png](output_68_0.png)
    



```python

```
