# !pip install streamlit
# !pip install yfinance
# !pip install pandas

import streamlit as st
"""
# Python and Steamlit Simple Apps
## Integrated practical user guides
"""

import yfinance as yf
import pandas as pd
import yfinance as yf


## Quick glimpse

"""
We will work with stock data and we can get started by defining the ticker symbol. Here we choose `GOOGL` like so:
"""
tickerSymbol = 'GOOGL'

"""
Now, extract data from the ticker and store it in an object named `tickerData`.
"""
tickerData = yf.Ticker(tickerSymbol)

"""
Get prices for for the past ten years.
"""

tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2020-1-31')

"""
## Data structure
![](img/stock.png)
"""

tickerDf.head(10).to_csv('stockhead.csv')
stockhead = pd.read_csv('stockhead.csv')
stockhead

"""
### Line charts for **Open**
"""
st.line_chart(tickerDf.Open)

"""
### Line charts for **Close**
"""
st.line_chart(tickerDf.Close)

"""
### Line charts for **Volume**
"""
st.line_chart(tickerDf.Volume)


"""
### Line charts for **Four Features**
"""
st.line_chart(data = tickerDf.iloc[:, 0:4])

