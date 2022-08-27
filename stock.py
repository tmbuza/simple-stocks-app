# !pip install streamlit
# !pip install yfinance
# !pip install pandas

import streamlit as st
"""
# Python and Streamlit Simple Apps
"""

import yfinance as yf
import pandas as pd


"""
##  `GOOGL` stock data from 2010 to 2020

"""
tickerSymbol = 'GOOGL'

# Now, extract data from the ticker and store it in an object named `tickerData`.
tickerData = yf.Ticker(tickerSymbol)


# Get prices for for the past ten years.
tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2020-1-31')

"""
![](img/stock.png)
"""

tickerDf.head(10).to_csv('stockhead.csv')
stockhead = pd.read_csv('stockhead.csv')
stockhead

"""
### Line charts for `Open`
"""
st.line_chart(tickerDf.Open)

"""
### Line charts for `Close`
"""
st.line_chart(tickerDf.Close)

"""
### Line charts for `Volume`
"""
st.line_chart(tickerDf.Volume)


"""
### Line charts for Four Features
"""
st.line_chart(data = tickerDf.iloc[:, 0:4])
