import plotly.express as px
import pandas as pd
import streamlit as st

# Load datasets
tweets_sentiment_df = pd.read_csv('./data/stock_tweets_with_sentiment.csv')
stock_data_df = pd.read_csv('./data/stock_yfinance_data.csv')

# Tweet Length vs. Sentiment Score
fig1 = px.scatter(tweets_sentiment_df, x='Tweet Length', y='compound_score', title='Tweet Length vs. Sentiment Score')


# Sentiment Scores Over Time
tweets_sentiment_df['Date'] = pd.to_datetime(tweets_sentiment_df['Date']).dt.date
daily_avg_sentiment = tweets_sentiment_df.groupby('Date')['compound_score'].mean().reset_index()
fig2 = px.line(daily_avg_sentiment, x='Date', y='compound_score', title='Average Sentiment Score Over Time')


# Stock Price Movement
fig3 = px.line(stock_data_df, x='Date', y='Close', title='Stock Price Movement Over Time')


# Trading Volume Over Time
fig4 = px.bar(stock_data_df, x='Date', y='Volume', title='Trading Volume Over Time')


def app():
    st.title('Data Insights')

    st.header('A visualization of the sentiment scores in our dataset')
    st.plotly_chart(fig1)

    st.header('Average sentiment scores over time')
    st.plotly_chart(fig2)

    st.header('Stock price movement over time')
    st.plotly_chart(fig3)

    st.header('Trading volume over time')
    st.plotly_chart(fig4)
