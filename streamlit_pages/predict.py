import os
import streamlit as st
import pandas as pd
import tensorflow as tf
#Other necessary imports
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from load_and_process_data import StockDataProcessor, GANModel
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from pickle import load
import plotly.express as px
import numpy as np

stock_names =  ['AAPL', 'AMZN', 'MSFT', 'TSLA']
all_tweets = pd.read_csv('./data/stock_tweets_with_sentiment.csv')
stock_prices_df = pd.read_csv('./data/stock_yfinance_data.csv')
stock_processor = StockDataProcessor('./data/stock_tweets_with_sentiment.csv', './data/stock_yfinance_data.csv')

def plot_tech_indicators(dataset):
    fig,ax = plt.subplots(figsize=(15,8), dpi = 200)
    x_ = range(3,dataset.shape[0])
    x_ = list(dataset.index)

    ax.plot(dataset['Date'], dataset['MA7'], label='Moving Average 7 Days', color='g',linestyle='--')
    ax.plot(dataset['Date'], dataset['Close'],label='Closing Price', color='b')
    ax.plot(dataset['Date'], dataset['MA20'],label='Moving Average (20 days)', color='r',linestyle='--')
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.title('Techincal Indicators')
    plt.ylabel('Close Price in USD')
    plt.xlabel('Date')
    plt.legend()

    return fig

def load_latest_model(stock_name, models_dir='models_gan'):
    # Construct the path to the specific stock's model directory
    stock_model_dir = os.path.join(models_dir, stock_name)
    
    # List all generator model files for the given stock
    generator_models = [file for file in os.listdir(stock_model_dir) if 'generator' in file and file.endswith('.h5')]
    
    # Find the latest model (highest version number)
    latest_model = max(generator_models, key=lambda x: int(x.split('_V_')[1].split('.h5')[0]))
    
    # Load the latest model
    latest_model_path = os.path.join(stock_model_dir, latest_model)
    model = tf.keras.models.load_model(latest_model_path)
    
    return model


def make_predictions(gan_model, model, X_test):
    # Preprocess the data
    predictions = gan_model.eval_op(model, X_test)
    return predictions

def plot_predictions (y_test, predictions, index_test, stock_name):
    data = pd.DataFrame({'Date': index_test, 'Real Price': y_test.reshape(-1), 'Predicted Price': predictions.reshape(-1)})
    fig = px.line(data, x='Date', y=['Real Price', 'Predicted Price'], 
                  title=f'Stock Price Prediction for {stock_name}',
                  labels={'value': 'Price', 'variable': 'Price Type'})
    
    #Enhance the layout
    fig.update_layout(xaxis_title='Date',
                      yaxis_title='Price in USD',
                      legend_title = 'Price Type',
                      hovermode='x unified')
    
    return fig
                      



def plot_tech_indicators_with_plotly(dataset):
    # Convert 'Date' to datetime if not already done
    dataset['Date'] = pd.to_datetime(dataset['Date'])

    # Create a line plot with customizations
    fig = px.line(dataset, x='Date', y=['MA7', 'Close', 'MA20'], title="Technical Indicators")

    # Customization: update traces for specific styling
    fig.update_traces(
        line=dict(dash='dash'),  # Dashed line for MA7
        selector=dict(name="MA7")  # This assumes your column name is 'MA7'
    )
    fig.update_traces(
        line=dict(dash='solid'),  # Solid line for Close
        selector=dict(name="Close")  # This assumes your column name is 'Close'
    )
    fig.update_traces(
        line=dict(dash='dash'),  # Dashed line for MA20
        selector=dict(name="MA20")  # This assumes your column name is 'MA20'
    )

    
    colors = ['#FF0000', '#00FF00', '#0000FF']  
    fig.update_traces(marker=dict(color=colors[0]), selector=dict(name="MA7"))
    fig.update_traces(marker=dict(color=colors[1]), selector=dict(name="Close"))
    fig.update_traces(marker=dict(color=colors[2]), selector=dict(name="MA20"))

    # Customization: update layout for legend, axes titles, etc.
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Close Price in USD",
        legend_title="Indicator",
        font=dict(size=14),
    )

    return fig

def align_data_with_index(dataframe, index, column_name):
    # Ensure the 'Date' column is in datetime format
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])

    # Set the index of the dataframe to the date
    aligned_df = dataframe.set_index('Date')

    # Reindex the dataframe to match index_test and fill missing values
    aligned_df = aligned_df.reindex(index, fill_value=np.nan)

    return aligned_df[column_name].values



def app():
    selected_stock = st.selectbox('Select a stock for analysis:', stock_names)
    stock_name =selected_stock

    if selected_stock :
        st.write(f'You selected: {selected_stock}')
        df  = stock_processor._load_data(selected_stock)
        df= stock_processor._get_tech_indicators(df)
        X_train, X_test, y_train, y_test, yc_train, yc_test, index_train, index_test = stock_processor.get_processed_data(selected_stock)


        st.write("Plotting the Technical Indicators")
        with st.spinner('Plotting...'):
            fig = plot_tech_indicators(df)
            st.pyplot(fig)

        st.write("Plotting the Technical Indicators  with PX")
        with st.spinner('Plotting...'):
            fig = plot_tech_indicators_with_plotly(df)
            st.plotly_chart(fig)

    st.write("Data Insights")
    # Example: Show the shape of the dataset
    st.write("Training data shape:", X_train.shape)
    st.write("Test data shape:", X_test.shape)

    df['Date'] = pd.to_datetime(df['Date'])

    X_train, X_test, y_train, y_test, _, yc_test, _, index_test = stock_processor.get_processed_data(selected_stock)


    input_dim = X_train.shape[1]
    feature_size = X_train.shape[2]
    output_dim = y_train.shape[1]
    learning_rate = 5e-4
    epochs = 500

    gan_model = GANModel(input_dim, feature_size, output_dim, learning_rate,selected_stock)
    
    date_options = index_test.strftime('%Y-%m-%d').tolist()

    st.write("Select the date range:")
    #start_date = st.selectbox('Start date', date_options, index=0)
    #end_date = st.selectbox('End date', date_options, index=len(date_options) - 1)

    #start_date = pd.to_datetime(start_date)
    #end_date = pd.to_datetime(end_date)

    min_date =  index_test.min()
    max_date = index_test.max()

    start_date =  st.date_input('Start date', min_value=min_date, max_value=max_date, value=min_date)
    EncodingWarning = st.warning('End date must fall after start date.')
    end_date = st.date_input('End date', min_value=min_date, max_value=max_date, value=max_date)

    #df['Date'] = pd.to_datetime(df['Date'])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    
    if start_date < end_date:

        if st.button(f'Load model and predict prices for {selected_stock}') and selected_stock:
            

            model = load_latest_model(selected_stock)
            st.success(f'Successfully loaded model for {selected_stock}')

            
            

            predictions = make_predictions(gan_model, model, X_test)
            y_scaler = load(open(f'./scalers/{selected_stock}_y_scaler.pkl', 'rb'))
            predictions = y_scaler.inverse_transform(predictions)

            #st.write('length of predictions', len(predictions))
            #st.write('length of index_test', len(index_test))
            #st.write('length of y_test', len(y_test))
            #st.write('index_test', index_test)
                     
            mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
            filtered_data = df.loc[mask]

            #st.write('length of filtered data', len(filtered_data['Date']))
            #st.write('length of filtered data', len(filtered_data['Close']))
            filtered_index_test = index_test[(index_test >= start_date) & (index_test <= end_date)]


            aligned_real_prices = align_data_with_index(filtered_data, filtered_index_test, 'Close')

            aligned_predictions = predictions[:len(filtered_index_test)]

            fig = plot_predictions(aligned_real_prices, aligned_predictions, filtered_index_test, selected_stock)
            st.plotly_chart(fig)
    else:
        st.error('Error: End date must fall after start date.')








    



