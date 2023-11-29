import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import math
import time
import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense, Flatten, Conv1D, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras import Sequential
from keras.utils import plot_model
from pickle import load, dump
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import statsmodels.api as sm
from math import sqrt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata
import warnings
warnings.filterwarnings("ignore")


all_tweets = pd.read_csv('./data/stock_tweets_with_sentiment.csv')
stock_prices_df = pd.read_csv('./data/stock_yfinance_data.csv')
stock_names = ['AAPL', 'AMZN', 'FB', 'GOOGL', 'MSFT', 'NFLX', 'TSLA']

sentiment_analyzer = SentimentIntensityAnalyzer()


class StockDataProcessor:
    def __init__(self, tweets_file, stock_prices_file):
        self.tweets_file = tweets_file
        self.stock_prices_file = stock_prices_file
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.all_tweets = pd.read_csv(self.tweets_file)
        self.stock_prices_df = pd.read_csv(self.stock_prices_file)

    def _load_data(self, stock_name):
        df = self.all_tweets[self.all_tweets['Stock Name'] == stock_name]
        df["Negative"] = ''
        df["Neutral"] = ''
        df["Positive"] = ''
        for idx, row in df.iterrows():
            try:
                sentence = unicodedata.normalize('NFKD', row["Tweet"])
                sentiment_scores = self.sentiment_analyzer.polarity_scores(sentence)
                df.at[idx, "Negative"] = sentiment_scores["neg"]
                df.at[idx, "Neutral"] = sentiment_scores["neu"]
                df.at[idx, "Positive"] = sentiment_scores["pos"]
            except Exception as e:
                print(f"Error processing row index {idx}: {e}")
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = df['Date'].dt.date
        df = df.drop(columns=['Tweet', 'Tweet Length', 'Stock Name', 'Company Name'])
        numeric_cols = df.select_dtypes(include=[np.number])
        twitter_df = numeric_cols.groupby(df['Date']).mean().reset_index()
        stock_df = self.stock_prices_df[self.stock_prices_df['Stock Name'] == stock_name]
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        stock_df['Date'] = stock_df['Date'].dt.date
        final_df = pd.merge(twitter_df, stock_df, on='Date', how='inner')
        return final_df.drop(columns=['Stock Name'])

    def _get_tech_indicators(self, data):
        # Calculate moving averages
        data['MA7'] = data['Close'].rolling(window=7).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()

        # Exponential Moving Averages
        data['26ema'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['12ema'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['MACD'] = data['12ema'] - data['26ema']
        data.drop(['26ema', '12ema'], axis=1, inplace=True)

        # Bollinger Bands
        data['20sd'] = data['Close'].rolling(20).std()
        data['upper_band'] = data['MA20'] + (data['20sd'] * 2)
        data['lower_band'] = data['MA20'] - (data['20sd'] * 2)
        data.drop(['20sd'], axis=1, inplace=True)

        # Exponential moving average
        data['ema'] = data['Close'].ewm(com=0.5).mean()

        # Log Momentum
        data['log_momentum'] = np.log(data['Close'] / data['Close'].shift(1))

        return data
    
    def _tech_ind(self, dataset):
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

        plt.show()


    def _normalize_data(self,df, range, target_column,stock_name):
    
        target_df_series = pd.DataFrame(df[target_column])
        data = pd.DataFrame(df.iloc[:, :])

        X_scaler = MinMaxScaler(feature_range=range)
        y_scaler = MinMaxScaler(feature_range=range)
        X_scaler.fit(data)
        y_scaler.fit(target_df_series)

        X_scale_dataset = X_scaler.fit_transform(data)
        y_scale_dataset = y_scaler.fit_transform(target_df_series)
        
        dump(X_scaler, open(f'./scalers/{stock_name}_X_scaler.pkl', 'wb'))
        dump(y_scaler, open(f'./scalers/{stock_name}_y_scaler.pkl', 'wb'))

        return (X_scale_dataset, y_scale_dataset)
    
    def batch_data(self,x_data,y_data, batch_size, predict_period):
        X_batched, y_batched, yc = list(), list(), list()

        for i in range(0,len(x_data),1):
            x_value = x_data[i: i + batch_size][:, :]
            y_value = y_data[i + batch_size: i + batch_size + predict_period][:, 0]
            yc_value = y_data[i: i + batch_size][:, :]
            if len(x_value) == batch_size and len(y_value) == predict_period:
                X_batched.append(x_value)
                y_batched.append(y_value)
                yc.append(yc_value)

        return np.array(X_batched), np.array(y_batched), np.array(yc)
    
    def split_train_test(self, data):
        train_size = len(data) - 20
        data_train = data[0:train_size]
        data_test = data[train_size:]
        return data_train, data_test
    
    def predict_index(self,dataset, X_train, batch_size, prediction_period):
        train_predict_index = dataset.iloc[batch_size: X_train.shape[0] + batch_size + prediction_period, :].index
        test_predict_index = dataset.iloc[X_train.shape[0] + batch_size:, :].index

        return train_predict_index, test_predict_index
    
    def get_processed_data(self, stock_name):
        df = self._load_data(stock_name)
        df = self._get_tech_indicators(df)
        df = df.iloc[20:, :].reset_index(drop=True)
        df.iloc[:, 1:] = pd.concat([df.iloc[:, 1:].ffill()])
        datetime_series = pd.to_datetime(df['Date'])
        datetime_index = pd.DatetimeIndex(datetime_series.values)
        df = df.set_index(datetime_index)
        df = df.sort_values(by='Date')
        df = df.drop(columns='Date')
        X_scale_dataset,y_scale_dataset = self._normalize_data(df, (-1,1), "Close",stock_name)
        X_batched, y_batched, yc = self.batch_data(X_scale_dataset, y_scale_dataset, batch_size = 5, predict_period = 1)
        X_train, X_test, = self.split_train_test(X_batched)
        y_train, y_test, = self.split_train_test(y_batched)
        yc_train, yc_test, = self.split_train_test(yc)
        index_train, index_test, = self.predict_index(df, X_train, 5, 1)
        return X_train, X_test, y_train, y_test, yc_train, yc_test, index_train, index_test
    

class GANModel:
    def __init__(self, input_dim, output_dim, feature_size,learning_rate,stock_name):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feature_size = feature_size
        self.learning_rate = learning_rate
        self.stock_name = stock_name
        self.generator = self.make_generator_model(input_dim,output_dim, feature_size)
        self.discriminator = self.make_discriminator_model(input_dim)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        X_scaler = load(open(f'./scalers/{stock_name}_X_scaler.pkl', 'rb'))
        y_scaler = load(open(f'./scalers/{stock_name}_y_scaler.pkl', 'rb'))


    def make_generator_model(self,input_dim,output_dim, feature_size):
        model = Sequential([
            LSTM(units=1024, return_sequences=True, input_shape=(self.input_dim, self.feature_size), recurrent_dropout=0.3),
            LSTM(units=512, return_sequences=True, recurrent_dropout=0.3),
            LSTM(units=256, return_sequences=True, recurrent_dropout=0.3),
            LSTM(units=128, return_sequences=True, recurrent_dropout=0.3),
            LSTM(units=64, recurrent_dropout=0.3),
            Dense(32),
            Dense(16),
            Dense(8),
            Dense(units=self.output_dim)
        ])
        return model

    def make_discriminator_model(self, input_dim):
        model = Sequential([
            Conv1D(8, input_shape=(self.input_dim+1, 1), kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)),
            Conv1D(16, kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)),
            Conv1D(32, kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)),
            Conv1D(64, kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)),
            Conv1D(128, kernel_size=1, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)),
            LeakyReLU(),
            Dense(220, use_bias=False),
            LeakyReLU(),
            Dense(220, use_bias=False, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        return model

    def discriminator_loss(self, real_output, fake_output):
        loss_f = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = loss_f(tf.ones_like(real_output), real_output)
        fake_loss = loss_f(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    def generator_loss(self, fake_output):
        loss_f = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return loss_f(tf.ones_like(fake_output), fake_output)
    
    @tf.function

    def train_step(self,real_x, real_y, yc, generator, discriminator, g_optimizer, d_optimizer):   
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = generator(real_x, training=True)
            generated_data_reshape = tf.reshape(generated_data, [generated_data.shape[0], generated_data.shape[1], 1])
            d_fake_input = tf.concat([tf.cast(generated_data_reshape, tf.float64), yc], axis=1)
            real_y_reshape = tf.reshape(real_y, [real_y.shape[0], real_y.shape[1], 1])
            d_real_input = tf.concat([real_y_reshape, yc], axis=1)

            real_output = discriminator(d_real_input, training=True)
            fake_output = discriminator(d_fake_input, training=True)

            g_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(g_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        g_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        d_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return real_y, generated_data, {'d_loss': disc_loss, 'g_loss': g_loss}


    def train(self, real_x, real_y, yc, epochs, generator, discriminator, g_optimizer, d_optimizer, checkpoint = 50):
        train_info = {}
        train_info["discriminator_loss"] = []
        train_info["generator_loss"] = []

        for epoch in tqdm(range(epochs)):
            real_price, fake_price, loss = self.train_step(real_x, real_y, yc, generator, discriminator, g_optimizer, d_optimizer)
            G_losses = []
            D_losses = []
            Real_price = []
            Predicted_price = []
            D_losses.append(loss['d_loss'].numpy())
            G_losses.append(loss['g_loss'].numpy())
            Predicted_price.append(fake_price.numpy())
            Real_price.append(real_price.numpy())

            if (epoch + 1) % checkpoint == 0:
                tf.keras.models.save_model(generator, f'./models_gan/{stock_name}/generator_V_%d.h5' % epoch)
                tf.keras.models.save_model(discriminator, f'./models_gan/{stock_name}/discriminator_V_%d.h5' % epoch)
                print('epoch', epoch + 1, 'discriminator_loss', loss['d_loss'].numpy(), 'generator_loss', loss['g_loss'].numpy())

            train_info["discriminator_loss"].append(D_losses)
            train_info["generator_loss"].append(G_losses)

        Predicted_price = np.array(Predicted_price)
        Predicted_price = Predicted_price.reshape(Predicted_price.shape[1], Predicted_price.shape[2])
        Real_price = np.array(Real_price)
        Real_price = Real_price.reshape(Real_price.shape[1], Real_price.shape[2])

        plt.subplot(2,1,1)
        plt.plot(train_info["discriminator_loss"], label='Disc_loss', color='#000000')
        plt.xlabel('Epoch')
        plt.ylabel('Discriminator Loss')
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(train_info["generator_loss"], label='Gen_loss', color='#000000')
        plt.xlabel('Epoch')
        plt.ylabel('Generator Loss')
        plt.legend()

        plt.savefig(f'./plots_gan/{stock_name}_loss.png')

        return Predicted_price, Real_price, np.sqrt(mean_squared_error(Real_price, Predicted_price)) / np.mean(Real_price)
    
    def plot_results(self, Real_price, Predicted_price, index_train):
        X_scaler = load(open(f'./scalers/{stock_name}_X_scaler.pkl', 'rb'))
        y_scaler = load(open(f'./scalers/{stock_name}_y_scaler.pkl', 'rb'))
        train_predict_index = index_train

        rescaled_Real_price = y_scaler.inverse_transform(Real_price)
        rescaled_Predicted_price = y_scaler.inverse_transform(Predicted_price)

        predict_result = pd.DataFrame()
        for i in range(rescaled_Predicted_price.shape[0]):
            y_predict = pd.DataFrame(rescaled_Predicted_price[i], columns=["predicted_price"], index=train_predict_index[i:i+output_dim])
            predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)
  
        real_price = pd.DataFrame()
        for i in range(rescaled_Real_price.shape[0]):
            y_train = pd.DataFrame(rescaled_Real_price[i], columns=["real_price"], index=train_predict_index[i:i+output_dim])
            real_price = pd.concat([real_price, y_train], axis=1, sort=False)
  
        predict_result['predicted_mean'] = predict_result.mean(axis=1)
        real_price['real_mean'] = real_price.mean(axis=1)

        plt.figure(figsize=(16, 8))
        plt.plot(real_price["real_mean"])
        plt.plot(predict_result["predicted_mean"], color = 'r')
        plt.xlabel("Date")
        plt.ylabel("Stock price")
        plt.legend(("Real price", "Predicted price"), loc="upper left", fontsize=16)
        plt.title("The result of Training", fontsize=20)
        plt.savefig(f'./plots_gan/{stock_name}_plot_result.png')
        plt.close()

        predicted = predict_result["predicted_mean"]
        real = real_price["real_mean"]
        For_MSE = pd.concat([predicted, real], axis = 1)
        RMSE = np.sqrt(mean_squared_error(predicted, real))
        print('-- Train RMSE -- ', RMSE)

    @tf.function
    def eval_op(self, generator, real_x):
        generated_data = generator(real_x, training = False)

        return generated_data
    
    def plot_test_data(self,Real_test_price, Predicted_test_price, index_test):
        X_scaler = load(open(f'./scalers/{stock_name}_X_scaler.pkl', 'rb'))
        y_scaler = load(open(f'./scalers/{stock_name}_y_scaler.pkl', 'rb'))
        test_predict_index = index_test

        rescaled_Real_price = y_scaler.inverse_transform(Real_test_price)
        rescaled_Predicted_price = y_scaler.inverse_transform(Predicted_test_price)

        predict_result = pd.DataFrame()
        for i in range(rescaled_Predicted_price.shape[0]):
            y_predict = pd.DataFrame(rescaled_Predicted_price[i], columns=["predicted_price"], index=test_predict_index[i:i+output_dim])
            predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)
  
        real_price = pd.DataFrame()
        for i in range(rescaled_Real_price.shape[0]):
            y_train = pd.DataFrame(rescaled_Real_price[i], columns=["real_price"], index=test_predict_index[i:i+output_dim])
            real_price = pd.concat([real_price, y_train], axis=1, sort=False)
  
        predict_result['predicted_mean'] = predict_result.mean(axis=1)
        real_price['real_mean'] = real_price.mean(axis=1)

        predicted = predict_result["predicted_mean"]
        real = real_price["real_mean"]
        For_MSE = pd.concat([predicted, real], axis = 1)
        RMSE = np.sqrt(mean_squared_error(predicted, real))
        print('Test RMSE: ', RMSE)
    
        plt.figure(figsize=(16, 8))
        plt.plot(real_price["real_mean"], color='#00008B')
        plt.plot(predict_result["predicted_mean"], color = '#8B0000', linestyle='--')
        plt.xlabel("Date")
        plt.ylabel("Stock price")
        plt.legend(("Real price", "Predicted price"), loc="upper left", fontsize=16)
        plt.title(f"Prediction on test data for {stock_name}", fontsize=20)
        plt.savefig(f'./plots_gan/{stock_name}_test.png')
        plt.close()


if __name__  == '__main__' :
    all_tweets = pd.read_csv('./data/stock_tweets_with_sentiment.csv')
    stock_prices_df = pd.read_csv('./data/stock_yfinance_data.csv')
    stock_names = ['AAPL', 'AMZN', 'MSFT', 'NFLX', 'TSLA']
    stock_names_error= ['FB', 'GOOGL']
    for stock_name in stock_names:
        stock_processor = StockDataProcessor('./data/stock_tweets_with_sentiment.csv', './data/stock_yfinance_data.csv')
        X_train, X_test, y_train, y_test, yc_train, yc_test, index_train, index_test = stock_processor.get_processed_data(stock_name)
        input_dim = X_train.shape[1]
        feature_size = X_train.shape[2]
        output_dim = y_train.shape[1]
        learning_rate = 5e-4
        epochs = 500

        gan_model = GANModel(input_dim, output_dim, feature_size, learning_rate)

        g_optimizer = gan_model.g_optimizer
        d_optimizer = gan_model.d_optimizer
        generator = gan_model.make_generator_model(input_dim, output_dim, feature_size)
        discriminator = gan_model.make_discriminator_model(input_dim)

        #Train the model
        predicted_price, real_price, RMSPE = gan_model.train(X_train, y_train, yc_train, epochs, generator, discriminator, g_optimizer, d_optimizer)
        test_generator = tf.keras.models.load_model(f'./models_gan/{stock_name}/generator_V_{epochs-1}.h5')
    
        predicted_test_data = gan_model.eval_op(test_generator, X_test)
        gan_model.plot_test_data(y_test, predicted_test_data,index_test)