import math
from pandas_datareader import data as pdr
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
import streamlit as st
plt.style.use('fivethirtyeight')

yf.pdr_override()
#Get the stock qoute
st.write("""
# Stock prices codes with company name 

|               Stock code                     |          Stock company   |
|----------------------------------------------|:------------------------:|
|TSLA                                          |Tesla                     |
|GOOG                                          |Alphabet Inc.             |
|AMZN                                          |Amazon.com, Inc.          |
|V                                             |Visa Inc                  |
|ENPH                                          |Enphase Energy, Inc.      |
|META                                          |Meta Platforms, Inc.      |
|NVDA                                          |NVIDIA Corporation        |
|NFLX                                          |Netflix, Inc.             |
|AAPL                                          |APPLE                     |

for more code condider the link ("https://finance.yahoo.com/lookup/")

write the code in the given box to predict the price of stocks and starting date and ending date which ypu want to consider for training the model


 """)

# Text input
name = st.text_input("Enter the code")
stdate=st.text_input("Enter starting date")
eddate=st.text_input("Enter ending date")



stockcode=""
# Button input

def fun():
  df = pdr.get_data_yahoo(stockcode, start=stdate, end=eddate)
  #df = web.DataReader('AAPL', data_source='yahoo', start='2007-10-01', end='2021-05-04')
  #show the data


  #Get the number of rows and columns in the data set
  df.shape

  #Visualize the closing price history
  plt.figure(figsize=(16,8))
  plt.title('Closing Price History')
  plt.plot(df['Close'])
  plt.xlabel('Date',fontsize=18)
  plt.ylabel('Close Price USD $', fontsize=18)
  plt.show()

  #Create a new Dataframe with only the Close Coloumn
  data = df.filter(['Close'])
  #Convert the dataframe to numpy array
  dataset = data.values
  #Get the number of rows to train the model on
  training_data_len = math.ceil(len(dataset)*.8)
  training_data_len

  #Scale the data
  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(dataset)


  #Create the training data set
  #Create the Scaled Training Data Set
  train_data = scaled_data[0:training_data_len, :] 
  #Split the data into x_train and y_train data sets
  x_train = []
  y_train = []



  for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
  #   if i<= 61:
  #     print(x_train)
  #     print(y_train)
  #     print()

  #Convert the x_train and y_train to numpy arrays
  x_train, y_train = np.array(x_train), np.array(y_train)

  #Reshape the data
  x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
  x_train.shape

  #Build the LSTM Model
  model = Sequential()
  model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
  model.add(LSTM(50,return_sequences=False))
  model.add(Dense(25))
  model.add(Dense(1))

  #Compile the model
  model.compile(optimizer='adam',loss='mean_squared_error')

  st.write("""
        
        HOLD ON WE ARE CALCULATIONG :)
  """)

  #train the model
  model.fit(x_train, y_train, batch_size=1, epochs=1)

  #Create the testing data set
  #Create a new array containing scaled values from index 1746 to 2257
  test_data = scaled_data[training_data_len-60:,:]
  #Create the data sets x_test and y_test
  x_test = []
  y_test = dataset[training_data_len:,:]
  for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0])

  #Convert the data to a numpy array
  x_test = np.array(x_test)

  #Reshape the data
  x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

  #Get the models predicted price values
  predictions = model.predict(x_test)
  predictions = scaler.inverse_transform(predictions)

  #Get the root mean sqaured error (RMSE)
  rmse = np.sqrt(np.mean(predictions-y_test)**2)


  #Plot the data
  train = data[:training_data_len]
  valid = data[training_data_len:]
  valid['Predictions'] = predictions
  #Visualize the data
  plt.figure(figsize=(16,8))
  plt.title('Model')
  plt.xlabel('Date',fontsize =18)
  plt.ylabel('Close Price USD $',fontsize =18)
  plt.plot(train['Close'])
  plt.plot(valid[['Close','Predictions']])
  plt.legend(['Train','Val','Predictions'],loc='lower right')
  plt.show()



  #Get the Quote 
  apple_quote = pdr.get_data_yahoo(stockcode, start="1986-01-01", end="2021-12-21")
  #Create a new dataFrame
  new_df = apple_quote.filter(['Close'])
  #Get the last 60 day closing price values and convert the dataframe to an array
  last_60_days = new_df[-60:].values
  #Scale the data to be values between 0 and 1
  last_60_days_scaled = scaler.transform(last_60_days)
  #Create an empty list
  X_test = []
  #Append the past 60 days 
  X_test.append(last_60_days_scaled)
  #Convert the X_test data set to a numpy array
  X_test = np.array(X_test)
  #Reshape the data
  X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
  #Get the predicted scaled price 
  pred_price = model.predict(X_test)
  #Undo the Scaling
  pred_price = scaler.inverse_transform(pred_price)

  st.write("""
        # Predicted price:
        
  """)
  # def ret(self):
  #     print(pred_price)
  #     return pred_price

  st.write(pred_price)


  #Get the Quote 
  #apple_quote2 = web.DataReader('AAPL', data_source='yahoo',start='2021-01-15',end = '2021-01-15')
  apple_quote2 = pdr.get_data_yahoo(stockcode, start="2021-12-20", end="2021-12-21")
  st.write("""
  # Original price:
  """)
  st.write(apple_quote2['Close'])


if st.button("Submit"):
   stockcode=name
   fun()
   


#to run streamlit run main.py