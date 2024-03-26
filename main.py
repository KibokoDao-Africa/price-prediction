from fastapi import FastAPI
from giza_datasets import DatasetsLoader
from pydantic import BaseModel
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import datetime


app = FastAPI()

loader = DatasetsLoader()
df = loader.load('tokens-ohcl')
data = df.to_pandas()

data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

#  Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Define a function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

class Token(BaseModel):
    userdate:str
    usertoken:str

@app.get('/')
def initialize():
    return {"start":"Initialized"}

@app.post('/predict')
def predict(data:Token):
    # Hyperparameters
    userdate = data.userdate
    usertoken = data.usertoken
    sequence_length = 100
    train_size = 0.8

    # Create sequences
    X, y = create_sequences(scaled_data, sequence_length)

    # Split data into training and testing sets
    split_index = int(train_size * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Reshape data for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)

    # Evaluate the model
    mse = model.evaluate(X_test, y_test, verbose=0)
    print(f'Mean Squared Error on Test Set: {mse}')
    def predict_closing_price(date, token):
    # Reshape input for prediction
        input_data = scaled_data[-sequence_length:].reshape((1, sequence_length, 1))
        prediction = model.predict(input_data)[0][0]
        # Inverse transform the prediction
        predicted_price = scaler.inverse_transform([[prediction]])[0][0]
        return predicted_price
    userdate = datetime.datetime.strptime(userdate, '%Y-%m-%d')

    # Predict closing price
    predicted_price = predict_closing_price(userdate, usertoken)
    print(f"Predicted closing price for {usertoken} on {userdate.strftime('%Y-%m-%d')}: {predicted_price}")
    return {"result":f"Predicted closing price for {usertoken} on {userdate.strftime('%Y-%m-%d')}: {predicted_price}"}



