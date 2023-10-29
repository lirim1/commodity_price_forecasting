import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load your original dataset (as provided earlier)
data = pd.read_csv('Nat_Gas.csv')
data['Dates'] = pd.to_datetime(data['Dates'])
data['Numerical_Date'] = data['Dates'].sub(data['Dates'].min()).dt.days  # Convert to numerical format

# Create a list of dates for the next 18 months
last_date = data['Dates'].max()
future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 19)]
future_numerical_dates = [(d - data['Dates'].min()).days for d in future_dates]

# Create a new DataFrame for future predictions
future_data = pd.DataFrame({'Numerical_Date': future_numerical_dates})

# Train a Linear Regression model on the original data
X = data[['Numerical_Date']]
y = data['Prices']
model = LinearRegression()
model.fit(X, y)

# Predict prices for the next 18 months
future_prices = model.predict(future_data)

# Create a DataFrame for future dates and prices
future_prices_df = pd.DataFrame({'Date': future_dates, 'Predicted_Price': future_prices})

# Output the 18 months into the future prices
print(future_prices_df)
