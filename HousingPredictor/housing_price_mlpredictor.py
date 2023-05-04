# Below are all the imported modules and functions required for the application
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import streamlit as st

# Read Median_Value.csv data and placed into the panda dataframe named "mv_dataframe"
# Column names are noted as Date and MedianValue in accordance with the data
mv_dataframe = pd.read_csv('Median_Value.csv', names=['Date', 'MedianValue'], parse_dates=['Date'], header=None)

# Read HPI.csv data is read and placed into the dataframe named "hpi_dataframe" for clarity
# Note: the HPI values are multiped by 1000, this done to make the values usable for later
# as it will match the median values in the model (HPI is a different metric for measuring median pricing)
hpi_dataframe = pd.read_csv('HPI.csv', names=['Date', 'HPI'], parse_dates=['Date'], header=None)
hpi_dataframe['HPI'] = hpi_dataframe['HPI'] * 1000  # Adjusted HPI values from 100.1 to 100,100 for example

# Combine the two datasets via an outer join
# The date column is used as the "join" location
combined_dataset = mv_dataframe.merge(hpi_dataframe, on='Date', how='outer')
# The "combined_dataset" is sorted by date in ascending order while also resetting the index
combined_dataset = combined_dataset.sort_values('Date').reset_index(drop=True)
# New column named "price estimate" is created which fills in missing values with similar values in HPI column
# Note how fillna is used from pandas to fill the missing values
combined_dataset['PriceEstimate'] = combined_dataset['MedianValue'].fillna(combined_dataset['HPI'])

# Below are additional new columns created for the dataset, year, month, and yearmonth.
combined_dataset['Year'] = combined_dataset['Date'].dt.year
combined_dataset['Month'] = combined_dataset['Date'].dt.month
# This creates another unique value by triplying the year by 100 and adding the month to it
# Similar to a primary key in SQL where it will make all the values unique
combined_dataset['YearMonth'] = combined_dataset['Year'] * 100 + combined_dataset['Month']

# The columns below are converted into NumPy arrays for scikit-learn to process
# A 1D and 2D array is created noted by the [] and [[]] respectively
# YearMonth and Price Estimate columns are selected for this model
X = combined_dataset[['YearMonth']].values
y = combined_dataset['PriceEstimate'].values

# Below we split the feature matrix into training and testing sets of data
# This is crucial for training any form of ML model
# Test size is 20% shown by the .2 which I found to be the norm, 42 is an arbitrary number chosen to be the seed value
# train_test_split is using the skilearn module, purpose of this function is to split the dataset into a training and testing set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# An instance/object of the LinearRegression class is created form the scikit learn module
model_trained = LinearRegression()
# The model is then trained using the training data from above
model_trained.fit(X_train, y_train)
# This is where the predictions are created, they are then stored in the variable "y_prediction"
y_prediction = model_trained.predict(X_test)
# The r-squared score is calculated here for accuracy
# compares the target values with the predicted target values
r2accuracy = r2_score(y_test, y_prediction)

# Title of the streamlit interface is made, st calls the streamlit module
st.title("[Housing Price Predictor Phoenix, AZ]")

# Subheading is drawn on streamlit interface, uses Markdown language
st.write("### Median Housing Price Data")
# Line chart is created in streamlit
st.line_chart(mv_dataframe[['Date', 'MedianValue']].set_index('Date'))

# Another subheading is written and bar chart is created
st.write("### Modified HPI Bar Chart")
st.bar_chart(hpi_dataframe[['Date', 'HPI']].set_index('Date'))

# Subheading is written for scatter plot
st.write("### HPI and Median Values Scatter Plot")
# Matplotlib chart and axis are created
fig, ax = plt.subplots()
# Median and HPI values are added to scatter plot
ax.scatter(mv_dataframe['Date'], mv_dataframe['MedianValue'], label='Median Values')
ax.scatter(hpi_dataframe['Date'], hpi_dataframe['HPI'], label='HPI Values')
# Legend added for identification of which points is which
ax.legend()
# Scatter plot is displayed
st.pyplot(fig)

# Subheading and sliders for the user to interact with
# Year for prediction can be within the range of 2024 to 2030
st.write("### Predict Average Median Home Price")
year = st.slider("Select a year (2024-2030)", min_value=2024, max_value=2030, step=1)
month = st.slider("Select a month (1-12)", min_value=1, max_value=12, step=1)

# This is calculation of the unique yearmonth value
# 100 * the year + the month
year_month = year * 100 + month
# Previously trained linear regression model is used to predict the average median home price for specific year and month
predicted_price = model_trained.predict([[year_month]])

# Formatted strings below write the average median home price for the selected year, month, and price
st.markdown(f"## **Predicted average median home price for {month}/{year}: ${predicted_price[0]:,.2f}**")
# Another formatted string which writs the models r-squared score from above as a percentage in streamlit
st.markdown(f"### Accuracy of the model: {r2accuracy * 100:.2f}%")

# Calculate percentage error is done in these lines
# The last row is selected as noted by the index [-1] in the combined dataset.
# This value is then extracted from the PriceEstimate column, the variable last_known_price now stores this value
last_known_price = combined_dataset.iloc[-1]['PriceEstimate']
# To calculate the percentage error we need to find the absolute (abs) difference between the "predicted price" and "last known price"
# Once this difference is found it is divided by the "last known price" to determine the proportion
# Once the proportion is found it is then multiplied by 100 to find the percentage error
# Overall, the percentage error allows the user to see how close the predicted value is to the "real value", lower % is ideal
percent_error = abs(predicted_price[0] - last_known_price) / last_known_price * 100
# Percentage error is displayed via the formatted string below on streamlit web app
st.markdown(f"### Percentage error compared to the last known price: {percent_error:.2f}%")
