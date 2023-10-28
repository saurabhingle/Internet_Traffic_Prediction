# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime, timedelta

# Load the Exponential Smoothing model
model_filename = 'model.pkl'

# Create a Streamlit app
st.title('Daily Visitors Prediction App')
st.sidebar.header("Input Parameters")

# Upload a CSV file with historical visitor data
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with historical visitor data", type=["csv"])

# Custom CSS for styling
st.markdown(
    """
    <style>
    .st-bw {
        background-color: #f0f4f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if uploaded_file is not None:
    historical_visitor_data = pd.read_csv(uploaded_file)

    # Check if the 'Daily Visitors' column exists
    if 'Daily Visitors' not in historical_visitor_data.columns:
        st.warning("The 'Daily Visitors' column does not exist in the uploaded CSV file.")
    else:
        # Convert the 'Date' column to datetime
        historical_visitor_data['Date'] = pd.to_datetime(historical_visitor_data['Date'])
        st.write("Uploaded Historical Visitor Data:")
        st.dataframe(historical_visitor_data, height=400)

        # Input date for prediction
        last_date = historical_visitor_data['Date'].max()
        input_date = st.sidebar.date_input("Select a start date for prediction", min_value=(last_date + timedelta(days=1)).date())

        # Input number of days for prediction (1 to 365 days)
        prediction_days = st.sidebar.number_input("Select the number of days for prediction", min_value=1, max_value=365, value=1)

        if st.sidebar.button("Predict"):
            if len(historical_visitor_data) == 0:
                st.warning("Please upload historical visitor data first.")
            else:
                # Convert the input_date to Pandas Timestamp
                input_date = pd.Timestamp(input_date)
                # Create a date range for predictions
                prediction_dates = [input_date + timedelta(days=i) for i in range(prediction_days)]

                # Perform predictions for each day
                predictions = []
                for prediction_date in prediction_dates:
                    # Filter historical data up to the current prediction date
                    historical_data = historical_visitor_data[historical_visitor_data['Date'] <= prediction_date]
                    es_fit = ExponentialSmoothing(historical_data['Daily Visitors'], seasonal='add', seasonal_periods=7).fit()
                    predicted_visitors = es_fit.forecast(steps=1).iloc[0]  # Prediction for one day
                    
                    # Append the actual observation to the historical data
                    historical_visitor_data = historical_visitor_data.append({'Date': prediction_date, 'Daily Visitors': predicted_visitors}, ignore_index=True)
                    
                    # Format the date as 'dd-mm-yyyy'
                    formatted_date = prediction_date.strftime('%d-%m-%Y')
                    predictions.append((formatted_date, predicted_visitors))

                # Create a DataFrame for the predictions
                predictions_df = pd.DataFrame(predictions, columns=['Date', 'Predicted Visitors'])

                st.subheader("Predicted Visitors for Each Day:")
                st.dataframe(predictions_df, height=400)

                # Create a line chart for date-wise visitors using Matplotlib
                st.subheader("Date-wise Visitors")

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(predictions_df['Date'], predictions_df['Predicted Visitors'], label='Predicted Visitors', marker='o', markersize=5)
                ax.set_xlabel('Date')
                ax.set_ylabel('Visitors')
                ax.set_title('Date-wise Visitors')
                ax.grid(True)
                ax.legend()

                # Highlight the predicted values with a different color
                for prediction in predictions:
                    ax.annotate(f'Visitors: {prediction[1]:.2f}', (prediction[0], prediction[1]), textcoords="offset points", xytext=(0, 10), ha='center')

                # Display the Matplotlib chart using Streamlit
                st.pyplot(fig)
else:
    st.warning("Please upload a CSV file with historical visitor data in the sidebar.")











































