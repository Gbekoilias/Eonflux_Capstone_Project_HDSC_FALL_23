import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse

# Load dataset
data_url = 'modelling_data.csv'
df = pd.read_csv(data_url)

# Set the date column as the index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Function to fit VAR model and make forecasts
def fit_var_model(data, lags=1, forecast_years=1):
    model = VAR(data)
    model_fitted = model.fit(lags, ic='aic')
    forecast_steps = int(365 * forecast_years)  # Forecast for the specified number of years
    forecast = model_fitted.forecast(data.values[-lags:], steps=forecast_steps)
    return model_fitted, forecast

# Function to create the Streamlit app
def main():
    # Set Streamlit app to fullscreen
    st.set_page_config(page_title="West Africa Climate Change Analysis", layout="wide")

    # Centered title
    st.markdown("<h1 style='text-align: center;'>Forecasting Temperature Trends for Weather Stations in West Africa</h1>", unsafe_allow_html=True)

    # Display dataset
    st.subheader("Weather Station Dataset (1980 - 2022)")
    st.write(df)

    # Choose a weather station location for forecasting
    location_options = df.columns[1:]
    selected_locations = st.multiselect("Select weather station for forecasting:", location_options)
    
    if not selected_locations:
        st.warning("Please select at least two weather station for forecasting.")
        return

    # Choose number of lags for VAR model
    num_lags = st.slider("Select the number of lags for the VAR model:", min_value=1, max_value=12, value=1)

    # Choose number of forecast years
    forecast_years = st.slider("Select the number of forecast years:", min_value=1, max_value=10, value=1)

    # Split data into training and testing sets using 80:20
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    # Fit VAR model
    st.subheader("VAR Model Forecasting")

    st.write(f"Fitting VAR model for {', '.join(selected_locations)} with {num_lags} lags and forecasting for {forecast_years} years...")
    model_fitted, forecast = fit_var_model(train[selected_locations], lags=num_lags, forecast_years=forecast_years)

    # Plot actual vs. forecasted values
    st.write("Plotting actual vs. forecasted average temperature values...")
    
    # Create a new date index for the forecast   
    forecast_index = pd.date_range(start=test.index[-1], periods=len(forecast) + 1, freq='D')[1:]
    forecast_df = pd.DataFrame(forecast, index=forecast_index[:len(forecast)], columns=selected_locations)

    # Create separate subplots for each location using Plotly Express
    fig = make_subplots(rows=len(selected_locations) // 2 + len(selected_locations) % 2, cols=2,
                        subplot_titles=[f"Average Temperature for Weather Station - {location}" for location in selected_locations],
                        shared_xaxes=True, vertical_spacing=0.1)

    for i, location in enumerate(selected_locations):
        row = i // 2 + 1
        col = i % 2 + 1

        # Add training data trace
        fig.add_trace(go.Scatter(x=train.index, y=train[location], mode='lines', name=f"Training Data ({location})"), row=row, col=col)

        # Add test data trace
        fig.add_trace(go.Scatter(x=test.index, y=test[location], mode='lines', name=f"Test Data ({location})"), row=row, col=col)

        # Add forecast trace
        fig.add_trace(go.Scatter(x=forecast_index[:len(forecast)], y=forecast_df[location], mode='lines', name=f"VAR Forecast ({location})"), row=row, col=col)

        # Update subplot title
        fig.update_layout(title_text="Actual vs. Forecasted Values for Selected Weather Stations", showlegend=False)

    # Update layout and show the plot
    fig.update_layout(height=600 * (len(selected_locations) // 2 + len(selected_locations) % 2), showlegend=True, width=1500)
    st.plotly_chart(fig)

    # Evaluate the model
    # st.subheader("Model Evaluation")

    # # Align lengths for RMSE calculation
    # overlap_start = max(pd.to_datetime(test.index).min(), forecast_index.min())
    # overlap_end = min(pd.to_datetime(test.index).max(), forecast_index.max())

    # test_overlap = test.loc[overlap_start:overlap_end][selected_locations]
    # forecast_overlap = forecast_df.loc[overlap_start:overlap_end][selected_locations]
    
    # # Calculate RMSE
    # rmse_result = rmse(test_overlap, forecast_overlap)

    # # Round the RMSE to two decimal places
    # rounded_rmse = np.round(rmse_result, 2)

    # # Display rounded RMSE
    # st.write(f"Root Mean Squared Error (RMSE): {rounded_rmse}")

    # Display forecasted values
    st.subheader("Forecasted Values")

    # Create a DataFrame with forecasted values and station names as columns
    forecast_display = pd.DataFrame(forecast, index=forecast_index[:len(forecast)], columns=selected_locations)

    # Station names added as column names in the Forecast DataFrame
    forecast_display.columns = [f"{location} Forecast" for location in selected_locations]

    # Show the DataFrame in the Streamlit app
    st.write(forecast_display)
    
if __name__ == "__main__":
    main()