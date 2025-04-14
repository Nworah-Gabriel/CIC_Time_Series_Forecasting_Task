import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta, datetime

# Set random seed for reproducibility
np.random.seed(42)
sns.set_style("whitegrid")

def load_and_prepare_data(filepath):
    """Load and prepare the dataset with proper frequency"""
    df = pd.read_excel(filepath, parse_dates=['Date'], index_col='Date')
    df.columns = ['CIC']
    
    # Ensure the index has daily frequency
    df = df.asfreq('D')
    
    # Basic data validation
    print("\nMissing values before processing:", df.isnull().sum())
    if df.isnull().sum().any():
        print("Interpolating missing values...")
        df = df.interpolate(method='time')
        print("Missing values after processing:", df.isnull().sum())
    
    return df

def explore_data(df):
    """Perform exploratory data analysis"""
    print("\n=== Data Summary ===")
    print(df.describe())
    
    # Plot raw data
    plt.figure(figsize=(14, 6))
    df['CIC'].plot(title='Currency in Circulation (2011-2025)', color='navy')
    plt.ylabel('Billions of Pesos')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Currency_in_Circulation_(2011-2025).png')
    
    return df

def analyze_time_components(df):
    """Analyze time series components"""
    print("\n=== Time Series Decomposition ===")
    
    if df.isnull().sum().any():
        print("Warning: Missing values found in data. Interpolating...")
        df = df.interpolate(method='time')
    
    try:
        # Using 7-day period for weekly seasonality
        decomposition = seasonal_decompose(df['CIC'], model='additive', period=7, extrapolate_trend='freq')
        fig = decomposition.plot()
        fig.set_size_inches(14, 8)
        plt.tight_layout()
        plt.savefig('first_analized_time_series.png')
    except ValueError as e:
        print(f"Decomposition error: {str(e)}")
        return df
    
    # Stationarity check
    from statsmodels.tsa.stattools import adfuller
    adf_result = adfuller(df['CIC'].dropna())
    print(f'ADF Statistic: {adf_result[0]:.4f}')
    print(f'p-value: {adf_result[1]:.4f}')
    print('Critical Values:')
    for key, value in adf_result[4].items():
        print(f'\t{key}: {value:.4f}')
    
    # ACF/PACF plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(df['CIC'].dropna(), lags=60, ax=ax1)
    plot_pacf(df['CIC'].dropna(), lags=60, ax=ax2)
    plt.tight_layout()
    plt.savefig('second_analized_time_series.png')
    
    return df

def train_and_evaluate_model(df):
    """Train and evaluate SARIMA model on daily data"""
    print("\n=== Model Training ===")
    # Train-test split (last 60 days for testing)
    train = df.iloc[:-60]
    test = df.iloc[-60:]
    
    # Model parameters - adjusted for daily data
    p, d, q = 1, 1, 1
    P, D, Q, m = 1, 1, 1, 7  # Weekly seasonality
    
    # Fit SARIMA model with frequency information
    model = SARIMAX(train['CIC'],
                   order=(p, d, q),
                   seasonal_order=(P, D, Q, m),
                   enforce_stationarity=False,
                   enforce_invertibility=False,
                   freq='D')
    
    model_fit = model.fit(disp=False)
    print(model_fit.summary())
    
    # Forecast and evaluate
    forecast = model_fit.get_forecast(steps=len(test))
    forecast_values = forecast.predicted_mean
    
    # Ensure we align the indices properly
    test_values = test['CIC'].reindex(forecast_values.index)
    
    # Calculate metrics
    mae = mean_absolute_error(test_values, forecast_values)
    rmse = np.sqrt(mean_squared_error(test_values, forecast_values))
    
    # Safe MAPE calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((test_values - forecast_values) / test_values)) * 100
        mape = np.nan_to_num(mape, nan=0.0, posinf=0.0, neginf=0.0)
    
    print("\n=== Model Performance ===")
    print(f'MAE: {mae:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'MAPE: {mape:.2f}%')
    
    # Plot test vs forecast
    plt.figure(figsize=(14, 6))
    plt.plot(train.index[-120:], train['CIC'].tail(120), label='Training Data', color='navy')
    plt.plot(test.index, test['CIC'], label='Actual Values', color='darkgreen')
    plt.plot(forecast_values.index, forecast_values, label='Forecast', color='crimson')
    plt.fill_between(forecast.conf_int().index,
                    forecast.conf_int().iloc[:, 0],
                    forecast.conf_int().iloc[:, 1],
                    color='pink', alpha=0.3)
    plt.title('Model Evaluation on Test Set (Last 60 Days)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Model_Evaluation_test_set.png')
    
    return model_fit, (p, d, q), (P, D, Q, m)

def make_daily_forecast(df, model_fit, order, seasonal_order):
    """Generate daily forecasts for specific working days"""
    print("\n=== Working Days Forecast (April 2025) ===")
    
    # Define working days (excluding weekends and holidays)
    working_days = [
        '2025-04-01', '2025-04-02', '2025-04-03', '2025-04-04',
        '2025-04-07', '2025-04-08', '2025-04-09', '2025-04-10', '2025-04-11',
        '2025-04-14', '2025-04-15', '2025-04-16', '2025-04-17',
        '2025-04-22', '2025-04-23', '2025-04-24', '2025-04-25',
        '2025-04-28', '2025-04-29', '2025-04-30'
    ]
    forecast_dates = pd.to_datetime(working_days)
    
    # Calculate how many days to forecast
    last_date = df.index[-1]
    days_needed = (forecast_dates[-1] - last_date).days
    
    if days_needed <= 0:
        print("Error: Forecast dates are before the last available data point.")
        return pd.DataFrame()
    
    # Fit final model on all data with frequency information
    final_model = SARIMAX(df['CIC'],
                         order=order,
                         seasonal_order=seasonal_order,
                         enforce_stationarity=False,
                         enforce_invertibility=False,
                         freq='D')
    
    final_model_fit = final_model.fit(disp=False)
    
    # Generate forecast for all needed days
    forecast = final_model_fit.get_forecast(steps=days_needed)
    
    # Create forecast dataframe with proper dates
    forecast_df = pd.DataFrame({
        'Forecast': forecast.predicted_mean,
        'Lower_CI': forecast.conf_int().iloc[:, 0],
        'Upper_CI': forecast.conf_int().iloc[:, 1]
    }, index=pd.date_range(start=last_date + timedelta(days=1), periods=days_needed, freq='D'))
    
    # Filter for our specific working days
    daily_forecast_df = forecast_df.loc[forecast_dates]
    
    # Plot forecast
    plt.figure(figsize=(14, 6))
    df['CIC'].plot(label='Historical Data', color='navy', alpha=0.7)
    forecast_df['Forecast'].plot(label='Daily Forecast', color='lightcoral', alpha=0.5)
    daily_forecast_df['Forecast'].plot(label='Working Day Forecast', color='crimson', marker='o', linestyle='')
    plt.fill_between(forecast_df.index,
                    forecast_df['Lower_CI'],
                    forecast_df['Upper_CI'],
                    color='pink', alpha=0.2)
    
    # Highlight holidays
    plt.axvspan(pd.to_datetime('2025-04-18'), pd.to_datetime('2025-04-21'), 
               color='lightgray', alpha=0.5, label='Easter Holidays')
    
    plt.title('20 Working Days Forecast (April 2025)\nExcluding Weekends and Easter Holidays')
    plt.ylabel('Billions of Pesos')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('20_Working_Days_Forecast_April_2025.png')
    
    print("\n=== Working Days Forecast Results ===")
    print(daily_forecast_df)
    
    return daily_forecast_df

def main():
    """Main execution function"""
    try:
        # Load and prepare data
        df = load_and_prepare_data('./data.xlsx')
        
        # Explore data and analyze components
        df = explore_data(df)
        df = analyze_time_components(df)
        
        # Train and evaluate model
        model_fit, order, seasonal_order = train_and_evaluate_model(df)
        
        # Make daily forecasts
        daily_forecast_df = make_daily_forecast(df, model_fit, order, seasonal_order)
        
        print("\n=== Final Forecast ===")
        print(daily_forecast_df)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    main()