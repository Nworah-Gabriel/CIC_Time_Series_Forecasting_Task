import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

# Set random seed for reproducibility
np.random.seed(42)
sns.set_style("whitegrid")

def load_and_prepare_data(filepath):
    """Load and prepare the dataset"""
    df = pd.read_excel(filepath, parse_dates=['Date'], index_col='Date')
    df.columns = ['CIC']
    
    # Basic data validation
    print("\nMissing values before processing:", df.isnull().sum())
    if df.isnull().sum().any():
        print("Interpolating missing values...")
        df = df.interpolate(method='time')  # Time-based interpolation
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
    
    # Resample to weekly data and ensure no missing values
    df_weekly = df.resample('W').mean()
    df_weekly = df_weekly.interpolate(method='time')  # Handle any new missing values from resampling
    
    return df_weekly

def analyze_time_components(df_weekly):
    """Analyze time series components"""
    print("\n=== Time Series Decomposition ===")
    
    # Ensuring that no values are missing before decomposition
    if df_weekly.isnull().sum().any():
        print("Warning: Missing values found in weekly data. Interpolating...")
        df_weekly = df_weekly.interpolate(method='time')
    
    try:
        decomposition = seasonal_decompose(df_weekly['CIC'], model='additive', period=52, extrapolate_trend='freq')
        fig = decomposition.plot()
        fig.set_size_inches(14, 8)
        plt.tight_layout()
        plt.savefig('first_analized_time_series.png')
    except ValueError as e:
        print(f"Decomposition error: {str(e)}")
        return df_weekly
    
    # Stationarity check
    from statsmodels.tsa.stattools import adfuller
    adf_result = adfuller(df_weekly['CIC'].dropna())  # Ensure no NaNs for ADF test
    print(f'ADF Statistic: {adf_result[0]:.4f}')
    print(f'p-value: {adf_result[1]:.4f}')
    print('Critical Values:')
    for key, value in adf_result[4].items():
        print(f'\t{key}: {value:.4f}')
    
    # ACF/PACF plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(df_weekly['CIC'].dropna(), lags=52, ax=ax1)
    plot_pacf(df_weekly['CIC'].dropna(), lags=52, ax=ax2)
    plt.tight_layout()
    plt.savefig('second_analized_time_series.png')
    
    return df_weekly

def train_and_evaluate_model(df_weekly):
    """Train and evaluate SARIMA model"""
    print("\n=== Model Training ===")
    # Train-test split (last 52 weeks for testing)
    train = df_weekly.iloc[:-52]
    test = df_weekly.iloc[-52:]
    
    # Based on ACF/PACF analysis - these can be adjusted
    p, d, q = 1, 1, 1
    P, D, Q, m = 1, 1, 1, 52
    
    # Fit SARIMA model
    model = SARIMAX(train['CIC'],
                   order=(p, d, q),
                   seasonal_order=(P, D, Q, m),
                   enforce_stationarity=False,
                   enforce_invertibility=False)
    
    model_fit = model.fit(disp=False)
    print(model_fit.summary())
    
    # Forecast and evaluate
    forecast = model_fit.get_forecast(steps=len(test))
    forecast_values = forecast.predicted_mean
    
    # Calculate metrics
    mae = mean_absolute_error(test['CIC'], forecast_values)
    rmse = np.sqrt(mean_squared_error(test['CIC'], forecast_values))
    mape = np.mean(np.abs((test['CIC'] - forecast_values) / test['CIC'])) * 100
    
    print("\n=== Model Performance ===")
    print(f'MAE: {mae:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'MAPE: {mape:.2f}%')
    
    # Plot test vs forecast
    plt.figure(figsize=(14, 6))
    plt.plot(train.index[-104:], train['CIC'].tail(104), label='Training Data', color='navy')
    plt.plot(test.index, test['CIC'], label='Actual Values', color='darkgreen')
    plt.plot(test.index, forecast_values, label='Forecast', color='crimson')
    plt.fill_between(test.index,
                    forecast.conf_int().iloc[:, 0],
                    forecast.conf_int().iloc[:, 1],
                    color='pink', alpha=0.3)
    plt.title('Model Evaluation on Test Set (Last 52 Weeks)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Model_Evaluation_test_set.png')
    
    return model_fit, (p, d, q), (P, D, Q, m)

def make_forecasts(df_weekly, model_fit, order, seasonal_order):
    """Generate future forecasts"""
    print("\n=== Future Forecasting ===")
    forecast_steps = 20
    last_date = df_weekly.index[-1]
    
    # Fit final model on all data
    final_model = SARIMAX(df_weekly['CIC'],
                         order=order,
                         seasonal_order=seasonal_order,
                         enforce_stationarity=False,
                         enforce_invertibility=False)
    
    final_model_fit = final_model.fit(disp=False)
    
    # Generate forecast
    forecast = final_model_fit.get_forecast(steps=forecast_steps)
    forecast_dates = pd.date_range(start=last_date + timedelta(weeks=1),
                                 periods=forecast_steps,
                                 freq='W')
    
    forecast_df = pd.DataFrame({
        'Forecast': forecast.predicted_mean,
        'Lower_CI': forecast.conf_int().iloc[:, 0],
        'Upper_CI': forecast.conf_int().iloc[:, 1]
    }, index=forecast_dates)
    
    # Plot forecast
    plt.figure(figsize=(14, 6))
    df_weekly['CIC'].plot(label='Historical Data', color='navy')
    forecast_df['Forecast'].plot(label='Forecast', color='crimson')
    plt.fill_between(forecast_df.index,
                    forecast_df['Lower_CI'],
                    forecast_df['Upper_CI'],
                    color='pink', alpha=0.3)
    plt.title('20-Week Currency in Circulation Forecast')
    plt.ylabel('Billions of Pesos')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('20-Week_Currency_in_Circulation_Forecast.png')
    
    print("\n=== Forecast Results ===")
    print(forecast_df)
    
    return forecast_df

def main():
    """Main execution function"""
    try:
        # Load and prepare data
        df = load_and_prepare_data('./data.xlsx')
        
        # Explore data and analyze components
        df_weekly = explore_data(df)
        df_weekly = analyze_time_components(df_weekly)
        
        # Train and evaluate model
        model_fit, order, seasonal_order = train_and_evaluate_model(df_weekly)
        
        # Make forecasts
        forecast_df = make_forecasts(df_weekly, model_fit, order, seasonal_order)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    main()