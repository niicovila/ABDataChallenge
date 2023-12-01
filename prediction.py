import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from prophet import Prophet
from scipy.stats import zscore
import time
import os
import pickle
from tqdm import tqdm
import random
from reportlab.pdfgen import canvas
from multiprocessing import Pool, cpu_count

def run_prophet_parallel(args):
        df, key, zone = args
        df.rename(columns={"ConsLDayMeter":"y", "Date":"ds"}, inplace=True)

        # df['y'] = normalize(df[['y']])

        #  Calculate Z-scores for 'y'
        # df['y_zscore'] = zscore(df['y'])

        # # Define a threshold for Z-score to identify outliers (e.g., Z-score > 3 or < -3)
        # zscore_threshold = 2
        # # Identify and impute outliers with the average of neighboring points
        # window_size = 3  # Adjust the window size based on your preference
        # df['y_imputed'] = df['y'].copy()
        # # Find indices of outliers
        # outlier_indices = abs(df['y_zscore']) > zscore_threshold
        # # Iterate through outlier indices and impute with the average of neighboring points
        # for i in df.index[outlier_indices]:
        #     start_idx = max(0, i - window_size)
        #     end_idx = min(len(df), i + window_size + 1)
        #     avg_neighbor = df['y'].iloc[start_idx:end_idx].mean()
        #     df.at[i, 'y_imputed'] = avg_neighbor
        # # Drop the temporary 'y_zscore' column
        # df.drop(columns=['y_zscore'], inplace=True)

        # PLOT OUTLIER DETECTION AND IMPUTATION
        # plt.figure(figsize=(10, 6))
        # plt.plot(df['ds'], df['y'], label='Original y')
        # plt.plot(df['ds'], df['y_imputed'], label='Imputed y (Outliers Removed)', linestyle='--', marker='o', color='red')
        # plt.xlabel('Date')
        # plt.ylabel('y')
        # plt.legend()
        # plt.title('Original y vs Imputed y with Outliers Removed')
        # plt.show()

        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values(by='ds')
        split_index = len(df) - 7
        train_df = df.iloc[:split_index].copy()
        y_test = df.iloc[split_index:].copy()
        test_df = df.iloc[split_index:].drop('y', axis=1)

        model =  Prophet(seasonality_mode='additive', yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)

        for column in train_df.columns:
                if column not in ['y', 'ds', 'Muni']:
                        model.add_regressor(column)

        model.fit(train_df)
        forecast = model.predict(test_df)
        
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        y_test['ds'] = pd.to_datetime(y_test['ds'])
        df['ds'] = pd.to_datetime(df['ds'])
        forecast['yhat'] = np.where(forecast['yhat']<0, 0, forecast['yhat'])

        # wmae = mae_with_division(y_test['y'].values, forecast['yhat'])

        percentage_loss = np.mean(np.where(y_test['y'].values > forecast['yhat_upper'], 
                            np.abs((y_test['y'].values - forecast['yhat_upper']) / np.where(y_test['y'].values != 0, y_test['y'].values, 1)), 
                            np.where(y_test['y'].values < forecast['yhat_lower'], 
                            np.abs((y_test['y'].values - forecast['yhat_lower']) / np.where(y_test['y'].values != 0, y_test['y'].values, 1)), 
                            0))) * 100
        
        # print(f"MAPE: {percentage_loss:.2f} %")

        # PLOT forecast
        if random.random() < 0.2:
                fig, ax = plt.subplots(figsize=(10, 6))
                train_df['ds'] = pd.to_datetime(train_df['ds'])        
                ax.axvline(x=train_df['ds'].iloc[-1], linestyle='--', color='red', label='Train/Test Split')
                ax.plot(df['ds'][-50:], df['y'][-50:], label='True Data', color='blue', marker = 'o')
                ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='green', linestyle='--')

                ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightgreen', alpha=0.2, label='Prediction Interval')
                ax.set_xlabel('Date')
                ax.set_ylabel('Water Daily Cinsumption per meter')
                ax.legend()
                ax.set_title('Prophet Forecast vs True Data with Prediction Interval')
                
                save_folder = f'Plots/{zone}/Prophet'
                plt.savefig(os.path.join(save_folder, f'prophet_forecast_plot_{key}.png'))
                plt.close()

        return forecast, model, percentage_loss