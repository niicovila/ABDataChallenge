import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import rrcf
import heapq
import matplotlib.pyplot as plt
import os


def main():
    # Create a directory to store results
    os.makedirs('anomaly_results', exist_ok=True)
    file_path = '/Users/nicolasvila/workplace/uni/ABData/abdataset1_barcelona_temp_seq.csv'
    original_data = pd.read_csv(file_path)
    # Assuming 'original_data' is your DataFrame
    original_data['Type of economic activity'] = original_data['Type of economic activity'].str.replace('*', '')

   
        # Print the positions of the top 5 highest anomalies
        # print("Positions of Top 5 Highest Anomalies:")
        # print(top_anomaly_positions)
        # for index in top_anomaly_positions:
        #     print(anomaly_score[index])

        # selected_rows = df_encoded.iloc[top_anomaly_positions]
        # selected_rows.head()

        # plt.figure(figsize=(12, 6))
        # plt.plot(consumption, label='Consumption (L/Day)')
        # plt.plot(anomaly_score, label='Anomaly Score')
        # plt.xlabel('Time Step')
        # plt.ylabel('Consumption (L/Day)')
        # plt.title('Time-Dependent Data with Anomaly Scores')
        # plt.legend()
        # plt.show()

        # plt.figure(figsize=(8, 6))
        # plt.hist(anomaly_score, bins=30, edgecolor='black')
        # plt.xlabel('Anomaly Score')
        # plt.ylabel('Frequency')
        # plt.title('Distribution of Anomaly Scores')
        # plt.show()


    # # Standardize features
    # scaler = StandardScaler()
    # scaled_features = scaler.fit_transform(df_encoded[features])

    # # Number of trees in the forest (you can adjust this based on your dataset)
    # num_trees = 100

    # # Build the RRCF model
    # forest = []
    # for _ in range(num_trees):
    #     tree.insert_points(scaled_features)
    #     forest.append(tree)

    # # Calculate anomaly scores
    # anomaly_scores = []
    # for tree in forest:
    #     # Use the negative logarithm of the raw score as the anomaly score
    #     score = -tree.log_score(scaled_features)
    #     anomaly_scores.append(score)

    # # Average anomaly scores from all trees
    # average_scores = pd.DataFrame(anomaly_scores).mean(axis=0)

    # # Set a threshold for anomaly detection (you may adjust this based on your dataset)
    # threshold = 0.1  # Example threshold value

    # # Print dates where there was an anomaly
    # anomaly_dates = df_encoded.loc[average_scores > threshold, ['Year', 'Month', 'Day']]
    # print(anomaly_dates)

    # print(len(anomaly_dates))

################################################################################################
     ################ CODE TO REORDER TIME DATA ####################
################################################################################################
    # import pandas as pd

    # # Assuming df_sorted is your DataFrame sorted by 'Date' in ascending order
    # # Assuming 'Date' is a datetime column
    # # Reset the index to use row indices for accessing the data
    # df_sorted.reset_index(drop=True, inplace=True)

    # # Initialize an empty list to store rows of the new DataFrame
    # new_rows = []

    # # Iterate through the DataFrame to create rows for the new DataFrame
    # for i in range(10, len(df_sorted)):
    #     current_row = df_sorted.iloc[i]
    #     past_rows = df_sorted.iloc[i-10:i]
        
    #     # Extract data and dates for the last 10 days for each variable
    #     consumption_data = [(past_rows.at[i - j, 'Consumption (L/Day)'], past_rows.at[i - j, 'Date']) for j in range(10, 0, -1)]
        
    #     print(consumption_data)
    #     break
    ################################################################################################################################################

################################################################################################################################################


    #     precipitation_data = [(past_rows.at[i - j, 'Precipitaciones (mm)'], past_rows.at[i - j, 'Date']) for j in range(1, 11)]
    #     tmax_data = [(past_rows.at[i - j, 'T_max (C)'], past_rows.at[i - j, 'Date']) for j in range(1, 11)]
    #     tmin_data = [(past_rows.at[i - j, 'T_min (C)'], past_rows.at[i - j, 'Date']) for j in range(1, 11)]
    #     spi12_data = [(past_rows.at[i - j, 'spi_12'], past_rows.at[i - j, 'Date']) for j in range(1, 11)]
    #     spi9_data = [(past_rows.at[i - j, 'spi_9'], past_rows.at[i - j, 'Date']) for j in range(1, 11)]
    #     scpdsi_data = [(past_rows.at[i - j, 'scpdsi'], past_rows.at[i - j, 'Date']) for j in range(1, 11)]

    #     # Create a new row with the required columns
    #     new_row = {
    #         'District': current_row['District'],
    #         'Postcode': current_row['Postcode'],
    #         'Use': current_row['Use'],
    #         'Type of economic activity': current_row['Type of economic activity'],
    #         'Number of meters': current_row['Number of meters'],
    #         'Consumption (L/Day)-10': consumption_data[0],
    #         'Consumption (L/Day)-9': consumption_data[1],
    #         'Consumption (L/Day)-8': consumption_data[2],
    #         'Consumption (L/Day)-7': consumption_data[3],
    #         'Consumption (L/Day)-6': consumption_data[4],
    #         'Consumption (L/Day)-5': consumption_data[5],
    #         'Consumption (L/Day)-4': consumption_data[6],
    #         'Consumption (L/Day)-3': consumption_data[7],
    #         'Consumption (L/Day)-2': consumption_data[8],
    #         'Consumption (L/Day)-1': consumption_data[9],
    #         'Precipitaciones (mm)-10': precipitation_data[0],
    #         'Precipitaciones (mm)-9': precipitation_data[1],
    #         'Precipitaciones (mm)-8': precipitation_data[2],
    #         'Precipitaciones (mm)-7': precipitation_data[3],
    #         'Precipitaciones (mm)-6': precipitation_data[4],
    #         'Precipitaciones (mm)-5': precipitation_data[5],
    #         'Precipitaciones (mm)-4': precipitation_data[6],
    #         'Precipitaciones (mm)-3': precipitation_data[7],
    #         'Precipitaciones (mm)-2': precipitation_data[8],
    #         'Precipitaciones (mm)-1': precipitation_data[9],
    #         'T_max (C)-10': tmax_data[0],
    #         'T_max (C)-9': tmax_data[1],
    #         'T_max (C)-8': tmax_data[2],
    #         'T_max (C)-7': tmax_data[3],
    #         'T_max (C)-6': tmax_data[4],
    #         'T_max (C)-5': tmax_data[5],
    #         'T_max (C)-4': tmax_data[6],
    #         'T_max (C)-3': tmax_data[7],
    #         'T_max (C)-2': tmax_data[8],
    #         'T_max (C)-1': tmax_data[9],
    #         'T_min (C)-10': tmin_data[0],
    #         'T_min (C)-9': tmin_data[1],
    #         'T_min (C)-8': tmin_data[2],
    #         'T_min (C)-7': tmin_data[3],
    #         'T_min (C)-6': tmin_data[4],
    #         'T_min (C)-5': tmin_data[5],
    #         'T_min (C)-4': tmin_data[6],
    #         'T_min (C)-3': tmin_data[7],
    #         'T_min (C)-2': tmin_data[8],
    #         'T_min (C)-1': tmin_data[9],
    #         'spi_12-10': spi12_data[0],
    #         'spi_12-9': spi12_data[1],
    #         'spi_12-8': spi12_data[2],
    #         'spi_12-7': spi12_data[3],
    #         'spi_12-6': spi12_data[4],
    #         'spi_12-5': spi12_data[5],
    #         'spi_12-4': spi12_data[6],
    #         'spi_12-3': spi12_data[7],
    #         'spi_12-2': spi12_data[8],
    #         'spi_12-1': spi12_data[9],
    #         'spi_9-10': spi9_data[0],
    #         'spi_9-9': spi9_data[1],
    #         'spi_9-8': spi9_data[2],
    #         'spi_9-7': spi9_data[3],
    #         'spi_9-6': spi9_data[4],
    #         'spi_9-5': spi9_data[5],
    #         'spi_9-4': spi9_data[6],
    #         'spi_9-3': spi9_data[7],
    #         'spi_9-2': spi9_data[8],
    #         'spi_9-1': spi9_data[9],
    #         'scpdsi-10': scpdsi_data[0],
    #         'scpdsi-9': scpdsi_data[1],
    #         'scpdsi-8': scpdsi_data[2],
    #         'scpdsi-7': scpdsi_data[3],
    #         'scpdsi-6': scpdsi_data[4],
    #         'scpdsi-5': scpdsi_data[5],
    #         'scpdsi-4': scpdsi_data[6],
    #         'scpdsi-3': scpdsi_data[7],
    #         'scpdsi-2': scpdsi_data[8],
    #         'scpdsi-1': scpdsi_data[9]
    #     }
        
    #     # Add the new row to the list
    #     new_rows.append(new_row)


if __name__ == '__main__':
    main()