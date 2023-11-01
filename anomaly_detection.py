#%%
import pandas as pd
import numpy as np

def read_data(file_path):
    df = pd.read_csv(file_path)
    return df

def prepare_data(df):

    return 0
#%%
def main():
    #%%
    file_path = '/Users/nicolasvila/workplace/uni/ABData/abdataset1_barcelona_temp_seq.csv'
    original_data = read_data(file_path)
    #%%
    # Group the DataFrame by 'District', 'Use', and 'Type of economic activity'
    grouped_data = original_data.groupby(['District', 'Use', 'Type of economic activity'])

    # Create a dictionary to store dataframes for each district and unique combination of 'Use' and 'Type of economic activity'
    district_activity_dataframes = {}

    # Iterate through grouped data and create separate dataframes for each district and activity combination
    for (district, use, activity), data in grouped_data:
        key = f'{district}_{use}_{activity}'  # Unique key for each combination
        print(key)
        district_activity_dataframes[key] = data.copy()
        district_activity_dataframes[key]['Date'] = pd.to_datetime(district_activity_dataframes[key][['Year', 'Month', 'Day']])
        district_activity_dataframes[key] = district_activity_dataframes[key].sort_values(by='Date', ascending=True)
    #%%
    district_activity_dataframes['1_Commercial_ * DIY items'].head()
    district_activity_dataframes['1_Commercial_ * DIY items']['Date'] = pd.to_datetime(district_activity_dataframes['1_Commercial_ * DIY items'][['Year', 'Month', 'Day']])
    district_activity_dataframes['1_Commercial_ * DIY items'] = district_activity_dataframes['1_Commercial_ * DIY items'].sort_values(by='Date', ascending=True)
    #%%
    district_activity_dataframes['1_Commercial_ * DIY items'].head()
    #%%
    df_sorted =  district_activity_dataframes['1_Commercial_ * DIY items']
    #%%
    df_encoded = pd.get_dummies(df_sorted, columns=['District', 'Postcode', 'Use', 'Type of economic activity'])
    #%%
    df_encoded.head()
    #%%
    data = df_encoded.values
    #%%
    type(data)
    data[0]
    #%%
    features = ['Year', 'Month', 'Day' ,'Number of meters', 'Consumption (L/Day)', 'Precipitaciones (mm)', 'T_max (C)', 'T_min (C)', 'spi_12', 'spi_9', 'scpdsi', 'District_1',	'Postcode_8001',	'Postcode_8002',	'Postcode_8003',	'Postcode_8039','Use_Commercial',	'Type of economic activity_ * DIY items']
    from sklearn.preprocessing import StandardScaler
    import rrcf

    # Set tree parameters
    num_trees = 40
    shingle_size = 4
    tree_size = 256

    # Create a forest of empty trees
    forest = []
    for _ in range(num_trees):
        tree = rrcf.RCTree()
        forest.append(tree)
        
    # Use the "shingle" generator to create rolling window
    points = rrcf.shingle(data, size=shingle_size)
    #%%
    # Create a dict to store anomaly score of each point
    avg_codisp = {}

    # For each shingle...
    for index, point in enumerate(points):
        # For each tree in the forest...
        for tree in forest:
            # If tree is above permitted size...
            if len(tree.leaves) > tree_size:
                # Drop the oldest point (FIFO)
                tree.forget_point(index - tree_size)
            # Insert the new point into the tree
            tree.insert_point(point, index=index)
            # Compute codisp on the new point...
            new_codisp = tree.codisp(index)
            # And take the average over all trees
            if not index in avg_codisp:
                avg_codisp[index] = 0
            avg_codisp[index] += new_codisp / num_trees
    #%%
    consumption = []
    anomaly_score = []
    for row in data:
        consumption.append(row[4])
    for key in avg_codisp.keys():
        anomaly_score.append(avg_codisp[key])

    import heapq

    # Your existing code here...

    # Find the indices of the top 5 anomalies with highest scores
    top_anomaly_positions = heapq.nlargest(5, range(len(anomaly_score)), key=lambda i: anomaly_score[i])

    # Print the positions of the top 5 highest anomalies
    print("Positions of Top 5 Highest Anomalies:")
    print(top_anomaly_positions)
    for index in top_anomaly_positions:
        print(anomaly_score[index])

    #%%
    selected_rows = df_encoded.iloc[top_anomaly_positions]
    selected_rows.head()
    #%%
    import matplotlib.pyplot as plt
    # Plot the time-dependent data along with the corresponding anomaly scores
    plt.figure(figsize=(12, 6))
    plt.plot(consumption, label='Consumption (L/Day)')
    #plt.plot(anomaly_score, label='Anomaly Score')
    plt.xlabel('Time Step')
    plt.ylabel('Consumption (L/Day)')
    plt.title('Time-Dependent Data with Anomaly Scores')
    plt.legend()
    plt.show()
    #%%
    plt.figure(figsize=(8, 6))
    plt.hist(anomaly_score, bins=30, edgecolor='black')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Anomaly Scores')
    plt.show()

    #%%
    anomaly_scores = []
    indices = []
    for index, point in enumerate(points):
        # Your existing processing code...

        # Store the average anomaly score and its index
        anomaly_scores.append(avg_codisp[index])
        indices.append(index)
    print(anomaly_scores)
    #%%
    #%%
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



    #%%
    import pandas as pd

    # Assuming df_sorted is your DataFrame sorted by 'Date' in ascending order
    # Assuming 'Date' is a datetime column
    # Reset the index to use row indices for accessing the data
    df_sorted.reset_index(drop=True, inplace=True)

    # Initialize an empty list to store rows of the new DataFrame
    new_rows = []

    # Iterate through the DataFrame to create rows for the new DataFrame
    for i in range(10, len(df_sorted)):
        current_row = df_sorted.iloc[i]
        past_rows = df_sorted.iloc[i-10:i]
        
        # Extract data and dates for the last 10 days for each variable
        consumption_data = [(past_rows.at[i - j, 'Consumption (L/Day)'], past_rows.at[i - j, 'Date']) for j in range(10, 0, -1)]
        
        print(consumption_data)
        break
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

    # # Create a new DataFrame from the list of rows
    # new_df = pd.DataFrame(new_rows)

    # # Now, new_df contains the pivoted data with columns representing the past 10 days' data for each variable
    # print(new_df)
    #%%
if __name__ == '__main__':
    main()