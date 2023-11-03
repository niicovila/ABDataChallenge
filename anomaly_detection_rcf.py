import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import rrcf
import heapq
import matplotlib.pyplot as plt
import os

#TODO: Change code so that we ask the user for an input DISTRICT, USE and TYPE OF ECONOMIC ACTIVITY

def main():
    # Create a directory to store results
    os.makedirs('anomaly_results', exist_ok=True)
    file_path = '/Users/nicolasvila/workplace/uni/ABData/abdataset1_barcelona_temp_seq.csv'
    original_data = pd.read_csv(file_path)
    # Assuming 'original_data' is your DataFrame
    original_data['Type of economic activity'] = original_data['Type of economic activity'].str.replace('*', '')

    # Now all '*' characters are removed from the 'Type of economic activity' column
    distinct_activities = original_data['Type of economic activity'].unique()

    # Write distinct values to a text file
    with open(os.path.join('', 'distinct_activities.txt'), 'w') as file:
        for i, activity in enumerate(distinct_activities):
            file.write(f'{i}. {activity}' + '\n')

    # Group the DataFrame by 'District', 'Use', and 'Type of economic activity'
    grouped_data = original_data.groupby(['District', 'Use', 'Type of economic activity'])
    
    unique_combinations = grouped_data[['Use', 'Type of economic activity']].nunique().shape[0]
    print("Number of Unique Combinations of Use and Type of Economic Activity:", unique_combinations)

    # Create a dictionary to store dataframes for each district and unique combination of 'Use' and 'Type of economic activity'
    district_activity_dataframes = {}

    # Iterate through grouped data and create separate dataframes for each district and activity combination
    for (district, use, activity), data in grouped_data:
        key = f'{district}_{use}_{activity}'  # Unique key for each combination
        district, use, activity = key.split('_')
        
        district_dir = os.path.join('anomaly_results', district)
        os.makedirs(district_dir, exist_ok=True)

        subdir = os.path.join(district_dir, f'{use}_{activity.replace('*', '')}')
        os.makedirs(subdir, exist_ok=True)
        
        district_activity_dataframes[key] = data.copy()
        district_activity_dataframes[key]['Date'] = pd.to_datetime(district_activity_dataframes[key][['Year', 'Month', 'Day']])
        district_activity_dataframes[key] = district_activity_dataframes[key].sort_values(by='Date', ascending=True)
        df_encoded = pd.get_dummies(district_activity_dataframes[key], columns=['District', 'Postcode', 'Use', 'Type of economic activity'])
        df_encoded = df_encoded.drop(columns=['Date'])

        data = df_encoded.values
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        # Set tree parameters
        num_trees = 10
        shingle_size = 5
        tree_size = 256

        # Create a forest of empty trees
        forest = []
        for _ in range(num_trees):
            tree = rrcf.RCTree()
            forest.append(tree)
            
        # Use the "shingle" generator to create rolling window
        points = rrcf.shingle(data, size=shingle_size)
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
                
        consumption = []
        anomaly_score = []
        for row in data:
            consumption.append(row[4])
        for i in avg_codisp.keys():
            anomaly_score.append(avg_codisp[i])

        # Why there is always a 9 row difference --> it is because of 'shingle size' in the model. 
        print(len(anomaly_score))
        print(len(district_activity_dataframes[key]))

        top_anomaly_positions = heapq.nlargest(5, range(len(anomaly_score)), key=lambda i: anomaly_score[i])
        df_encoded['Date'] = pd.to_datetime(district_activity_dataframes[key][['Year', 'Month', 'Day']])
    
        selected_rows = df_encoded.iloc[top_anomaly_positions]
        selected_rows.to_csv(os.path.join(subdir, 'top_anomalies_info.csv'), index=False)
    
        df_encoded.set_index('Date', inplace=True)
        # Save the distribution plot as an image
        plt.figure(figsize=(8, 6))
        plt.hist(anomaly_score, bins=30, edgecolor='black')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Anomaly Scores')
        plt.savefig(os.path.join(subdir, 'anomaly_score_distribution.png'))
        plt.close()
        
        # Save the time series plot as an image
        plt.figure(figsize=(12, 6))
        plt.semilogy(df_encoded.index, df_encoded['Consumption (L/Day)'], label='Log Consumption (L/Day)')
        plt.plot(df_encoded.iloc[:-shingle_size+1].index, anomaly_score, label='Anomaly Score')
        plt.xlabel('Time Step')
        plt.ylabel('Consumption (L/Day)')
        plt.title('Time-Dependent Data with Anomaly Scores')
        plt.legend()
        plt.savefig(os.path.join(subdir, 'time_series_plot.png'))
        plt.close()
        break


if __name__ == '__main__':
    main()