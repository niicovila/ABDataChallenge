import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import rrcf
import argparse
import matplotlib.pyplot as plt
import os
from prepare_data import prepare_data_ae


def rrcf_anomalies(path):
    try:
    
        # Create a directory to store results
        os.makedirs('anomaly_results', exist_ok=True)
        original_data = pd.read_csv(path)
        original_data['Type of economic activity'] = original_data['Type of economic activity'].str.replace('*', '')
        
        tensor_data, dataframes = prepare_data_ae(original_data)
        
        # Iterate through grouped data and create separate dataframes for each district and activity combination
        uses = {'0': 'Commercial', '1': 'Industrial'}
        unique_activities = original_data['Type of economic activity'].unique()
        activities = {str(i): activity for i, activity in enumerate(unique_activities)}
        districts = list(range(1, 11))

        district = input("Enter district: ")
        use = input("Enter use (0 for Commercial, 1 for Industrial): ")
        activity = input("Enter activity index: ")
        
        percentile = 99
        
        # percentile = (input("Choose a percentile to include anomalies based on anomaly score distribution"))
        # if percentile.isnumeric():
        #     # Convert percentile to float
        #     percentile = float(percentile)
        # else:
        #     # Handle the case when percentile is not numeric (optional)
        #     print("Percentile wasn't numeric.")

        if use not in uses.keys() or activity not in activities.keys() or int(district) not in districts:
            print("Invalid input. Please provide valid use, activity, and district.")
        else:
            key = f'{district}_{uses[use]}_{activities[activity]}'
            print("Key:", key)
        print(len(dataframes.keys()))
        if key in dataframes.keys():
            print('Computing anomalies ...')
            scaler = StandardScaler()
            dataframe = dataframes[key]
            dataframe['Consumption per meter (L/Day)'] = scaler.fit_transform(dataframe[['Consumption per meter (L/Day)']])
            dataframe.reset_index(drop=True, inplace=True)
            
            district, use, activity = key.split('_')
            district_dir = os.path.join('anomaly_results', district)
            os.makedirs(district_dir, exist_ok=True)
            subdir = os.path.join(district_dir, f'{use}_{activity.replace("*", "")}')
            os.makedirs(subdir, exist_ok=True)
            dataframe.drop(columns=['Date'], inplace=True)
            
        
            # Set tree parameters
            data = dataframe.values
            num_trees = 100
            shingle_size = 10
            tree_size = 512
            time_series_length = len(dataframe)
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

        
            percentile_threshold = np.percentile(anomaly_score, percentile)
            selected_anomaly_positions = [position for position, score in enumerate(anomaly_score) if score > percentile_threshold]
        
            anomaly_windows = []
            for position in selected_anomaly_positions:
                start_position = max(0, position - shingle_size + 1)  # Ensure start position is within the bounds
                end_position = min(position + 1, time_series_length)  # Ensure end position is within the bounds
                anomaly_windows.append((start_position, end_position))

            dataframe['Date'] = pd.to_datetime(dataframe[['Year', 'Month', 'Day']])
        
            formatted_dates = '\n'.join([f"{dataframe['Date'].iloc[anomaly_windows[i][0]]} <--> {dataframe['Date'].iloc[anomaly_windows[i][1] - 1]}" for i in range(len(anomaly_windows))])
            result = f'''With the selected percentile, we found a total of {len(anomaly_windows)} for the selected district, use, and economic activity.\nThe anomalies were found between the following dates:\n{formatted_dates}'''
            with open(os.path.join(subdir, 'result.txt'), 'w') as file:
                file.write(result)
            

            # Save the distribution plot as an image
            plt.figure(figsize=(8, 6))
            plt.hist(anomaly_score, bins=30, edgecolor='black')
            plt.axvline(x=percentile_threshold, color='red', linestyle='dashed', linewidth=2, label='95th Percentile')
            plt.xlabel('Anomaly Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Anomaly Scores')
            plt.savefig(os.path.join(subdir, 'anomaly_score_distribution.png'))
            plt.close()
            
            # Save the time series plot as an image
            plt.figure(figsize=(12, 6))
            plt.plot(dataframe['Date'], dataframe['Consumption per meter (L/Day)'], label='Consumption per meter (L/Day)')
        
            for window in anomaly_windows:
                plt.axvspan(dataframe['Date'].iloc[window[0]], dataframe['Date'].iloc[window[1] - 1], facecolor='red', alpha=0.3)

            plt.xlabel('Time Step')
            plt.ylabel('Consumption (L/Day)')
            plt.title('Time-Dependent Data with Anomaly Scores and Detected Anomalies')
            plt.legend()
            plt.savefig(os.path.join(subdir, 'time_series_plot_with_anomalies.png'))
            plt.show()
            plt.close()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data anomalies based on percentile.')
    parser.add_argument('path', type=str, help='Path to the CSV file')

    args = parser.parse_args()
    rrcf_anomalies(args.path)