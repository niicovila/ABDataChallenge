import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def prepare_data_ae(original_data):
    distinct_activities = original_data['Type of economic activity'].unique()
    original_data['Consumption per meter (L/Day)'] = original_data['Consumption (L/Day)'] /original_data['Number of meters']
    columns_to_drop = ['Consumption (L/Day)', 'Number of meters']
    original_data = original_data.drop(columns=columns_to_drop)

    # Group the DataFrame by 'District', 'Use', and 'Type of economic activity'
    grouped_data = original_data.groupby(['Postcode', 'Use', 'Type of economic activity'])

    postcode_activity_dataframes = {}
    reshaped_data_list = []
    for (postcode, use, activity), data in grouped_data:
        key = f'{postcode}_{use}_{activity}'  
        postcode, use, activity = key.split('_')
    
        postcode_activity_dataframes[key] = data.copy()
        #postcode_activity_dataframes[key]['Date'] = pd.to_datetime(postcode_activity_dataframes[key][['Year', 'Month', 'Day']])
        postcode_activity_dataframes[key] = postcode_activity_dataframes[key].sort_values(by='Date', ascending=True)
       
        label_encoder = LabelEncoder()

        df_encoded = postcode_activity_dataframes[key]
       
        df_encoded['Postcode'] = label_encoder.fit_transform(df_encoded['Postcode'])
        df_encoded['Use'] = label_encoder.fit_transform(df_encoded['Use'])
        df_encoded['Type of economic activity'] = label_encoder.fit_transform(df_encoded['Type of economic activity'])
        
        df_encoded.reset_index(inplace=True)
        df_encoded = df_encoded.groupby('Date').agg({
            'Postcode': 'first',
            'Use': 'first',
            'Type of economic activity':'first',
            'Consumption per meter (L/Day)': 'max',
            # 'Precipitaciones (mm)': 'mean',  
            # 
            'Date' : 'first',
            # 'T_max (C)': 'mean',
            # 'T_min (C)': 'mean',  
            'spi_12': 'mean',
            'spi_9': 'mean',
            'scpdsi': 'mean'
        }).reset_index(drop=True)
   
        data = df_encoded.values
        postcode_activity_dataframes[key] = df_encoded
        if (data.shape[0]) == 1460 :
            reshaped_data_list.append(data)

    reshaped_data = np.stack(reshaped_data_list, axis=0)

    return reshaped_data, postcode_activity_dataframes

if __name__ == '__main__':
    path = '/Users/nicolasvila/workplace/uni/ABData/abdataset1_barcelona_temp_seq.csv'
    prepare_data_ae(path)
