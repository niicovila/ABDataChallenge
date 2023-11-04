
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#TODO: Change code so that we ask the user for an input DISTRICT, USE and TYPE OF ECONOMIC ACTIVITY

def prepare_data_ae(original_data):
    distinct_activities = original_data['Type of economic activity'].unique()
    original_data['Consumption per meter (L/Day)'] = original_data['Consumption (L/Day)'] /original_data['Number of meters']
    columns_to_drop = ['Postcode', 'Consumption (L/Day)', 'Number of meters']
    original_data = original_data.drop(columns=columns_to_drop)

    # Group the DataFrame by 'District', 'Use', and 'Type of economic activity'
    grouped_data = original_data.groupby(['District', 'Use', 'Type of economic activity'])

    district_activity_dataframes = {}
    reshaped_data_list = []
    dataframes = []
    for (district, use, activity), data in grouped_data:
        key = f'{district}_{use}_{activity}'  
        district, use, activity = key.split('_')
    
        district_activity_dataframes[key] = data.copy()
        district_activity_dataframes[key]['Date'] = pd.to_datetime(district_activity_dataframes[key][['Year', 'Month', 'Day']])
        district_activity_dataframes[key] = district_activity_dataframes[key].sort_values(by='Date', ascending=True)
       
        label_encoder = LabelEncoder()

        df_encoded = district_activity_dataframes[key]
       
        df_encoded['District'] = label_encoder.fit_transform(df_encoded['District'])
        df_encoded['Use'] = label_encoder.fit_transform(df_encoded['Use'])
        df_encoded['Type of economic activity'] = label_encoder.fit_transform(df_encoded['Type of economic activity'])
        
        df_encoded.reset_index(inplace=True)
        df_encoded = df_encoded.groupby('Date').agg({
            'District': 'first',
            'Year': 'first',
            'Month': 'first',
            'Day':'first',
            'Use': 'first',
            'Type of economic activity':'first',
            'Consumption per meter (L/Day)': 'mean',
            'Precipitaciones (mm)': 'mean',  
            'Date' : 'first', 
            'T_max (C)': 'mean',
            'T_min (C)': 'mean',  
            'spi_12': 'mean',
            'spi_9': 'mean',
            'scpdsi': 'mean'
        }).reset_index(drop=True)
   
        data = df_encoded.values
        district_activity_dataframes[key] = df_encoded
        if (data.shape[0]) == 1460 :
            reshaped_data_list.append(data)

    reshaped_data = np.stack(reshaped_data_list, axis=0)

    return reshaped_data, district_activity_dataframes

if __name__ == '__main__':
    path = '/Users/nicolasvila/workplace/uni/ABData/abdataset1_barcelona_temp_seq.csv'
    prepare_data_ae(path)
