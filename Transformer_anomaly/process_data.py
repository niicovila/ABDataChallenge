import torch
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

torch.manual_seed(1)


def prepare_data(original_data):
    distinct_activities = original_data['EconAct'].unique()
    original_data['ConsLDayMeter'] = original_data['ConsLDay'] /original_data['NumMeters']
    columns_to_drop = ['Muni', 'ConsLDay', 'NumMeters']
    original_data = original_data.drop(columns=columns_to_drop)

    # Group the DataFrame by 'District', 'Use', and 'Type of economic activity'
    grouped_data = original_data.groupby(['PCode', 'Use', 'EconAct'])
    
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

        df_encoded['PCode'] = label_encoder.fit_transform(df_encoded['PCode'])
        df_encoded['Use'] = label_encoder.fit_transform(df_encoded['Use'])
        df_encoded['EconAct'] = label_encoder.fit_transform(df_encoded['EconAct'])

        df_encoded.reset_index(inplace=True)
        df_encoded = df_encoded.groupby('Date').agg({
            'PCode': 'first',
            'Year': 'first',
            'Month': 'first',
            'Day':'first',
            'Use': 'first',
            'EconAct':'first',
            'ConsLDayMeter': 'max',
            'Precip': 'mean',  
            'Date' : 'first', 
            'T_max': 'mean',
            'T_min': 'mean',  
            'spi_12': 'mean',
            'spi_9': 'mean',
            'scpdsi': 'mean',
            'Dist': 'first',
            'WDay': 'first',
            'Day_Yr': 'first',
            'Price_m2': 'first',
            'ShopAge': 'first',
            'BldgAge': 'first',
            'Pop': 'first',
            'NumHH': 'first',
            'AvgPPHH': 'first',
            'NetIncPP': 'first',
            'NetIncHH': 'first',
            'IncPC': 'first',
            'MedIncPC': 'first',
            'GrossIncPP': 'first',
            'GrossIncHH': 'first',
            'Pop0_14': 'first',
            'Pop15_64': 'first',
            'Pop65_': 'first',
            'MedAge': 'first',
            'ForBorn': 'first',
            'Foreigners': 'first',
            'BirthRate': 'first',
            'DeathRate': 'first',
            'LifeExp': 'first',
            'NumChild': 'first',
            'UnempRate': 'first',
            'EmpRate': 'first',
            'ActRate': 'first',
            'ServEmp': 'first',
            'IndEmp': 'first',
            'Area': 'first',
            'ChildCare': 'first'
        }).reset_index(drop=True)

        data = df_encoded.values
        postcode_activity_dataframes[key] = df_encoded
        if (data.shape[0]) == 1460 :
            reshaped_data_list.append(data)

    reshaped_data = np.stack(reshaped_data_list, axis=0)

    return reshaped_data, postcode_activity_dataframes

def convert_to_windows(data, window_size):
	windows = []; w_size = window_size
	for i, g in enumerate(data):
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w)
	return torch.stack(windows)

def processed_main(tensor, window_size, index_feature):
    # delete non-continuous data
    delete_tables = [16, 15, 14, 8, 5, 4, 2, 1, 0]
    
    #delete date string
    # delete_tables = [8]
    for col in delete_tables:
        tensor = np.delete(tensor, col, axis=-1)

    tensor = tensor.astype(float)
    scaler = StandardScaler()
    num_features = len(tensor[0][0])
    for col in list(range(num_features)):
        if col != index_feature:
                reshaped_feature = tensor[:, :, col].reshape(-1, 1)
                # Fit and transform with the scaler
                normalized_feature = scaler.fit_transform(reshaped_feature)
                # Reshape the normalized feature and update the original tensor
                tensor[:, :, col] = normalized_feature.reshape(tensor.shape[:2])
    bs = 1460
    tensor = torch.tensor(tensor[0]).unsqueeze(0)
    tensor = tensor.reshape(tensor.shape[0]*tensor.shape[1], tensor.shape[2])
    #tensor = torch.tensor(tensor)
    trainO = tensor
    trainD = convert_to_windows(tensor, window_size)

    data_x = trainD.float(); dataset = TensorDataset(data_x, data_x)
    dataloader = DataLoader(dataset, batch_size = bs)

    return trainO, dataloader, num_features
