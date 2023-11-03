import pandas as pd

def read_data(file_path):
    file_path = '/Users/nicolasvila/workplace/uni/ABData/abdataset1_barcelona_temp_seq.csv'
    original_data = pd.read_csv(file_path)
    # Assuming 'original_data' is your DataFrame
    original_data['Type of economic activity'] = original_data['Type of economic activity'].str.replace('*', '')
    return original_data

if __name__ == "__main__":
    file_path = '/Users/nicolasvila/workplace/uni/ABData/abdataset1_barcelona_temp_seq.csv'
    read_data(file_path)