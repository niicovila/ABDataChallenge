import pandas as pd
from process_data import prepare_data, processed_main
from train import train
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, ExponentialLR, LinearLR, LambdaLR, ReduceLROnPlateau
from model import TranAD
from matplotlib.backends.backend_pdf import PdfPages
from train import save_model
from plot import plot_accuracies
from test import test
import json
import os
import numpy as np
import argparse

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def main(args):

    original_data = pd.read_csv(args.file_path)
    tensor, dataframes = prepare_data(original_data)
    
    index_feature = 1

    if args.train == 'True':
        print("Preparing the data ... ")
        train_tensor = tensor[0:math.ceil(len(tensor)*0.8)]
        trainO, dataloader, num_features = processed_main(train_tensor, args.num_window, index_feature)
        num_epochs = args.num_epochs
        num_window = args.num_window

        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        model = TranAD(feats=num_features, out_feat = 1, n_window=num_window)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min = 0)
        print("Training the model ... ")
        loss = train(model, device, num_epochs, num_window, scheduler, optimizer, dataloader, num_features, index_feature, print_loss=True)
        save_model(model, f'model_{num_epochs}.pth')
        print("Model was saved")
        plot_accuracies(loss, f'Transformer_AD')

    if args.test == 'True':
        model = torch.load(f'model_{num_epochs}.pth')
        dfs = []
        os.makedirs(os.path.join('plots', 'Transformer_AD'), exist_ok=True)
        pdf = PdfPages(f'plots/Transformer_AD/output_reconstruction_anomalies.pdf')
        for i, key in enumerate(dataframes):
            df = dataframes[key]
            test_tensor = df.values[np.newaxis, :, :]
            if test_tensor.shape[1] == 1460:
                testO, test_loader, num_features = processed_main(test_tensor,  args.num_window, index_feature)
                losses, anomalies = test(test_loader, model, num_features, device, index_feature, testO, pdf)
                df['anomaly'] = 0
                df.loc[anomalies[0], 'anomaly'] = 1
                dfs.append(df)
        pdf.close()
        total_df = pd.concat(dfs, axis=0, ignore_index=True)
        total_df.to_csv("anomalies.csv", index=False)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--json_config', type=str, required=True, help='Path to the JSON config file')
    # Parse the command line arguments
    args = parser.parse_args()

    if args.json_config:
        with open(args.json_config, 'r') as json_file:
            json_args = json.load(json_file)
    args = Config(**json_args)


    main(args)


