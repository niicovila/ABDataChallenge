import torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os

from process_data import convert_to_windows
from plot import plotter

def test(test_loader, model, num_features, device, feature_index, trainO, pdf):
    torch.zero_grad = True
    model.eval()
    l = nn.HuberLoss(reduction = 'none')
    losses = []
    anomalies_list = []
    
    for d, _ in test_loader:
        d = d.to(device)
 
        local_bs = d.shape[0]
        window = d.permute(1, 0, 2)
        elem = window[-1, :, :].view(1, local_bs, num_features)
        elem_feature = elem[:, :, 0].unsqueeze(2) # Change 0 to the index of the feature

        z = model(window, elem_feature)
        if isinstance(z, tuple): z = z[1]
        loss = l(z, elem_feature)[0]

        loss, y_pred = loss.detach().cpu().numpy(), z.detach().cpu().numpy()[0]
        testO = torch.roll(trainO, 1, 0)
        y_true = testO[:, feature_index].unsqueeze(1)
        plotter(y_true, y_pred, loss, pdf)

        percentile_threshold = np.percentile(loss, 97)
        anomalies = np.where(loss > percentile_threshold)[0]
        anomalies_list.append(anomalies)

        losses.append((loss, y_pred, y_true))

    return losses, anomalies_list

def main(args):
    tensor_v2 = args.tensor
    tensor_single_test = tensor_v2[66]
    train_loader = DataLoader(tensor_single_test, batch_size=tensor_single_test.shape[0])
    trainD = next(iter(train_loader))
    trainO = trainD
    trainD = convert_to_windows(trainD)
    data_x = trainD.float(); dataset = TensorDataset(data_x, data_x)
    bs = 1460
    test_loader = DataLoader(dataset, batch_size = bs)

if __name__ == "__main__":
    args = 0
    main(args)