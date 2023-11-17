import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, ExponentialLR, LinearLR, LambdaLR, ReduceLROnPlateau
from model import TranAD
from plot import plot_accuracies
import numpy as np

def save_model(model, filepath):
    torch.save(model, filepath)

def train(model, device, num_epochs, num_window, scheduler, optimizer, dataloader, num_features, index_feature, print_loss):
    model.to(device)
    accuracy_list = []
    for epoch in range(num_epochs):
        l1s, l2s = [], []
        n = epoch + 1; w_size = num_window
        l = nn.HuberLoss(reduction = 'none')
        for d, _  in dataloader:
                d = d.to(device)
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, num_features) #[:, :, 0].unsqueeze(2)
                elem_feature = elem[:, :, index_feature].unsqueeze(2) # Change 0 to the index of the feature
                z = model(window, elem_feature)
                # Select the feature from elem for which you want to compute the loss

                l1 = l(z, elem_feature) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem_feature) + (1 - 1/n) * l(z[1], elem_feature)
                if isinstance(z, tuple): z = z[1]
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
        scheduler.step()
        if print_loss: print(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
        lossT, lr = np.mean(l1s), optimizer.param_groups[0]['lr']
        accuracy_list.append((lossT, lr))
    return accuracy_list

def main(args):
    dataloader = args.dataloader
    num_epochs = args.num_epochs
    num_window = args.num_window
    num_features = args.num_features

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = TranAD(feats=num_features, out_feat = 1)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min = 0)

    accuracy_list = train(model, device, num_epochs, num_window, scheduler, optimizer, dataloader, print_loss=False)
    save_model(model)
    plot_accuracies(accuracy_list, f'TransAD')
    return model