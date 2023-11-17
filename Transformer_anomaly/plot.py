import torch
import numpy as np
from matplotlib import pyplot as plt
import torch
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
import os
torch.manual_seed(1)

def plot_accuracies(accuracy_list, folder):
	os.makedirs(f'plots/{folder}/', exist_ok=True)
	trainAcc = [i[0] for i in accuracy_list]
	lrs = [i[1] for i in accuracy_list]
	plt.xlabel('Epochs')
	plt.ylabel('Average Training Loss')
	plt.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', linewidth=1, linestyle='-', marker='.')
	plt.twinx()
	plt.plot(range(len(lrs)), lrs, label='Learning Rate', color='r', linewidth=1, linestyle='--', marker='.')
	plt.savefig(f'plots/{folder}/training-graph.pdf')
	plt.clf()
def smooth(y, box_pts=1):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plotter(name, y_true, y_pred, ascore, i):
	if 'TranAD' in name: y_true = torch.roll(y_true, 1, 0)
	os.makedirs(os.path.join('plots', name), exist_ok=True)
	pdf = PdfPages(f'plots/{name}/output_reconstruction_anomalies.pdf')
	for dim in range(y_true.shape[1]):
		y_t = y_true[:, dim]
		y_p = y_pred[:, dim]
		a_s = ascore[:, dim]
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
		ax1.set_ylabel('Value')
		ax1.set_title(f'Dimension = {dim}')
		# if dim == 0: np.save(f'true{dim}.npy', y_t); np.save(f'pred{dim}.npy', y_p); np.save(f'ascore{dim}.npy', a_s)
		ax1.plot(smooth(y_t), linewidth=0.2, label='True')
		ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.3, label='Predicted')
		ax3 = ax1.twinx()
		if dim == 0: ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))

		percentile_threshold = np.percentile(a_s, 97)
		anomalies = np.where(a_s > percentile_threshold)[0]

		ax2.plot(smooth(a_s), linewidth=0.2, color='g')
		ax2.scatter(anomalies, a_s[anomalies], color='red') # anomalies are marked with red
		ax2.set_xlabel('Timestamp')
		ax2.set_ylabel('Anomaly Score')
		pdf.savefig(fig)
		plt.close()
	pdf.close()