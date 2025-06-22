import numpy as np
import torch
from test import calculate_total_score
import os
import matplotlib.pyplot as plt

#models_dir = ['./dev_8/general', './dev_2/general', f'./dev_2/general_unet', f'./dev_3/general', f'./dev_4/general',
#              f'./dev_5/general', f'./dev_6/general',  f'./dev_7/general']
models_dir = 'dev_total/bearing'
machineFolders = os.listdir('./DCASE2024')
all_normal_scores = []
all_anomalous_scores = []

for m in machineFolders:
    datasetType, machineType = m.split('_')
    # previous_state = torch.load(f"{models_dir}/train_state_dict_CAE_normed.pt")
    auc = np.load(f'./{datasetType}_norm/{machineType}/auc_values.npy').tolist()
    pauc = np.load(f'./{datasetType}_norm/{machineType}/pauc_values.npy').tolist()

    normal_scores = np.load(f'./{datasetType}_norm/{machineType}/all_normal_scores.npy').tolist()
    anomalous_scores = np.load(f'./{datasetType}_norm/{machineType}/all_anomalous_scores.npy').tolist()
    mean = calculate_total_score(auc, pauc)
    fig, axs = plt.subplots(2, 1)
    fig.suptitle("Distributia scorurilor pentru esantioanele normale & anormale - "+m)
    axs[0].hist(normal_scores, density=False, bins=30)
    axs[0].set_ylabel('Numar de esantioane')
    axs[1].hist(anomalous_scores, density=False, bins=30)
    axs[1].set_ylabel('Numar de esantioane')
    axs[1].set_xlabel('Scor')
    plt.show()

