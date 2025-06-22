import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import random
import numpy as np
from utils import WindowedSpectrogramDataset, DCASE2024MachineDataset, DCASE2024Dataset, stratified_split, printDataInfo
from model import ConvAutoencoder
from training import training_model
from test_windows import test_model

torch.manual_seed(19)
random.seed(19)
np.random.seed(19)

"""# Information extracted from the task description article

Each recording is a single-channel audio with a duration of 6 to 18 seconds and a sampling rate of 16 kHz. In the Dev they are mainly 10 seconds.

# Rules:

1. Cannot use the test data from the development dataset as support for hyperparameter tuning (different machines)
2. Cannot rely on the id's of the machines found in the dev set | 1 + 2 = First Shot Problem
3. Anomaly Score to calculate = if after certain threshold is an anomaly
4. Source = a lot of data | Target = same machine different conditions, little data - domain adaptation techniques if info is known
5. Info is unknown so must develop domain generalization techniques so it matches test set too

"Detecting anomalies from different domains with a single threshold. These techniques, unlike domain adaptation
techniques, do not require detection of domain shifts or adaptation of the model during the testing phase."

# Evaluation Metric:

**Area Under Reciever Characteristics Curve** to assess the overall detection performance.

**Partial AUC (pAUC)** was utilized to measure performance in a low false-positive rate [0, p], p=0.1.

In domain generalization task, the AUC for each domain and pAUC for each section are calculated.

Notably, all eight teams that outperformed the
baselines in the official scores also surpassed the baselines in the
harmonic mean of the AUCs in the target domain.

https://arxiv.org/pdf/2305.07828.pdf


"""

# Create the test dataset and use the statistics computed from the training dataset
# dev_test_dataset = DCASE2024Dataset('./DCASE2024', ('dev', 'test'), extension='.npy', standardize=False)
# dev_test_dataset.mean = dev_train_dataset.mean
# dev_test_dataset.std = dev
# _train_dataset.std
# dev_test_dataset.standardize = True
# dev_test_dataset.preGenerateMelSpecs()


machineFolders = os.listdir('./DCASE2024')

# getting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] device used for training...{}".format(device))

for machinesName in machineFolders:
    datasetType, machineType = machinesName.split('_')

    if not os.path.exists(f'./{datasetType}_norm/{machineType}'):
        os.makedirs(f'./{datasetType}_norm/{machineType}')

    models_dir = f'./{datasetType}_norm/{machineType}'

    # initializing hyperparameters
    INIT_LR = 0.0001  # 0.1
    BATCH_SIZE = 75 # 32
    EPOCHS = 250

    print(f"[INFO] loading data for {machinesName}...")
    dev_train_dataset = DCASE2024MachineDataset('./DCASE2024', ('dev', 'train'), machinesName)
    # dev_train_dataset.preGenerateMelSpecs(True)

    # for 1 type of machine, 1000 train samples
    print(f'Number of train Samples: {len(dev_train_dataset)}')

    #input_shape = dev_train_dataset.shapeof(0)
    printDataInfo(dev_train_dataset)

    train_index, val_index = stratified_split(dev_train_dataset, random_state=19)

    train_dataset = Subset(dev_train_dataset, train_index)

    val_dataset = Subset(dev_train_dataset, val_index)

    # initializing  dataloaders
    print("[INFO] Initializing Dataloaders")
    trainDataLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valDataLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    global_min_value = 0
    global_max_value = 0
    for index, (mel_specs, _) in enumerate(dev_train_dataset):
        if index == 0:
            global_min_value = np.min(mel_specs)
            global_max_value = np.max(mel_specs)
        else:
            if np.max(mel_specs).item() > global_max_value:
                global_max_value = np.max(mel_specs)
            if np.max(mel_specs).item() < global_min_value:
                global_min_value = np.min(mel_specs)

    # calculate steps per epoch for training, validation set
    trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
    valSteps = len(valDataLoader.dataset) // BATCH_SIZE

    # print(f"[INFO] Number of steps (train/val/test): {trainSteps}, {valSteps}, {testSteps}")

    print(f"[INFO] Number of steps (train/val): {trainSteps}, {valSteps}")
    """
    for spec, cls in trainDataLoader:
        print(spec.shape)
        print(cls.type)
        break
        
    """

    print("[INFO] initializing AutoEncoder...")
    # initializing autoencoder
    model = ConvAutoencoder(256).to(device)
    model.load_state_dict(torch.load(f'{models_dir}/CAE_normed.pt'))
    model.eval()
    # model = UNet(input_shape, 1024).to(device)
    # initializig optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)

    # intializing lr_scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    mse = nn.MSELoss(reduction='none')
    def combined_loss(pred, target):
        mse = f.mse_loss(pred, target)
        try:
            from pytorch_msssim import ssim
            ssim_loss = 1 - ssim(pred, target, data_range=(global_max_value-global_min_value), size_average=True)
        except ImportError:
            ssim_loss = 0  # fallback if pytorch_msssim isn't available
        return 0.8 * mse + 0.2 * ssim_loss


    # def log_cosh_loss(y_pred, y_true):
    #     diff = y_pred - y_true
    #     return torch.mean(torch.log(torch.cosh(torch.clamp(diff, -10, 10))))


    H = {
        "total_train_loss": [],
        "total_val_loss": [],
        "time": []
    }

    #training_model(models_dir, model, optimizer, scheduler, mse, H, EPOCHS, trainDataLoader, trainSteps,
    #               valDataLoader, valSteps, device)

    prev_train_state = torch.load(f"{models_dir}/train_state_dict_CAE_normed.pt")
    model.load_state_dict(prev_train_state['model_state_dict'])
    optimizer.load_state_dict(prev_train_state['optimizer_state_dict'])
    scheduler.load_state_dict(prev_train_state['lr_scheduler'])
    last_epoch = prev_train_state['epoch']
    H = prev_train_state['train_loss_history']

    test_model(models_dir, model, machinesName, mse, device)





