import torch.nn as nn
from data_loader import dataloader_generator as train_generator
import torch
import torch.optim as optim
import os
import logging
import time
from model import LSTMModel,LSTMModel_IMU,NeuroPose
from test import infer
import random
import numpy as np


TRAINING_FOLDER = "/home/your_path_to/EMG/train_gc"
TESTING_FOLDER = "/home/your_path_to/EMG/test_gc"


# random seed for reproducibility
seed = 42
# PyTorch random number generator
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# set deterministic behavior for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def test_inference(model,inference_dataloader,device):
    model.eval()
  
    # excute inference
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inference_results, gt = infer(model, inference_dataloader, device)

    scores = 0
    from evaluate import calculate_pa_mpjpe,calculate_mpjpe
    # process inference results
    for i, result in enumerate(inference_results):
        # print(result )
        # print(gt[i])
        # print(calculate_pa_mpjpe(result,gt[i]))
        scores += (calculate_mpjpe(result,gt[i]))
    avg = scores / len(inference_results)

    pa_scores = 0
    for i, result in enumerate(inference_results):
        # print(result )
        # print(gt[i])
        # print(calculate_pa_mpjpe(result,gt[i]))
        pa_scores += (calculate_pa_mpjpe(result,gt[i])[0])
    pa_avg = pa_scores / len(inference_results)

    return avg ,pa_avg

dataloader = train_generator(path=TRAINING_FOLDER, batch_size=48)
# get current timestamp for logging
timestamp = time.strftime("%Y%m%d_%H%M%S")

file = f"{timestamp}.txt"

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, file),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)




# initialize model parameters
input_size = 128  # EMG channel
hidden_size = 256
output_size = 21  # prwedict future channel values

#chekc if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = NeuroPose().to(device)
# model = LSTMModel_IMU(input_size, hidden_size, output_size).to(device)

# define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) 
# set learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
#copy model to GPU if available
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
min_score = float('inf')
# train the model
epochs = 30
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for batch_inputs, batch_targets, batch_reference,batch_imu in dataloader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        batch_reference = batch_reference.to(device)
        batch_imu = batch_imu.to(device)
        # print(batch_imu.shape)
        optimizer.zero_grad()

        # check for NaN values in the tensors
        for tensor_name, tensor in {"batch_inputs": batch_inputs, 
                                    "batch_reference": batch_reference, 
                                    "batch_targets": batch_targets}.items():
            if torch.isnan(tensor).any():
                logging.warning(f"NaN detected in {tensor_name} during epoch {epoch + 1}. Skipping this batch.")
                continue

        # forward pass
        outputs = model(batch_inputs.zero_(), batch_reference,batch_imu.zero_())
        # outputs = model(batch_inputs, batch_reference, batch_imu)

        # process outputs and targets
        target_processed = batch_targets.view(batch_targets.size(0), -1)

        # calculate loss
        loss = criterion(outputs, target_processed)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()
    # log average loss for the epoch
    avg_loss = epoch_loss / len(dataloader)
    logging.info(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

    # save model every 10 epochs and evaluate
    if (epoch + 1) % 10 == 0:  
        inference_dataloader = train_generator(path=TESTING_FOLDER, batch_size=1)
        score ,pa_score = test_inference(model=model,inference_dataloader=inference_dataloader,device=device)
        if min_score>score:
            min_score = score

        logging.info(f"AVG PA Score {pa_score }")
        logging.info(f"AVG Score {score } , Min Score {min_score }")
        print(f"AVG Score {score } ,Min Score {min_score }")
        print(f"AVG PA Score {pa_score }")
        model_path = os.path.join(model_dir, f"model_epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), model_path)
        logging.info(f"Model saved at {model_path}")