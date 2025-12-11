import os
import math
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader

from torchvision.models import resnet18
import torchvision.transforms.v2 as transforms
from torchvision.io import read_image


class YourImageDataset(Dataset):
    def __init__(self, dataset_df, transform = None):
        self.data_frame = dataset_df
        self.transform = transform
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx, 0]
        # print(img_path)
        img_path = os.path.join("/janaki/backup/users/student/pg/pg23/vaibhav.rathore/datasets/PA-100K/data", img_path)
        labels = self.data_frame.iloc[idx, 1:].values.astype('float')
        image = read_image(img_path)

        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels, img_path
    
class par_bce_logit_loss(nn.Module):
    def __init__(self, device, weights, reduction = 'none', num_att = 26):
        super(par_bce_logit_loss, self).__init__()
        self.pos_weights = torch.tensor([math.exp(1-v) for v in weights]).to(device)
        self.neg_weights = torch.tensor([math.exp(w) for w in weights]).to(device)
        self.reduction = reduction
        self.num_att = num_att
    
    def forward(self, pred, labels):
        assert pred.size() == labels.size()
        
        weights = torch.where(labels == 1, self.pos_weights, self.neg_weights)
        
        loss = F.binary_cross_entropy_with_logits(pred, labels, weight = weights, reduction = self.reduction) * self.num_att
        
        return loss.mean()

def calc_weights(df, num_attr, attr_groups):
    weights = []
    freqCount = np.zeros(num_attr)
    labels = df.iloc[:, 1:].values
    freqCount = np.sum(labels, axis=0) # Vectorized sum


    # for _, label, _ in  dataset:
    #     for idx, j in enumerate(label):
    #         if j == 1:
    #             freqCount[idx] += 1

    for group in attr_groups:
        group_wts = []
        for i in group:
            if freqCount[i] == 0:
                attr_wt = 0
            else:
                attr_wt = len(df) / (len(group) * freqCount[i])
            group_wts.append(attr_wt)
        group_wts = np.array(group_wts)

        if np.isnan(group_wts).any() or np.isinf(group_wts).any():
            group_wts[np.isnan(group_wts)] = 0
            group_wts[np.isinf(group_wts)] = 0
        
        weights.extend(preprocessing.normalize([group_wts])[0].tolist())
    
    return weights

# def calc_weights(dataset, num_attr, attr_groups):
#     weights = []
#     freqCount = np.zeros(num_attr)

#     for _, label, _ in dataset:
#         for idx, j in enumerate(label):
#             if j == 1:
#                 freqCount[idx] += 1

#     for group in attr_groups:
#         group_wts = []
#         for i in group:
#             # FIX: Add a small epsilon or check for zero to avoid crash
#             count = freqCount[i]
#             if count == 0:
#                 print(f"Warning: Attribute index {i} has 0 positive samples.")
#                 attr_wt = 0 # or handle appropriately
#             else:
#                 attr_wt = len(dataset) / (len(group) * count)
#             group_wts.append(attr_wt)
        
#         group_wts = np.array(group_wts)

#         if np.isnan(group_wts).any() or np.isinf(group_wts).any():
#             group_wts[np.isnan(group_wts)] = 0
#             group_wts[np.isinf(group_wts)] = 0
        
#         # Handle case where group_wts is all zeros (to avoid error in normalize)
#         if np.sum(group_wts) == 0:
#              weights.extend(group_wts.tolist())
#         else:
#              weights.extend(preprocessing.normalize([group_wts])[0].tolist())
    
#     return weights

def finetune(train_df, val_df, num_attr, attr_groups, epochs, output_path):
    os.makedirs(output_path, exist_ok=True)
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(message)s",
        handlers = [
            logging.FileHandler(os.path.join(output_path, "training.log"), mode="w"),
            logging.StreamHandler()
        ]
    )

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    train_tx = transforms.Compose([
        transforms.ToImage(),
        transforms.Resize((224, 224)),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(30),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    val_tx = transforms.Compose([
        transforms.ToImage(),
        transforms.Resize((224, 224)),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    train_dataset = YourImageDataset(train_df, transform = train_tx)
    val_dataset = YourImageDataset(val_df, transform = val_tx)

    train_loader = DataLoader(train_dataset, batch_size = 32, num_workers = 4, pin_memory = True, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 32, num_workers = 4, pin_memory = True, shuffle = False)

    model = resnet18(weights = "DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, num_attr)
    model.to(device)

    # loss_wts = calc_weights(train_dataset, num_attr, attr_groups)
    loss_wts = calc_weights(train_df, num_attr, attr_groups)
    criterion = par_bce_logit_loss(device, loss_wts, reduction = "mean", num_att = num_attr)
    optimizer = SGD(model.parameters(), lr = 1e-2)
    scheduler = ExponentialLR(optimizer, gamma = 0.95)

    best_train_loss = math.inf
    best_val_loss = math.inf

    for ep in range(epochs):

        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        model.train()
        print(f"EPOCH {ep+1}: TRAINING PHASE")
        for images, labels, _ in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            train_total += labels.numel()
            train_correct += (predicted == labels).sum().item()
        
        scheduler.step()
        train_loss = running_train_loss/len(train_loader)
        train_accuracy = (train_correct / train_total) * 100

        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0 

        print(f"EPOCH {ep+1}: VALIDATION PHASE")
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) >= 0.5).float()
                val_total += labels.numel()
                val_correct += (predicted == labels).sum().item()
        
        val_loss = running_val_loss/len(val_loader)
        val_accuracy = (val_correct / val_total) * 100

        logging.info(
            f"Epoch [{ep+1}/{epochs}] "
            f"| Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% "
            f"| Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
        )

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_loss_model_path = os.path.join(output_path, "best_train_loss_model.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": ep + 1,
                "train_loss": train_loss,
                "val_loss": val_loss
            }, best_train_loss_model_path)
            print(f"Best training loss achieved: {train_loss} at epoch: {ep + 1}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_model_path = os.path.join(output_path, "best_val_loss_model.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": ep + 1,
                "train_loss": train_loss,
                "val_loss": val_loss
            }, best_val_loss_model_path)
            print(f"Best validation loss achieved: {val_loss} at epoch: {ep + 1}")

    final_model_path = os.path.join(output_path, "final_epoch_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss
    }, final_model_path)
    print(f"Final epoch model saved at {final_model_path} || Train Loss = {train_loss} || Val loss = {val_loss}")

def evaluate_model(test_df, model_path, num_attr):
    """
    Evaluates the trained model on the test dataset.

    Args:
        test_df (pd.DataFrame): DataFrame containing test data (image paths and labels).
        model_path (str): Path to the saved model checkpoint (.pth file).
        num_attr (int): Number of attributes/classes for the output layer.
    """
    logging.info("\n--- Starting Model Evaluation on Test Set ---")
    
    # 1. Setup Device
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 2. Define Transformations (using the validation transform)
    test_tx = transforms.Compose([
        transforms.ToImage(),
        transforms.Resize((224, 224)),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    # 3. Create Dataset and DataLoader
    test_dataset = YourImageDataset(test_df, transform = test_tx)
    test_loader = DataLoader(test_dataset, batch_size = 32, num_workers = 4, pin_memory = True, shuffle = False)
    
    # 4. Initialize Model and Load State 
    model = resnet18(weights = None) # Initialize without weights
    model.fc = nn.Linear(model.fc.in_features, num_attr)
    model.to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info(f"Loaded model from checkpoint trained at epoch: {checkpoint.get('epoch', 'N/A')}")
        logging.info(f"Checkpoint reported validation loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    except FileNotFoundError:
        logging.error(f"Model checkpoint not found at: {model_path}. Evaluation aborted.")
        return
    except Exception as e:
        logging.error(f"Error loading model state dict: {e}. Evaluation aborted.")
        return

    # 5. Define Loss Function (for reporting loss, we can use the same criterion)
    # Note: Weights are not strictly needed for evaluation but are required by the class constructor.
    # We will pass dummy weights or the train weights to initialize it.
    # Since we don't have the original train_dataset here, we'll initialize with arbitrary weights,
    # as the loss is only used for reporting and not for gradient updates.
    # A cleaner approach is to use standard BCEWithLogitsLoss for pure reporting.
    
    # Using a simple BCEWithLogitsLoss for test loss reporting
    test_criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # 6. Run Evaluation Loop
    model.eval()
    running_test_loss = 0.0
    test_correct = 0
    test_total = 0 
    
    print(f"TESTING PHASE on {len(test_dataset)} samples")
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Calculate loss for reporting
            loss = test_criterion(outputs, labels) 
            running_test_loss += loss.item()
            
            # Calculate metrics
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            test_total += labels.numel() # Total number of attribute predictions
            test_correct += (predicted == labels).sum().item() # Total correct attribute predictions

    # 7. Calculate Final Metrics
    test_loss = running_test_loss / len(test_loader)
    test_accuracy = (test_correct / test_total) * 100
    
    logging.info(
        f"\n--- Evaluation Results ---"
        f"\nTest Loss: {test_loss:.4f} "
        f"\nTest Accuracy (per attribute): {test_accuracy:.2f}%"
        f"\n--------------------------"
    )

if __name__ == "__main__":
    epochs = 70
    num_attr = 26
    
    if num_attr == 26:
        # PA-100K groups
        attr_groups = [
            range(0, 1),   # Gender
            range(1, 4),   # Age
            range(4, 7),   # Orientation
            range(7, 9),   # Hat, Glasses
            range(9, 13),  # Bags
            range(13, 19), # Upper Body
            range(19, 25), # Lower Body
            range(25, 26)  # Boots
        ]
    else:
        # Fallback to original groups (assuming 40)
        attr_groups = [
            range(0, 2),
            range(2, 4),
            range(4, 7),
            range(7, 9),
            range(9, 11),
            range(11, 24),
            range(24, 37),
            range(37, 40)
        ]
        
    train_csv = "/janaki/backup/users/student/pg/pg23/vaibhav.rathore/datasets/PA-100K/train.csv"
    val_csv = "/janaki/backup/users/student/pg/pg23/vaibhav.rathore/datasets/PA-100K/val.csv"
    test_csv = "/janaki/backup/users/student/pg/pg23/vaibhav.rathore/datasets/PA-100K/test.csv" # New test CSV path
    output_path = "/users/student/pg/pg23/vaibhav.rathore/PAR/Rethinking_of_PAR/exp_result"
    
    # Ensure all necessary folders are created and logging is set up
    os.makedirs(output_path, exist_ok=True)
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(message)s",
        handlers = [
            logging.FileHandler(os.path.join(output_path, "test_training.log"), mode="a"), # Append to existing log
            logging.StreamHandler()
        ]
    )

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    # --- 1. Train/Finetune the model ---
    finetune(train_df, val_df, num_attr, attr_groups, epochs, output_path)

    # --- 2. Evaluate the best model ---
    test_df = pd.read_csv(test_csv)
    
    # Path to the best validation loss model saved in the finetune function
    best_model_path = os.path.join(output_path, "best_val_loss_model.pth")
    
    # Call the new evaluation function
    evaluate_model(test_df, best_model_path, num_attr)

# if __name__ == "__main__":
#     epochs = 70
    
#     train_csv = "/janaki/backup/users/student/pg/pg23/vaibhav.rathore/datasets/PA-100K/train.csv"
#     val_csv = "/janaki/backup/users/student/pg/pg23/vaibhav.rathore/datasets/PA-100K/val.csv"
#     output_path = "/users/student/pg/pg23/vaibhav.rathore/PAR/Rethinking_of_PAR/exp_result"
    
#     train_df = pd.read_csv(train_csv)
#     val_df = pd.read_csv(val_csv)

#     num_attr = len(train_df.columns) - 1
#     print(f"Detected {num_attr} attributes.")

#     if num_attr == 26:
#         # PA-100K groups
#         attr_groups = [
#             range(0, 1),   # Gender
#             range(1, 4),   # Age
#             range(4, 7),   # Orientation
#             range(7, 9),   # Hat, Glasses
#             range(9, 13),  # Bags
#             range(13, 19), # Upper Body
#             range(19, 25), # Lower Body
#             range(25, 26)  # Boots
#         ]
#     else:
#         # Fallback to original groups (assuming 40)
#         attr_groups = [
#             range(0, 2),
#             range(2, 4),
#             range(4, 7),
#             range(7, 9),
#             range(9, 11),
#             range(11, 24),
#             range(24, 37),
#             range(37, 40)
#         ]
    
#     finetune(train_df, val_df, num_attr, attr_groups, epochs, output_path)