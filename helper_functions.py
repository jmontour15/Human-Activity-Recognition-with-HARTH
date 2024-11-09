import csv
from datetime import datetime
from torch import nn
import os
import pandas as pd
from matplotlib import pyplot as plt
import torch

def plot_activity_time(data_path, sampling_rate, label_key, label_column='label', multiple_files=True):
    """
    Plots the total time spent on each activity based on one or multiple CSV files.

    Parameters:
    - data_path (str): The path to the CSV file or directory containing CSV files.
    - sampling_rate (int): The sampling rate of the data (in Hz).
    - label_key (dict): A dictionary mapping labels to activity names.
    - label_column (str): The name of the column containing the labels.
    - multiple_files (bool): Whether to process multiple files (True) or a single file (False).
    """
    # Initialize an empty dataframe to store label counts (time spent in each activity)
    all_label_counts = pd.Series(dtype=int)
    
    if multiple_files:
        # Process multiple files in a directory
        for file_name in os.listdir(data_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(data_path, file_name)
                
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Count the occurrences of each label
                label_counts = df[label_column].value_counts()
                
                # Aggregate the label counts across files
                all_label_counts = all_label_counts.add(label_counts, fill_value=0)
    else:
        # Process a single file
        df = pd.read_csv(data_path)
        
        # Count the occurrences of each label
        all_label_counts = df[label_column].value_counts()
    
    # Convert counts (rows) to time in seconds, then minutes (assuming rows = time steps)
    all_label_time = all_label_counts / sampling_rate / 60  # Time in minutes

    # Rename the labels using the provided key
    all_label_time.index = all_label_time.index.map(str)  # Ensure index is strings
    all_label_time.index = all_label_time.index.map(lambda x: label_key.get(x, x))  # Map labels to activity names
    
    # Sort by most to least time spent in activity
    all_label_time = all_label_time.sort_values(ascending=False)
    
    # Plot the total time spent on each activity
    plt.figure(figsize=(12, 6))
    all_label_time.plot(kind='bar', color='skyblue')
    plt.title('Total Time Spent on Each Activity (in Minutes)')
    plt.xlabel('Activity Labels')
    plt.ylabel('Time (minutes)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("Figures/class_distribution.png")


def extract_samples_by_label(data_path, output_directory, output_file_name, labels, label_column='label'):
    """
    Extract rows with specific labels from multiple CSV files and save them into a new CSV file.
    
    Parameters:
        data_path (str): The directory containing the input CSV files.
        output_directory (str): The directory where the output CSV file will be saved.
        output_file_name (str): The name of the output CSV file.
        labels (list): List of labels to extract.
        label_column (str): The column name in the CSV files where the label is stored. Defaults to 'label'.
    """
    
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Full path for the output CSV file
    output_file = os.path.join(output_directory, output_file_name)
    
    # Open the output file in write mode and append the filtered data
    with open(output_file, mode='w', newline='') as out_file:
        header_written = False
        
        # Loop through all CSV files in the directory
        for file_name in os.listdir(data_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(data_path, file_name)
                
                # Read the CSV file and drop any unwanted index column
                df = pd.read_csv(file_path).drop(columns=['Unnamed: 0'], errors='ignore')
                
                # Filter by the labels
                filtered_df = df[df[label_column].isin(labels)]
                
                # Append to the output file
                filtered_df.to_csv(out_file, mode='a', header=not header_written, index=False)
                header_written = True
    
    print(f"Data extracted and saved to {output_file}")


def relabel_dataset(csv_file, outfile, relabel_map):
    # Load the dataset
    df = pd.read_csv(csv_file)

    # Apply the relabeling using the mapping
    df['label'] = df['label'].map(relabel_map)

    # Overwrite the original dataset
    df.to_csv(outfile, index=False)

    return df

def create_windows(data, sequence_length=50, overlap=25):
    windows = []
    labels = []

    # Adjusted step size for overlapping windows
    step_size = sequence_length - overlap

    for start in range(0, len(data) - sequence_length + 1, step_size):
        window = data.iloc[start:start + sequence_length]
        label = window['label'].mode()[0]  # Get the most frequent label in the window
        windows.append(window[['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']].values)
        labels.append(label)

    return windows, labels
    
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device,
               scores_storage):

    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        # 1. Forward pass (outputs raw logits)
        y_pred = model(X)

        # 2. Calculate Loss and Accuracy
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()  # Use .item() to avoid accumulating tensors
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))  # go from logits to prediction labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Backward pass
        loss.backward()

        # 5. Update model parameters
        optimizer.step()

        # Print out progress for every 100 batches for debugging
        '''if batch % 100 == 0:
            print(f'Batch {batch}: Processed {batch * len(X)} samples out of {len(data_loader.dataset)}.')
        '''
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train Loss: {train_loss:.5f}\tTrain Accuracy: {train_acc:.2f}%")

    scores_storage['train_loss'].append(train_loss)
    scores_storage['train_acc'].append(train_acc)

    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              accuracy_fn,
              device,
              scores_storage):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()

    # Turn on inference mode context manager
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)

            # Accumulate loss and accuracy
            test_loss += loss_fn(test_pred, y).item()  # Use .item() for scalar
            test_acc += accuracy_fn(y_true=y,
                                    y_pred=test_pred.argmax(dim=1))

        # Calculate the average test loss and accuracy
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test Loss: {test_loss:.5f}\tTest Accuracy: {test_acc:.2f}%")

        # Store scores
        scores_storage['test_loss'].append(test_loss)
        scores_storage['test_acc'].append(test_acc)

    # Return test loss and accuracy
    return test_loss# , test_acc


def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def get_preds(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              device):
    
    # Switch model to evaluation mode
    model.eval()

    preds = []
    true = []

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            
            # Forward pass (get predictions)
            y_pred = model(X)
            
            # Get predicted labels by taking argmax across dimension 1 (logits to class labels)
            y_pred_labels = y_pred.argmax(dim=1)
            
            # Store predictions and true labels
            preds.extend(y_pred_labels.cpu().numpy())  # Move to CPU and convert to numpy
            true.extend(y.cpu().numpy())  # Move to CPU and convert to numpy

    return preds, true
