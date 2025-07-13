from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm.auto import tqdm
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch


def get_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    mean = 0.
    std = 0.
    total_images = 0
    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)  # (B, C, H*W)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += images.size(0)
    mean /= total_images
    std /= total_images
    return mean, std

def visualize_classes(classes, dataset):
    y_train = np.array(dataset.targets)

    plt.figure(figsize=(10, 10), dpi=500)
    for i in range(len(classes)):
        for j in range(len(y_train)):
            if y_train[j] == i:
                idx = j
                break

        image = dataset[idx][0]

        ax = plt.subplot(2, 5, i + 1)
        plt.imshow(image.permute(1, 2, 0).cpu().numpy())
        plt.title(classes[i])
        plt.axis('off')

    plt.subplots_adjust(hspace=-0.8)
    plt.tight_layout()
    plt.show()

class EarlyStopping():
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.no_improvement_count = 0

    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                print("Early stopping triggered.")
                return True  # Signal to stop training
        return False

def train(model, optimizer, criterion, train_loader, val_loader, device, epochs):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    early_stopping = EarlyStopping(patience=5, delta=0.001)

    for epoch in tqdm(range(epochs)):
        running_train_loss = 0.0
        running_train_acc = 0.0
        running_val_loss = 0.0
        running_val_acc = 0.0

        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.long().to(device)
            y_pred = model(X)

            loss = criterion(y_pred, y)
            running_train_loss += loss.item()
            running_train_acc += accuracy_score(y.detach().cpu().numpy(), y_pred.argmax(dim=1).detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_train_loss /= len(train_loader)
        running_train_acc /= len(train_loader)
        train_loss.append(running_train_loss)
        train_acc.append(running_train_acc)
        print(f"Epoch {epoch+1}/{epochs} --> Loss: {running_train_loss:.3f} | Acc: {running_train_acc*100:.2f}%")

        model.eval()
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.long().to(device)
                y_pred = model(X)
                loss = criterion(y_pred, y)
                running_val_loss += loss.item()
                running_val_acc += accuracy_score(y.detach().cpu().numpy(), y_pred.argmax(dim=1).detach().cpu().numpy())


            running_val_loss /= len(val_loader)
            running_val_acc /= len(val_loader)
            val_loss.append(running_val_loss)
            val_acc.append(running_val_acc)
            print(f"Val Loss: {running_val_loss:.3f} | Val Acc: {running_val_acc*100:.2f}%\n")
            if early_stopping.check_early_stop(running_val_loss):
                print(f"Stopping training at epoch {epoch+1}")
                break

    return train_loss, train_acc, val_loss, val_acc

def test(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, criterion: torch.nn.Module, device: str='cpu'):
    """Performs a testing loop step on model going over data_loader"""
    test_loss, test_acc = 0.0, 0.0

    # Put the model in eval mode
    model.eval()

    # Turn on inference mode context manager
    with torch.no_grad():
        for X, y in dataloader:
            # Send the data to the target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass (outputs raw logits)
            test_pred = model(X)

            # 2. Calculate the loss/acc
            test_loss += criterion(test_pred, y)
            test_acc += accuracy_score(y.detach().cpu().numpy(), test_pred.argmax(dim=1).detach().cpu().numpy())

        # Adjust metrics and print out
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        print(f"Test loss: {test_loss:.3f} | Test acc: {test_acc*100:.2f}%\n")

def print_execution_time(start: float, end: float, device: str='cpu'):
    """Prints difference between start and end time."""
    total_time = end - start
    print(f"\nExecution time on {device} : {total_time:.3f} seconds")

def make_predictions(model: torch.nn.Module, data: list, device: torch.device='cpu'):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare the sample (add a batch dimension and pass to target device)
            sample = torch.unsqueeze(sample, dim=0).to(device)

            # Forward pass (model outputs raw logits)
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            # Get pred_prob off the GPU further calculations
            pred_probs.append(pred_prob.cpu())

    # Stack the pred_probs to return list into a tensor
    return torch.stack(pred_probs)

def plot_results(result_norm_name_file, result_aug_name_file):
    results_norm = np.load(result_norm_name_file)
    results_aug = np.load(result_aug_name_file)

    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
    axs[0].plot(results_norm['train_loss'], label='Train Loss Norm', marker='o')
    axs[0].plot(results_norm['val_loss'], label='Validation Loss Norm', marker='s')
    axs[0].plot(results_aug['train_loss'], label='Train Loss Aug', marker='x')
    axs[0].plot(results_aug['val_loss'], label='Validation Loss Aug', marker='d')
    axs[0].set_title('Training and Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(results_norm['train_acc'], label='Train Accuracy Norm', marker='o')
    axs[1].plot(results_norm['val_acc'], label='Validation Accuracy Norm', marker='s')
    axs[1].plot(results_aug['train_acc'], label='Train Accuracy Aug', marker='x')
    axs[1].plot(results_aug['val_acc'], label='Validation Accuracy Aug', marker='d')
    axs[1].set_title('Training and Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)

    plt.suptitle("Training and Validation Metrics")
    plt.tight_layout()
    plt.show()

def confusion_matrix_plot(model, dataloader, classes, device='cpu'):
    """Plots confusion matrix for the model predictions on the given dataloader."""

    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)

            y_true.append(y.detach().cpu().numpy())
            y_pred.append(predicted.detach().cpu().numpy())

    # Concatenate all true and predicted labels
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()