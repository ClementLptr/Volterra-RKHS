
import os
import timeit
import pandas as pd
import torch
from torch import nn, optim
from config.logger import setup_logger
from config.dataloaders.toy_dataset import ToyDataset
from torch.utils.data import DataLoader
from network.rgb_of.simple_VNN import SimpleVNN
# Set environment variable for CUDA allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Setup logger
logger = setup_logger()


def initialize_dataloaders(dataset='hmdb51', clip_len=16, batch_size=4):
    """Initializes data loaders with a smaller batch size."""
    train_dataloader = DataLoader(
        ToyDataset(dataset=dataset, split='train', clip_len=clip_len),
        batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_dataloader = DataLoader(
        ToyDataset(dataset=dataset, split='val', clip_len=clip_len),
        batch_size=batch_size, num_workers=2, pin_memory=True
    )
    return train_dataloader, val_dataloader

def top_k_accuracy(outputs, labels, k=5):
    """Computes top-k accuracy."""
    _, top_k_preds = outputs.topk(k, dim=1)
    labels_expanded = labels.view(-1, 1).expand_as(top_k_preds)
    correct = top_k_preds.eq(labels_expanded).sum().float()
    return correct / labels.size(0)

def save_to_csv(metrics, csv_path):
    """Saves metrics to a CSV file."""
    df = pd.DataFrame([metrics])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)
        
def train_model(dataset='hmdb51', num_classes=51, lr=1e-3, num_epochs=10, save_csv=True):
    """Simplified training loop for the model."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize model, optimizer, and criterion
    model = SimpleVNN(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss().to(device)

    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params}")

    # Initialize data loaders
    train_dataloader, val_dataloader = initialize_dataloaders(dataset, batch_size=4)
    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}

    # CSV path for metrics
    csv_path = f"simple_vnn_{dataset}_metrics.csv"

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0
            running_top5_corrects = 0.0

            for inputs, labels in trainval_loaders[phase]:
                try:
                    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    logger.debug(f"{phase} batch input shape: {inputs.shape}, labels shape: {labels.shape}")

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        probs = nn.Softmax(dim=1)(outputs)
                        preds = torch.max(probs, 1)[1]
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            # Log gradient norm
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            logger.debug(f"Gradient norm: {grad_norm:.4f}")
                            optimizer.step()

                        # Check for NaNs or infinities
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.error(f"{phase} loss is NaN or Inf: {loss}")
                            return

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        running_top5_corrects += top_k_accuracy(outputs, labels, k=5) * inputs.size(0)

                except RuntimeError as e:
                    logger.error(f"Error in {phase} phase: {e}")
                    if 'out of memory' in str(e):
                        torch.cuda.empty_cache()
                    continue

            epoch_loss = running_loss / len(trainval_loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(trainval_loaders[phase].dataset)
            epoch_top5_acc = running_top5_corrects / len(trainval_loaders[phase].dataset)

            logger.info(f"{phase} Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f} "
                        f"Acc: {epoch_acc:.4f} Top-5 Acc: {epoch_top5_acc:.4f}")

            if save_csv:
                metrics = {
                    'phase': phase,
                    'epoch': epoch + 1,
                    'loss': epoch_loss,
                    'accuracy': epoch_acc.item(),
                    'top5_accuracy': epoch_top5_acc.item()
                }
                save_to_csv(metrics, csv_path)

    # Save final model
    torch.save(model.state_dict(), f"simple_vnn_{dataset}_final.pth")
    logger.info("Training completed and model saved.")

if __name__ == "__main__":
    train_model(dataset='hmdb51', num_classes=51, save_csv=True)
