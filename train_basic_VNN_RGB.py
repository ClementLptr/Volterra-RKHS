import timeit
import os
from datetime import datetime
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
from config.logger import setup_logger, load_config
from config.dataloaders.dataset import VideoDataset
from network.rgb_of import vnn_rgb_of_highQ

from torch.cuda.amp import autocast, GradScaler  
from torch.profiler import profile, record_function, ProfilerActivity

# Set environment variable for CUDA allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def initialize_models(num_classes=51):
    """Initializes and returns the models."""
    model_RGB = vnn_rgb_of_highQ.VNN(num_classes=num_classes, num_ch=3, pretrained=False)
    return model_RGB

def initialize_optimizer(model_RGB, lr=1e-4):
    """Initializes the optimizer with all parameters from get_1x_lr_params."""
    train_params = [{'params': vnn_rgb_of_highQ.get_1x_lr_params(model_RGB), 'lr': lr}]
    optimizer = optim.Adam(train_params, weight_decay=5e-4)
    return optimizer

def initialize_scheduler(optimizer, num_epochs, warmup_epochs=5):
    """Initializes a hybrid scheduler with warmup and cosine annealing."""
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
    return optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

def initialize_dataloaders(dataset='hmdb51', clip_len=16, batch_size=8):
    """Initializes and returns the data loaders."""
    train_dataloader = DataLoader(
        VideoDataset(dataset=dataset, split='train', clip_len=clip_len), 
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_dataloader = DataLoader(
        VideoDataset(dataset=dataset, split='val', clip_len=clip_len), 
        batch_size=batch_size, num_workers=4, pin_memory=True
    )
    test_dataloader = DataLoader(
        VideoDataset(dataset=dataset, split='test', clip_len=clip_len), 
        batch_size=batch_size, num_workers=4, pin_memory=True
    )
    return train_dataloader, val_dataloader, test_dataloader

def save_checkpoint(epoch, model_RGB, optimizer, scheduler, save_dir, saveName, model_version):
    """Saves the model checkpoint with versioning, including scheduler state."""
    version_dir = os.path.join(save_dir, 'models', model_version)
    os.makedirs(version_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(version_dir, f"{saveName}_epoch-{epoch}.pth.tar")
    torch.save({
        'epoch': epoch + 1,
        'state_dict_rgb': model_RGB.state_dict(),
        'opt_dict': optimizer.state_dict(),
        'scheduler_dict': scheduler.state_dict(),
    }, checkpoint_path)
    return checkpoint_path

def top_k_accuracy(outputs, labels, k=5):
    """Computes top-k accuracy."""
    _, top_k_preds = outputs.topk(k, dim=1)
    labels_expanded = labels.view(-1, 1).expand_as(top_k_preds)
    correct = top_k_preds.eq(labels_expanded).sum().float()
    return correct / labels.size(0)

def train_epoch(phase, model_RGB, trainval_loaders, optimizer, criterion, device, logger, writer, epoch, num_epochs, scaler):
    """Handles the training/validation of a single epoch with mixed precision."""
    start_time = timeit.default_timer()
    running_loss = 0.0
    running_corrects = 0.0
    running_top5_corrects = 0.0

    if phase == 'train':
        model_RGB.train()
    else:
        model_RGB.eval()

    for inputs, labels in trainval_loaders[phase]:
        try:
            with torch.no_grad() if phase != 'train' else torch.enable_grad():
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                with autocast(enabled=(phase == 'train')):
                    outputs = model_RGB(inputs)
                    probs = nn.Softmax(dim=1)(outputs)
                    preds = torch.max(probs, 1)[1]
                    loss = criterion(outputs, labels)

                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_top5_corrects += top_k_accuracy(outputs, labels, k=5) * inputs.size(0)

        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.error(f"OOM during {phase} phase - skipping batch")
                torch.cuda.empty_cache()
                continue
            else:
                logger.error(f"Unexpected error during {phase} phase: {e}")
                raise

    epoch_loss = running_loss / len(trainval_loaders[phase].dataset)
    epoch_acc = running_corrects.double() / len(trainval_loaders[phase].dataset)
    epoch_top5_acc = running_top5_corrects / len(trainval_loaders[phase].dataset)

    logger.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Top-5 Acc: {epoch_top5_acc:.4f}")
    writer.add_scalar(f'{phase}/Loss', epoch_loss, epoch)
    writer.add_scalar(f'{phase}/Accuracy', epoch_acc, epoch)
    writer.add_scalar(f'{phase}/Top5_Accuracy', epoch_top5_acc, epoch)

    elapsed_time = timeit.default_timer() - start_time
    logger.info(f"{phase} epoch {epoch + 1}/{num_epochs} took {elapsed_time:.2f}s")

    return {
        'phase': phase,
        'epoch': epoch + 1,
        'loss': epoch_loss,
        'accuracy': epoch_acc.item(),
        'top5_accuracy': epoch_top5_acc.item(),
        'time': elapsed_time
    }

def save_to_csv(metrics, csv_path):
    """Sauvegarde les métriques dans un fichier CSV."""
    df = pd.DataFrame([metrics])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)

def train_model(dataset='hmdb51', save_dir=None, num_classes=51, lr=1e-4,
                num_epochs=100, save_epoch=20, useTest=True, test_interval=10, 
                warmup_epochs=5, save_csv=True):
    """Trains the model_RGB without mixed precision."""
    save_dir = save_dir or os.path.dirname(os.path.abspath(__file__))    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = setup_logger()
    logger.info(f"Using device: {device}")
    
    modelName = 'Basic_rgb_vnn_Five_Layers'
    saveName = f"{modelName}-{dataset}"
    scaler = GradScaler()
    config = load_config('config/config.json')
    model_version = config.get('model_version', 'default_version')
    csv_path = os.path.join(save_dir, f"{saveName}_training_metrics.csv")

    try:
        model_RGB = initialize_models(num_classes).to(device)
        optimizer = initialize_optimizer(model_RGB, lr)
        criterion = nn.CrossEntropyLoss().to(device)
        
        scheduler = initialize_scheduler(optimizer, num_epochs, warmup_epochs)
        
        total_params = sum(p.numel() for p in model_RGB.parameters())
        logger.info(f"Total parameters: {total_params}")
        
        train_dataloader, val_dataloader, test_dataloader = initialize_dataloaders(dataset)
        trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
        
        for epoch in range(num_epochs):
            # Removed scaler from train_epoch call
            train_metrics = train_epoch('train', model_RGB, trainval_loaders, optimizer, criterion, device, logger, writer, epoch, num_epochs, scaler)
            val_metrics = train_epoch('val', model_RGB, trainval_loaders, optimizer, criterion, device, logger, writer, epoch, num_epochs, scaler)

            if save_csv:
                save_to_csv(train_metrics, csv_path)
                save_to_csv(val_metrics, csv_path)
            
            scheduler.step()
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

            if (epoch + 1) % save_epoch == 0:
                checkpoint_path = save_checkpoint(epoch, model_RGB, optimizer, scheduler, save_dir, saveName, model_version)
                logger.info(f"Saved checkpoint: {checkpoint_path}")

            if useTest and (epoch + 1) % test_interval == 0:
                logger.info("Test phase not implemented in this version.")

    except Exception as e:
        logger.error(f"Error in training: {e}")

if __name__ == "__main__":
    logger = setup_logger()
    writer = SummaryWriter(log_dir=os.path.join("logs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    train_model(save_csv=True)