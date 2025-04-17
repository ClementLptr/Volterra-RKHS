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
from network.rgb_of import vnn_rgb_of_RKHS  # Assuming this is where RKHS_VNN is defined
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.profiler import profile, record_function, ProfilerActivity
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
logger = setup_logger()

def parse_args():
    parser = argparse.ArgumentParser(description="Train RKHS_VNN model with configurable number of projectors")
    parser.add_argument('--num_projectors', type=int, default=24, help='Number of projectors to use')
    return parser.parse_args()

def compute_projectors(num_projectors=24, projectors_dir="results/projectors", 
                      device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    """Loads pre-computed projectors from results/projectors/ directory."""
    pre_determined_projectors = {}
    layer_configs = {
        1: (3, 3, 3-insensitive),
        2: (24, 3, 3, 3),
        3: (48, 3, 3, 3),
        4: (96, 3, 3, 3),
        5: (192, 3, 3, 3),
    }
    
    for layer in range(1, 6):
        proj_file = os.path.join(projectors_dir, f"projectors_layer_{layer}_P{num_projectors}.pt")
        if os.path.exists(proj_file):
            projector = torch.load(proj_file, map_location=device)
            if projector.shape != (num_projectors, *layer_configs[layer]):
                logger.warning(f"Projectors for layer {layer} have shape {projector.shape}, expected {num_projectors, *layer_configs[layer]}. Regenerating.")
                projector = torch.randn(num_projectors, *layer_configs[layer]).to(device) * 0.1
            pre_determined_projectors[layer] = projector
            logger.info(f"Loaded projectors for layer {layer} from {proj_file}")
        else:
            logger.warning(f"Projector file {proj_file} not found. Using random initialization.")
            pre_determined_projectors[layer] = torch.randn(num_projectors, *layer_configs[layer]).to(device) * 0.1
    
    return pre_determined_projectors

def initialize_models(num_classes=51, num_projectors=24, projectors_dir="results/projectors", 
                      device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    pre_determined_projectors = compute_projectors(num_projectors=num_projectors, projectors_dir=projectors_dir, device=device)
    model_RGB = vnn_rgb_of_RKHS.RKHS_VNN(
        num_classes=num_classes,
        num_projector=num_projectors,
        num_ch=3,
        input_shape=(16, 112, 112),
        dropout_rate=0.5,
        pretrained=False,
        pre_determined_projectors=pre_determined_projectors
    ).to(device)
    logger.info(f"Model initialized with {num_projectors} projectors on {device}")
    return model_RGB

def initialize_optimizer(model_RGB, lr=1e-3):
    train_params = [{'params': vnn_rgb_of_RKHS.get_1x_lr_params(model_RGB), 'lr': lr}]
    optimizer = optim.Adam(train_params, weight_decay=5e-4)
    return optimizer

def initialize_scheduler(optimizer, num_epochs, warmup_epochs=5):
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
    return optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

def initialize_dataloaders(dataset='hmdb51', clip_len=16, batch_size=8):
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
    _, top_k_preds = outputs.topk(k, dim=1)
    labels_expanded = labels.view(-1, 1).expand_as(top_k_preds)
    correct = top_k_preds.eq(labels_expanded).sum().float()
    return correct / labels.size(0)

def train_epoch(phase, model_RGB, trainval_loaders, optimizer, criterion, device, logger, writer, epoch, num_epochs, scaler):
    start_time = timeit.default_timer()
    running_loss = 0.0
    running_corrects = 0.0
    running_top5_corrects = 0.0
    failed_batches = 0
    
    model_RGB.train() if phase == 'train' else model_RGB.eval()
    for i, (inputs, labels) in enumerate(trainval_loaders[phase]):
        try:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast('cuda'):
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
        except Exception as batch_error:
            logger.error(f"Error in {phase} phase, batch {i}: {batch_error}")
            failed_batches += 1
            continue

    dataset_size = len(trainval_loaders[phase].dataset)
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size
    epoch_top5_acc = running_top5_corrects / dataset_size
    
    logger.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Top-5 Acc: {epoch_top5_acc:.4f}")
    if failed_batches > 0:
        logger.warning(f"{failed_batches} batches failed during {phase} phase")
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
        'time': elapsed_time,
        'failed_batches': failed_batches
    }
    
def save_to_csv(metrics, csv_path):
    df = pd.DataFrame([metrics])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)

def train_model(dataset='hmdb51', save_dir=None, num_classes=51, lr=1e-3,
                num_epochs=100, save_epoch=20, useTest=True, test_interval=10, 
                warmup_epochs=5, save_csv=True, num_projectors=24):
    save_dir = save_dir or os.path.dirname(os.path.abspath(__file__))    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    modelName = 'RKHS_5_RGB'
    saveName = f"{modelName}-{dataset}-P{num_projectors}"  # Ajout de num_projectors dans le nom
    config = load_config('config/config.json')
    model_version = config.get('model_version', 'default_version')
    csv_path = os.path.join(save_dir, f"{saveName}_training_metrics.csv")  # CSV inclut num_projectors

    try:
        train_dataloader, val_dataloader, test_dataloader = initialize_dataloaders(dataset)
        model_RGB = initialize_models(num_classes=num_classes, num_projectors=num_projectors, device=device)
        optimizer = initialize_optimizer(model_RGB, lr)
        criterion = nn.CrossEntropyLoss().to(device)
        scaler = GradScaler()
        scheduler = initialize_scheduler(optimizer, num_epochs, warmup_epochs)
        
        total_params = sum(p.numel() for p in model_RGB.parameters())
        logger.info(f"Total parameters: {total_params}")
        
        trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
        
        for epoch in range(num_epochs):
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
    args = parse_args()
    logger = setup_logger()
    writer = SummaryWriter(log_dir=os.path.join("logs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    train_model(save_csv=True, num_projectors=args.num_projectors)