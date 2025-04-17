import timeit
import os
import socket
from datetime import datetime
import cv2
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from joblib import Parallel, delayed
from tensorboardX import SummaryWriter
from config.logger import setup_logger, load_config
from config.dataloaders.dataset import VideoDataset
from network.fusion import vnn_fusion_highQ
from network.rgb_of import vnn_rgb_of_highQ
from network.rgb_of import vnn_rgb_of_RKHS

# Set environment variable for CUDA allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def flow(X, Ht, Wd, of_skip=1, polar=False, logger=None):
    """Computes optical flow for a batch of frames."""
    try:
        X_of = np.zeros([int(X.shape[0] / of_skip), Ht, Wd, 2])
        of_ctr = -1
        
        for j in range(0, X.shape[0] - of_skip, of_skip):
            of_ctr += 1
            flow = cv2.normalize(
                cv2.calcOpticalFlowFarneback(
                    cv2.cvtColor(np.array(X[j + of_skip, :, :, :], dtype=np.uint8), cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(np.array(X[j, :, :, :], dtype=np.uint8), cv2.COLOR_BGR2GRAY),
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                ),
                None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
            )
            
            if polar:
                mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
                X_of[of_ctr, :, :, :] = np.concatenate([np.expand_dims(mag, axis=2), np.expand_dims(ang, axis=2)], axis=2)
            else:
                X_of[of_ctr, :, :, :] = flow
        
        return X_of
    except Exception as e:
        if logger:
            logger.error(f"Error in optical flow computation: {e}")
        raise


def compute_optical_flow(X, Ht, Wd, logger, num_proc=4, of_skip=1, polar=False):
    """Computes optical flow for a dataset using parallel processing."""
    try:
        X = (X.permute(0, 2, 3, 4, 1)).detach().cpu().numpy()
        optical_flow = Parallel(n_jobs=num_proc)(
            delayed(flow)(X[i], Ht, Wd, of_skip, polar, logger) for i in range(X.shape[0])
        )
        X_of = torch.tensor(np.asarray(optical_flow)).float()
        return X_of.permute(0, 4, 1, 2, 3)
    except Exception as e:
        if logger:
            logger.error(f"Error in computing optical flow: {e}")
        raise


def initialize_models(num_classes=51):
    """Initializes and returns the models."""
    # model_RGB = vnn_rgb_of_highQ.VNN(num_classes=num_classes, num_ch=3, pretrained=False)
    # model_OF = vnn_rgb_of_highQ.VNN(num_classes=num_classes, num_ch=2, pretrained=False)
    model_RGB = vnn_rgb_of_RKHS.RKHS_VNN(num_classes=num_classes, num_projector=15, num_ch=3, input_shape=(16, 16, 16), dropout_rate=0.5, pretrained=False,
        pre_determined_projectors = {
            1: torch.randn(15, 3, 3, 3, 3),
            2: torch.randn(15, 24, 3, 3, 3)
        }
    )
    # model_fuse = vnn_fusion_highQ.VNN_F(num_classes=num_classes, num_ch=64, pretrained=False)
    
    return model_RGB


def initialize_optimizer(model_RGB, model_OF, model_fuse, lr=1e-4):
    """Initializes the optimizer and returns it."""
    train_params = [
        # {'params': vnn_rgb_of_highQ.get_1x_lr_params(model_RGB), 'lr': lr},
        # {'params': vnn_rgb_of_highQ.get_1x_lr_params(model_OF), 'lr': lr},
        {'params': vnn_rgb_of_RKHS.get_1x_lr_params(model_RGB), 'lr': lr},
        # {'params': vnn_rgb_of_RKHS.get_1x_lr_params(model_OF), 'lr': lr},
        # {'params': vnn_fusion_highQ.get_1x_lr_params(model_fuse), 'lr': lr},
        # {'params': vnn_fusion_highQ.get_10x_lr_params(model_fuse), 'lr': lr}
    ]
    
    optimizer = optim.Adam(train_params, lr=lr, weight_decay=5e-4)
    return optimizer


def initialize_dataloaders(dataset='hmdb51', clip_len=16, batch_size=2):
    """Initializes and returns the data loaders."""
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train', clip_len=clip_len), 
                                  batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(VideoDataset(dataset=dataset, split='val', clip_len=clip_len), 
                                batch_size=batch_size, num_workers=4)
    test_dataloader = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=clip_len), 
                                 batch_size=batch_size, num_workers=4)
    
    return train_dataloader, val_dataloader, test_dataloader


def save_checkpoint(epoch, model_RGB, model_OF, model_fuse, optimizer, save_dir, saveName, model_version):
    """Saves the model checkpoint with versioning."""
    version_dir = os.path.join(save_dir, 'models', model_version)
    os.makedirs(version_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(version_dir, f"{saveName}_epoch-{epoch}.pth.tar")
    torch.save({
        'epoch': epoch + 1,
        'state_dict_rgb': model_RGB.state_dict(),
        'state_dict_of': model_OF.state_dict(),
        'state_dict_fuse': model_fuse.state_dict(),
        'opt_dict': optimizer.state_dict(),
    }, checkpoint_path)
    return checkpoint_path


def train_epoch(phase, model_RGB, model_OF, model_fuse, trainval_loaders, optimizer, criterion, device, logger, writer, epoch, num_epochs):
    """Handles the training/validation of a single epoch."""
    start_time = timeit.default_timer()
    running_loss = 0.0
    running_corrects = 0.0
    
    # Set model to training/evaluation mode
    model_RGB.train() if phase == 'train' else model_RGB.eval()
    # model_OF.train() if phase == 'train' else model_OF.eval()
    # model_fuse.train() if phase == 'train' else model_fuse.eval()

    for inputs, labels in trainval_loaders[phase]:
        try:
            inputs_of = compute_optical_flow(inputs, 112, 112, logger)
            inputs = Variable(inputs, requires_grad=True).to(device)
            inputs_of = Variable(inputs_of, requires_grad=True).to(device)
            labels = Variable(labels).to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs_rgb = model_RGB(inputs)
                outputs_of = model_OF(inputs_of)
                outputs = model_fuse(torch.cat((outputs_rgb, outputs_of), 1))

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        except Exception as batch_error:
            logger.error(f"Error processing batch in {phase} phase: {batch_error}")
            continue

    epoch_loss = running_loss / len(trainval_loaders[phase].dataset)
    epoch_acc = running_corrects / len(trainval_loaders[phase].dataset)

    writer.add_scalar(f'data/{phase}_loss_epoch', epoch_loss, epoch)
    writer.add_scalar(f'data/{phase}_acc_epoch', epoch_acc, epoch)

    logger.info(f"[{phase}] Epoch: {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    stop_time = timeit.default_timer()
    logger.info(f"Execution time for {phase} phase: {stop_time - start_time:.2f} seconds")

def train_model(dataset='hmdb51', save_dir=None, num_classes=51, lr=1e-4,
                num_epochs=100, save_epoch=20, useTest=True, test_interval=10):
    """Trains the model on the given dataset with specified parameters."""
    
    save_dir = save_dir or os.path.dirname(os.path.abspath(__file__))    
    # Check device availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    modelName = 'RKHS_First_Try'
    saveName = f"{modelName}-{dataset}"

    config = load_config('config/config.json')
    model_version = config.get('model_version', 'default_version')  # Default value if not found
    try:
        model_RGB = initialize_models(num_classes)
        # optimizer = initialize_optimizer(model_RGB, model_OF, model_fuse, lr)
        optimizer = initialize_optimizer(model_RGB, lr)
        criterion = nn.CrossEntropyLoss()
        
        # total_params = sum(p.numel() for p in model_RGB.parameters()) + \
        #                sum(p.numel() for p in model_OF.parameters()) + \
        #                sum(p.numel() for p in model_fuse.parameters())
        total_params = sum(p.numel() for p in model_RGB.parameters()) 
        logger.info(f'Total model parameters: {total_params / 1_000_000:.2f}M')

        model_RGB.to(device)
        # model_OF.to(device)
        # model_fuse.to(device)
        criterion.to(device)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname()) 
        writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard log directory: {log_dir}")

        train_dataloader, val_dataloader, test_dataloader = initialize_dataloaders(dataset)
        trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
        
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
            for phase in ['train', 'val']:
                train_epoch(phase, model_RGB, model_OF, model_fuse, trainval_loaders, optimizer, criterion, device, logger, writer, epoch, num_epochs)
            
            if epoch % save_epoch == (save_epoch - 1):
                checkpoint_path = save_checkpoint(epoch, model_RGB, model_OF, model_fuse, optimizer, save_dir, saveName, model_version)
                logger.info(f"Saved model checkpoint at {checkpoint_path}")

            if useTest and epoch % test_interval == (test_interval - 1):
                model_RGB.eval()
                # model_OF.eval()
                # model_fuse.eval()

                start_time = timeit.default_timer()
                running_loss = 0.0
                running_corrects = 0.0
                test_size = len(test_dataloader.dataset)

                for inputs, labels in test_dataloader:
                    inputs_of = compute_optical_flow(inputs, 112, 112, logger)  # Optical flow computation
                    inputs = inputs.to(device)
                    inputs_of = inputs_of.to(device)
                    labels = labels.to(device)

                    with torch.no_grad():
                        outputs_rgb = model_RGB(inputs)
                        outputs_of = model_OF(inputs_of)
                        outputs = model_fuse(torch.cat((outputs_rgb, outputs_of), 1))  # Fusion of RGB and Optical Flow outputs
                    
                    probs = nn.Softmax(dim=1)(outputs)
                    preds = torch.max(probs, 1)[1]  # Get the class predictions
                    loss = criterion(outputs, labels)

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / test_size
                epoch_acc = running_corrects.double() / test_size

                writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

                logger.info(f"[test] Epoch: {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                stop_time = timeit.default_timer()
                logger.info(f"Execution time for test: {stop_time - start_time:.2f} seconds")

        writer.close()
        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    logger = setup_logger()
    train_model()
