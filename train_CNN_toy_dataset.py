import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from config.dataloaders.toy_dataset import ToyDataset
from config.logger import setup_logger
from network.rgb_of.CNN5Layers import CNN5Layers
import psutil
from sklearn.kernel_approximation import Nystroem
import gc


# Configuration de l'environnement
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
logger = setup_logger()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# LAYER_CONFIGS update for 5 layers
LAYER_CONFIGS = {
    1: (3, 3, 3, 3),
    2: (24, 3, 3, 3),
    3: (32, 3, 3, 3),
    4: (64, 3, 3, 3),
    5: (96, 3, 3, 3),
}

MAX_MEMORY_GB = 8.0

def get_memory_usage():
    """Retourne l'utilisation mémoire en Go."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 ** 3
    return mem

def extract_patches(inputs, patch_size=(3, 3, 3), stride=(1, 1, 1), max_patches=100_000):
    patches = inputs.unfold(2, patch_size[0], stride[0]).unfold(3, patch_size[1], stride[1]).unfold(4, patch_size[2], stride[2])
    patches = patches.contiguous().view(inputs.size(0), -1, *patch_size)
    result = patches.reshape(-1, np.prod(patch_size) * inputs.size(1))
    if result.shape[0] > max_patches:
        indices = np.random.choice(result.shape[0], max_patches, replace=False)
        result = result[indices]
    logger.debug(f"Extracted {result.shape[0]} patches")
    return result

def normalize_patches(patches, clip_value=1e6, epsilon=1e-4):
    patches = np.clip(patches, -clip_value, clip_value)
    mean = np.mean(patches, axis=0, keepdims=True)
    std = np.std(patches, axis=0, keepdims=True)
    normalized = (patches - mean) / (std + epsilon)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)
    return normalized.astype(np.float32)

def compute_kmeans_projectors(data, num_projectors, layer_idx, nystroem_components=100, kernel='rbf'):
    logger.info(f"Projectors: L{layer_idx} - {num_projectors} clusters")
    max_samples = int(MAX_MEMORY_GB * 1024**3 / (data.shape[1] * 4))
    if data.shape[0] > max_samples:
        data = data[np.random.choice(data.shape[0], max_samples, replace=False)]

    data = normalize_patches(data)
    nystroem = Nystroem(kernel=kernel, n_components=nystroem_components, random_state=42)
    data_transformed = nystroem.fit_transform(data)

    if get_memory_usage() > MAX_MEMORY_GB * 0.9:
        logger.warning("Reducing data post-Nystroem due to memory pressure")
        data_transformed = data_transformed[np.random.choice(data_transformed.shape[0], int(data_transformed.shape[0] * 0.5), replace=False)]

    kmeans = MiniBatchKMeans(n_clusters=num_projectors, random_state=42, batch_size=1000, max_iter=100)
    kmeans.fit(data_transformed)
    centroids = kmeans.cluster_centers_
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-6)

    flat_dim = np.prod(LAYER_CONFIGS[layer_idx])
    if centroids.shape[1] < flat_dim:
        centroids = np.pad(centroids, ((0, 0), (0, flat_dim - centroids.shape[1])), mode='constant')
    else:
        centroids = centroids[:, :flat_dim]

    projectors = torch.tensor(centroids.reshape((num_projectors, *LAYER_CONFIGS[layer_idx])), dtype=torch.float32).to(device)
    
    del data, data_transformed, kmeans, centroids
    gc.collect()
    torch.cuda.empty_cache()

    return projectors

def train_model(dataset='hmdb51', num_classes=10, lr=1e-4, num_epochs=100, batch_size=8, clip_len=16, patience=30, num_projectors=512):
    """Entraîne un modèle RGB sur HMDB51 avec calcul des projecteurs pendant l'entraînement."""
    logger.info(f"Using device: {device}")
    
    # Initialisation du modèle avec projecteurs aléatoires
    pre_determined_projectors = {layer: torch.randn(num_projectors, *LAYER_CONFIGS[layer]).to(device) * 0.1 for layer in range(1, 6)}
    model = CNN5Layers(
        num_classes=num_classes,
        input_shape=(16, 112, 112),
        num_channels=3,
        dropout_rate=0.5,
        pretrained=False,
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    
    # Dataloaders
    train_dataloader = DataLoader(
        ToyDataset(dataset=dataset, split='train', clip_len=clip_len),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_dataloader = DataLoader(
        ToyDataset(dataset=dataset, split='val', clip_len=clip_len),
        batch_size=batch_size, num_workers=4, pin_memory=True
    )
    test_dataloader = DataLoader(
        ToyDataset(dataset=dataset, split='test', clip_len=clip_len),
        batch_size=16, num_workers=4
    )
    
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Train videos: {len(train_dataloader.dataset)}, Val videos: {len(val_dataloader.dataset)}, Test videos: {len(test_dataloader.dataset)}")
    
    # Sauvegarde et early stopping
    best_val_acc = 0.0
    patience_counter = 0
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'default_version')
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"best_model_{dataset}.pth.tar")
    projectors_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'projectors')
    os.makedirs(projectors_dir, exist_ok=True)
    
    # Boucle d'entraînement
    for epoch in range(num_epochs):
        # Entraînement d'une époque
        model.train()
        running_loss = 0.0
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        logger.info(f"Epoch {epoch + 1} | Loss: {running_loss / len(train_dataloader):.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_acc = correct / total
        logger.info(f"Epoch {epoch + 1} | Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Model saved at epoch {epoch + 1}")

        # Scheduler Step
        scheduler.step()

        if patience_counter > patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
        if val_acc < best_val_acc:
            patience_counter += 1
        else:
            patience_counter = 0

if __name__ == "__main__":
    train_model()