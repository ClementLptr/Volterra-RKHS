import os
from config.dataloaders.dataset import VideoDataset
from config.dataloaders.toy_dataset import ToyDataset
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from config.logger import setup_logger
from network.rgb_of.vnn_rgb_of_RKHS_conv_bloc import RKHS_VNN
import psutil
from sklearn.kernel_approximation import Nystroem
import gc


# Configuration de l'environnement
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
logger = setup_logger()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LAYER_CONFIGS = {
    1: (3, 3, 3, 3),
    2: (16, 3, 3, 3),
    3: (24, 3, 3, 3),
    4: (48, 3, 3, 3),
    5: (64, 3, 3, 3),
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

def compute_projectors_for_layer(layer_idx, num_projectors, dataloader, model, device, max_patches=100_000, nystroem_components=100, kernel='rbf'):
    logger.info(f"Extracting patches for layer {layer_idx}")
    all_patches = []

    model.eval()
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device, non_blocking=True)

            if layer_idx == 1:
                patches = extract_patches(inputs, max_patches=max_patches // len(dataloader)).cpu().numpy()
            else:
                output = model.forward_up_to_layer(inputs, layer_idx - 1)
                patches = extract_patches(output, max_patches=max_patches // len(dataloader)).cpu().numpy()
            all_patches.append(patches)

            if sum(p.shape[0] for p in all_patches) >= max_patches:
                break
            if layer_idx == 1:
                del inputs
            else:
                del inputs, output
            torch.cuda.empty_cache()

    all_patches = np.vstack(all_patches)[:max_patches]
    
    if get_memory_usage() > MAX_MEMORY_GB * 0.8:
        logger.warning("Subsampling due to memory pressure")
        all_patches = all_patches[np.random.choice(all_patches.shape[0], all_patches.shape[0] // 2, replace=False)]

    return compute_kmeans_projectors(all_patches, num_projectors, layer_idx, nystroem_components, kernel)

def train_model(dataset='hmdb51', num_classes=10, lr=1e-4, num_epochs=100, batch_size=8, clip_len=16, patience=30, num_projectors=256):
    """Entraîne un modèle RGB sur HMDB51 avec calcul des projecteurs pendant l'entraînement."""
    logger.info(f"Using device: {device}")
    
    # Initialisation du modèle avec projecteurs aléatoires
    pre_determined_projectors = {layer: torch.randn(num_projectors, *LAYER_CONFIGS[layer]).to(device) * 0.1 for layer in range(1, 6)}
    model = RKHS_VNN(
        num_classes=num_classes,
        num_projector=num_projectors,
        num_ch=3,
        input_shape=(16, 112, 112),
        dropout_rate=0.5,
        pretrained=False,
        pre_determined_projectors=pre_determined_projectors
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
        # Calcul des projecteurs à la première époque et toutes les 30 epochs
        # Dans la boucle d'entraînement, à l'époque 0
        if epoch % 5 == 0:
            logger.info(f"Computing projectors at epoch {epoch + 1}")
            for layer in range(1, 4):
                projectors = compute_projectors_for_layer(
                    layer_idx=layer,
                    num_projectors=num_projectors,
                    dataloader=train_dataloader,
                    model=model,
                    device=device,
                    nystroem_components=100,  # Ajuster selon vos besoins
                    kernel='rbf'             # Peut être 'linear', 'poly', etc.
                )
                proj_file = os.path.join(projectors_dir, f"projectors_layer_{layer}_P{num_projectors}_epoch{epoch + 1}.pt")
                torch.save(projectors, proj_file)
                logger.info(f"Saved projectors for layer {layer} to {proj_file}")
                model.update_projectors(projectors, layer)
                logger.info(f"Updated model with new projectors")
        
        # Entraînement
        model.train()
        train_loss = 0.0
        train_corrects = 0.0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
        
        train_loss = train_loss / len(train_dataloader.dataset)
        train_acc = train_corrects.double() / len(train_dataloader.dataset)
        logger.info(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0.0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_dataloader.dataset)
        val_acc = val_corrects.double() / len(val_dataloader.dataset)
        logger.info(f"Epoch {epoch + 1}/{num_epochs} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Sauvegarde du meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
                'projectors': pre_determined_projectors
            }, checkpoint_path)
            logger.info(f"Saved best model at epoch {epoch + 1} with val top-1 acc: {val_acc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement.")
            break
        
        scheduler.step()
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Chargement du meilleur modèle pour le test
    logger.info("Loading best model for test evaluation...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.update_projectors(checkpoint['projectors'])
    
    # Évaluation sur le test set
    model.eval()
    test_loss = 0.0
    test_corrects = 0.0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)
    
    test_loss = test_loss / len(test_dataloader.dataset)
    test_acc = test_corrects.double() / len(test_dataloader.dataset)
    logger.info(f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

if __name__ == "__main__":
    train_model()