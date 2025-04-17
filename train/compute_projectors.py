import os
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import argparse
from config.dataloaders.dataset import VideoDataset
from config.logger import setup_logger
import time
import matplotlib.pyplot as plt
from network.rgb_of.vnn_rgb_of_RKHS import RKHS_VNN
import psutil

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
logger = setup_logger()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LAYER_CONFIGS = {
    1: (3, 3, 3, 3),
    2: (24, 3, 3, 3),
    3: (48, 3, 3, 3),
    4: (96, 3, 3, 3),
    5: (192, 3, 3, 3),
}

MAX_MEMORY_GB = 8.0

def get_memory_usage():
    """Retourne l'utilisation mémoire en Go."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 ** 3
    return mem

def extract_patches(inputs, patch_size=(3, 3, 3), stride=(1, 1, 1), max_patches=100_000):
    logger.debug(f"Extracting patches with size {patch_size} and stride {stride}")
    patches = inputs.unfold(2, patch_size[0], stride[0]).unfold(3, patch_size[1], stride[1]).unfold(4, patch_size[2], stride[2])
    patches = patches.contiguous().view(inputs.size(0), -1, *patch_size)
    result = patches.reshape(-1, np.prod(patch_size) * inputs.size(1))
    if result.shape[0] > max_patches:
        indices = np.random.choice(result.shape[0], max_patches, replace=False)
        result = result[indices]
    logger.debug(f"Extracted {result.shape[0]} patches with {result.shape[1]} features")
    return result

def stratified_sample(dataset, num_samples=500, num_classes=51):
    logger.info(f"Performing stratified sampling for {num_samples} samples across {num_classes} classes")
    labels = [dataset[i][1] for i in range(len(dataset))]
    samples_per_class = max(1, num_samples // num_classes)
    indices = []
    for cls in range(num_classes):
        cls_indices = [i for i, lbl in enumerate(labels) if lbl == cls]
        sampled = np.random.choice(cls_indices, min(samples_per_class, len(cls_indices)), replace=False)
        indices.extend(sampled)
    sampled_dataset = Subset(dataset, indices[:num_samples])
    logger.info(f"Stratified sample complete: {len(sampled_dataset)} samples selected")
    return sampled_dataset

def normalize_patches(patches):
    """Normalise les patches pour avoir une moyenne=0 et variance=1 par dimension."""
    mean = patches.mean(axis=0, keepdims=True)
    std = patches.std(axis=0, keepdims=True) + 1e-6  # Éviter division par zéro
    return (patches - mean) / std

def kernel_alignment(data, projectors, degree=2, subsample_size=500):
    logger.debug(f"Computing kernel alignment with subsample size {subsample_size}")
    if data.shape[0] > subsample_size:
        indices = np.random.choice(data.shape[0], subsample_size, replace=False)
        data = data[indices]
    true_kernel = polynomial_kernel(data, degree=degree, coef0=1)
    data_proj_kernel = polynomial_kernel(data, projectors, degree=degree, coef0=1)
    approx_kernel = data_proj_kernel @ data_proj_kernel.T
    norm_true = np.linalg.norm(true_kernel, 'fro')
    norm_approx = np.linalg.norm(approx_kernel, 'fro')
    alignment = np.sum(true_kernel * approx_kernel) / (norm_true * norm_approx + 1e-6)
    logger.debug(f"Kernel alignment: {alignment:.4f}")
    return alignment

def compute_reconstruction_error(data, projectors, degree=2, subsample_size=1000):
    logger.debug(f"Computing reconstruction error with subsample size {subsample_size}")
    if data.shape[0] > subsample_size:
        indices = np.random.choice(data.shape[0], subsample_size, replace=False)
        data = data[indices]
    true_kernel = polynomial_kernel(data, degree=degree, coef0=1)
    data_proj_kernel = polynomial_kernel(data, projectors, degree=degree, coef0=1)
    approx_kernel = data_proj_kernel @ data_proj_kernel.T
    error = np.mean((true_kernel - approx_kernel) ** 2)
    logger.debug(f"Reconstruction error (Volterra order 2): {error:.4f}")
    return error

def compute_kernel_kmeans_patches(data, num_projectors, degree=2, n_components=81):
    logger.info(f"Starting kernel k-means for {num_projectors} projectors with {data.shape[0]} samples")
    max_samples = int(MAX_MEMORY_GB * 1024 ** 3 / (n_components * 4))
    if data.shape[0] > max_samples:
        logger.info(f"Subsampling data from {data.shape[0]} to {max_samples}")
        indices = np.random.choice(data.shape[0], max_samples, replace=False)
        data = data[indices]
    
    # Normalisation des données avant Nyström
    data = normalize_patches(data)
    
    nystroem = Nystroem(kernel='polynomial', degree=degree, coef0=1, n_components=n_components, random_state=42)
    logger.debug(f"Applying Nyström with {n_components} components, Memory: {get_memory_usage():.2f} GB")
    feature_map = nystroem.fit_transform(data)
    logger.debug(f"Nyström feature map: {feature_map.shape}, Memory: {get_memory_usage():.2f} GB")

    kmeans = MiniBatchKMeans(n_clusters=num_projectors, random_state=42, batch_size=1000, max_iter=100, compute_labels=True)
    kmeans.fit(feature_map)
    labels = kmeans.labels_

    centroids = kmeans.cluster_centers_
    # Simplification : une seule normalisation par la norme L2
    normalized_centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-6)

    silhouette = silhouette_score(feature_map[:1000], labels[:1000], metric='euclidean')
    recon_error = compute_reconstruction_error(feature_map, normalized_centroids, degree=degree)
    alignment = kernel_alignment(feature_map, normalized_centroids, degree=degree)
    
    logger.info(f"Nyström results: Sil={silhouette:.4f}, Recon={recon_error:.4f}, Align={alignment:.4f}")
    return normalized_centroids, silhouette, recon_error, alignment

def random_fourier_features(data, num_projectors, degree=2, sigma=1.0, n_components=81):
    logger.info(f"Starting RFF for {num_projectors} projectors with {data.shape[0]} samples")
    np.random.seed(42)
    dim = data.shape[1]
    max_samples = int(MAX_MEMORY_GB * 1024 ** 3 / (num_projectors * 4))
    if data.shape[0] > max_samples:
        logger.info(f"Subsampling data from {data.shape[0]} to {max_samples}")
        indices = np.random.choice(data.shape[0], max_samples, replace=False)
        data = data[indices]
    
    # Normalisation des données avant RFF
    data = normalize_patches(data)
    
    n_cos = (n_components + 1) // 2
    n_sin = n_components - n_cos
    W = np.random.normal(0, 1/sigma, (dim, n_cos))
    b = np.random.uniform(0, 2 * np.pi, n_cos)
    proj = np.dot(data, W) + b
    rff = np.hstack([np.cos(proj), np.sin(proj)[:, :n_sin]]) * np.sqrt(2 / n_components)
    logger.debug(f"RFF feature map: {rff.shape}, Memory: {get_memory_usage():.2f} GB")

    kmeans = MiniBatchKMeans(n_clusters=num_projectors, random_state=42, batch_size=1000, max_iter=100, compute_labels=True)
    kmeans.fit(rff)
    labels = kmeans.labels_

    centroids = kmeans.cluster_centers_
    normalized_centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-6)

    silhouette = silhouette_score(rff[:1000], labels[:1000], metric='euclidean')
    recon_error = compute_reconstruction_error(rff, normalized_centroids, degree=degree)
    alignment = kernel_alignment(rff, normalized_centroids, degree=degree)
    
    logger.info(f"RFF results: Sil={silhouette:.4f}, Recon={recon_error:.4f}, Align={alignment:.4f}")
    return normalized_centroids, silhouette, recon_error, alignment

def compute_random_projectors(data, num_projectors, target_shape, degree=2):
    logger.info(f"Computing random projectors for {num_projectors}")
    dim = data.shape[1]
    # Normalisation des données avant Random
    data = normalize_patches(data)
    
    random_centroids = np.random.randn(num_projectors, dim)
    random_centroids = random_centroids / (np.linalg.norm(random_centroids, axis=1, keepdims=True) + 1e-6)

    recon_error = compute_reconstruction_error(data, random_centroids, degree=degree)
    alignment = kernel_alignment(data, random_centroids, degree=degree)
    silhouette = silhouette_score(data[:1000], MiniBatchKMeans(n_clusters=num_projectors, random_state=42).fit(data[:1000]).labels_)
    
    logger.info(f"Random projector results: Sil={silhouette:.4f}, Recon={recon_error:.4f}, Align={alignment:.4f}")
    
    flat_dim = np.prod(target_shape[1:])
    centroids_padded = np.pad(random_centroids, ((0, 0), (0, flat_dim - random_centroids.shape[1])), mode='constant') if flat_dim > random_centroids.shape[1] else random_centroids[:, :flat_dim]
    return torch.tensor(centroids_padded.reshape(target_shape), dtype=torch.float32).to(device), silhouette, recon_error, alignment

def compute_projectors_for_layer(layer_idx, P, train_dataloader, model_temp, device=device, save_dir="results/projectors"):
    start_time = time.time()
    logger.info(f"Starting projector computation for Layer {layer_idx}, P={P}")

    max_patches = 100_000
    all_patches = []

    if layer_idx == 1:
        dataset = stratified_sample(train_dataloader.dataset, num_samples=2000)
        logger.info("Extracting patches for Layer 1")
        for inputs, _ in DataLoader(dataset, batch_size=16, num_workers=2, pin_memory=True):
            inputs = inputs.to(device, non_blocking=True)
            patches = extract_patches(inputs, max_patches=max_patches // len(dataset)).cpu().numpy()
            all_patches.append(patches)
            if len(all_patches) * patches.shape[0] >= max_patches:
                break
        all_patches = np.vstack(all_patches)[:max_patches]
    
    else:
        logger.info(f"Extracting patches for Layer {layer_idx}")
        with torch.no_grad():
            for inputs, _ in train_dataloader:
                inputs = inputs.to(device, non_blocking=True)
                output = model_temp.forward_up_to_layer(inputs, layer_idx - 1)
                patches = extract_patches(output, max_patches=max_patches // len(train_dataloader.dataset)).cpu().numpy()
                all_patches.append(patches)
                if len(all_patches) * patches.shape[0] >= max_patches:
                    break
        all_patches = np.vstack(all_patches)[:max_patches]
    logger.info(f"Patches extracted: {all_patches.shape[0]} patches, Memory: {get_memory_usage():.2f} GB")
    
    if get_memory_usage() > MAX_MEMORY_GB * 0.8:
        logger.warning(f"Memory usage {get_memory_usage():.2f} GB approaching limit {MAX_MEMORY_GB} GB, reducing data")
        indices = np.random.choice(all_patches.shape[0], int(all_patches.shape[0] * 0.5), replace=False)
        all_patches = all_patches[indices]

    logger.info(f"Patches extracted: {all_patches.shape[0]} patches, Memory: {get_memory_usage():.2f} GB")
    
    logger.info("Computing Nyström + RFF projectors")
    nystrom_centroids, nyst_sil, nyst_recon, nyst_align = compute_kernel_kmeans_patches(all_patches, P // 2, n_components=81)
    rff_centroids, rff_sil, rff_recon, rff_align = random_fourier_features(all_patches, P // 2, n_components=81)
    
    assert nystrom_centroids.shape[1] == rff_centroids.shape[1], f"Dimension mismatch: {nystrom_centroids.shape} vs {rff_centroids.shape}"
    
    centroids = np.vstack([nystrom_centroids, rff_centroids])
    target_shape = (P, *LAYER_CONFIGS[layer_idx])
    flat_dim = np.prod(LAYER_CONFIGS[layer_idx])
    centroids_padded = np.pad(centroids, ((0, 0), (0, flat_dim - centroids.shape[1])), mode='constant') if flat_dim > centroids.shape[1] else centroids[:, :flat_dim]
    projectors = torch.tensor(centroids_padded.reshape(target_shape), dtype=torch.float32).to(device)

    avg_silhouette = (nyst_sil + rff_sil) / 2
    avg_recon_error = (nyst_recon + rff_recon) / 2
    avg_alignment = (nyst_align + rff_align) / 2

    logger.info("Computing random projectors")
    rand_proj, rand_sil, rand_recon, rand_align = compute_random_projectors(all_patches, P, target_shape)
    
    results = {
        'P': P,
        'recon_error': avg_recon_error,
        'silhouette': avg_silhouette,
        'alignment': avg_alignment,
        'recon_error_rand': rand_recon,
        'silhouette_rand': rand_sil,
        'alignment_rand': rand_align,
        'time': time.time() - start_time
    }
    
    # Ensure files are saved in save_dir
    proj_file = os.path.join(save_dir, f"projectors_layer_{layer_idx}_P{P}.pt")
    metrics_file = os.path.join(save_dir, f"metrics_layer_{layer_idx}_P{P}.npy")
    torch.save(projectors, proj_file)
    np.save(metrics_file, results)
    logger.info(f"Saved to {proj_file} and {metrics_file}, Memory: {get_memory_usage():.2f} GB")
    
    return projectors, results

def load_previous_projectors(layer_idx, P, save_dir):
    """Charge les projecteurs précédemment calculés pour les couches inférieures."""
    projectors = {}
    for i in range(1, layer_idx):
        proj_file = os.path.join(save_dir, f"projectors_layer_{i}_P{P}.pt")
        if os.path.exists(proj_file):
            projectors[i] = torch.load(proj_file, map_location=device)
            logger.info(f"Loaded projectors for Layer {i} from {proj_file}")
        else:
            logger.warning(f"Projectors for Layer {i} not found at {proj_file}, using random initialization")
            projectors[i] = torch.randn(P, *LAYER_CONFIGS[i]).to(device) * 0.1
    return projectors

def main():
    parser = argparse.ArgumentParser(description="Compute projectors for a specific layer on SLURM")
    parser.add_argument('--layer', type=int, required=True, help="Layer index (1-5)")
    parser.add_argument('--P', type=int, required=True, help="Number of projectors")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save projectors and metrics")
    args = parser.parse_args()

    logger.info(f"Starting job: Layer={args.layer}, P={args.P}, Save Dir={args.save_dir}")
    os.makedirs(args.save_dir, exist_ok=True)

    train_dataloader = DataLoader(
        VideoDataset(dataset='hmdb51', split='train', clip_len=16),
        batch_size=16, shuffle=True, num_workers=2, pin_memory=True
    )
    
    # Charger les projecteurs pour les couches inférieures
    temp_projectors = load_previous_projectors(args.layer, args.P, args.save_dir)
    
    # Ajouter une initialisation aléatoire pour les couches non chargées (y compris la couche courante et supérieures)
    for i in range(1, 6):
        if i not in temp_projectors:
            temp_projectors[i] = torch.randn(args.P, *LAYER_CONFIGS[i]).to(device) * 0.1
            
    model_temp = RKHS_VNN(
        num_classes=51, num_projector=args.P, num_ch=3, input_shape=(16, 112, 112),
        dropout_rate=0.5, pretrained=False, pre_determined_projectors=temp_projectors
    ).to(device)
    
    logger.info(f"Initial memory usage: {get_memory_usage():.2f} GB")
    projectors, results = compute_projectors_for_layer(args.layer, args.P, train_dataloader, model_temp, device, args.save_dir)
    logger.info(f"Job completed, Memory: {get_memory_usage():.2f} GB")

if __name__ == "__main__":
    main()