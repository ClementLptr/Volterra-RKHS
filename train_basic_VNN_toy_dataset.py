import os
from config.dataloaders.toy_dataset import ToyDataset
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from config.logger import setup_logger
from config.dataloaders.dataset import VideoDataset
from network.rgb_of import vnn_rgb_of_highQ

# Configuration de l'environnement
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train_model(dataset='hmdb51', num_classes=10, lr=1e-4, num_epochs=30, batch_size=8, clip_len=16, patience=10):
    """Entraîne un modèle RGB sur HMDB51 avec early stopping et évalue sur le test set avec la meilleure précision top-1."""
    # Initialisation
    logger = setup_logger()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Modèle
    model = vnn_rgb_of_highQ.VNN(num_classes=num_classes, num_ch=3, pretrained=False).to(device)
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
        ToyDataset(dataset=dataset, split='test', clip_len=16),
        batch_size=16, num_workers=4
    )

    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Train videos: {len(train_dataloader.dataset)}, Val videos: {len(val_dataloader.dataset)}, Test videos: {len(test_dataloader.dataset)}")

    # Sauvegarde du meilleur modèle et early stopping
    best_val_acc = 0.0
    patience_counter = 0
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'default_version')
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"best_model_{dataset}.pth.tar")

    # Boucle d'entraînement
    for epoch in range(num_epochs):
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
        logger.info(f"Epoch {epoch + 1}/{num_epochs} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} ")

        # Sauvegarde du meilleur modèle basé sur la précision top-1
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
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

    # Évaluation sur le test set avec le meilleur modèle
    logger.info("Starting test evaluation...")
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