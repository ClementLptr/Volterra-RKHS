
from config.logger import setup_logger
from torch.utils.data import Dataset
from config.dataloaders.dataset import VideoDataset
import os

# Set environment variable for CUDA allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Setup logger
logger = setup_logger()

class ToyDataset(Dataset):
    """A wrapper dataset to filter HMDB-51 to a subset of classes."""
    def __init__(self, dataset='hmdb51', split='train', clip_len=16, preprocess=True, selected_classes=["brush_hair", "clap", "jump"]):
        """
        Initialize the ToyDataset to filter specific classes from HMDB-51.

        Args:
            dataset (str): Name of dataset (e.g., 'hmdb51'). Defaults to 'hmdb51'.
            split (str): Dataset split ('train', 'val', 'test'). Defaults to 'train'.
            clip_len (int): Number of frames in each clip. Defaults to 16.
            preprocess (bool): Whether to preprocess dataset. Defaults to True.
            selected_classes (list): List of class names to include. Defaults to ["brush_hair", "clap", "jump"].
        """
        self.dataset_name = dataset
        self.split = split
        self.clip_len = clip_len
        self.selected_classes = selected_classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}

        # Initialize VideoDataset
        try:
            self.dataset = VideoDataset(dataset=dataset, split=split, clip_len=clip_len, preprocess=preprocess)
            logger.info(f"Initialized VideoDataset for {dataset}, split: {split}, clip_len: {clip_len}")
        except Exception as e:
            logger.error(f"Failed to initialize VideoDataset: {e}")
            raise

        # Get class names from VideoDataset's label2index
        try:
            self.dataset_classes = sorted(self.dataset.label2index.keys())  # List of class names
            logger.debug(f"Available classes: {self.dataset_classes}")
        except AttributeError:
            logger.warning("VideoDataset.label2index not available. Assuming integer labels.")
            self.dataset_classes = None

        # Validate selected classes
        if self.dataset_classes:
            invalid_classes = [cls for cls in selected_classes if cls not in self.dataset_classes]
            if invalid_classes:
                logger.error(f"Selected classes {invalid_classes} not found in dataset")
                raise ValueError(f"Invalid classes: {invalid_classes}")

        # Filter indices for selected classes
        self.indices = []
        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]
            if self.dataset_classes:
                class_name = self.dataset_classes[label]
                if class_name in self.selected_classes:
                    self.indices.append(idx)
            else:
                # Fallback: Assume first len(selected_classes) labels
                if label < len(selected_classes):
                    self.indices.append(idx)

        if not self.indices:
            logger.error(f"No samples found for classes {selected_classes} in {split} split")
            raise ValueError("No samples found for the selected classes")

        logger.info(f"ToyDataset created with {len(self.indices)} samples for classes: {selected_classes}, split: {split}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        dataset_idx = self.indices[idx]
        video, label = self.dataset[dataset_idx]

        # Remap label to new index
        if self.dataset_classes:
            class_name = self.dataset_classes[label]
            new_label = self.class_to_idx[class_name]
        else:
            # Fallback: Assume labels 0, 1, 2 map directly
            new_label = label

        logger.debug(f"Sample {idx}: Class {class_name if self.dataset_classes else label}, New label {new_label}, Video shape {video.shape}")
        return video, new_label