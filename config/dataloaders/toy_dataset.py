import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from config.logger import setup_logger
from config.path import Path

# Set environment variable for CUDA allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Setup logger
logger = setup_logger()

class ToyDataset(Dataset):
    """A dataset for HMDB-51 with a subset of classes, built from scratch."""
    def __init__(self, dataset="hmdb51", split='train', clip_len=16, preprocess=True, selected_classes=["brush_hair", "clap", "jump", "dive", "golf", "kick_ball", "pick", "pullup", "pushup", "situp"]):
        """
        Initialize the ToyDataset for specific HMDB-51 classes.

        Args:
            split (str): Dataset split ('train', 'val', 'test'). Defaults to 'train'.
            clip_len (int): Number of frames in each clip. Defaults to 16.
            preprocess (bool): Whether to preprocess videos into frames. Defaults to True.
            selected_classes (list): List of class names to include. Defaults to ["brush_hair", "clap", "jump"].
        """
        self.split = split
        self.clip_len = clip_len
        self.selected_classes = selected_classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
        try:
            # Get dataset paths
            self.root_dir, self.output_dir = Path.db_dir(dataset)
            logger.info(f'Dataset: {dataset}, Root: {self.root_dir}, Output: {self.output_dir}')
        except Exception as e:
            logger.error(f"Path configuration error: {e}")
            raise
        
        # Image processing parameters
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        # Collect video paths and labels
        self.video_paths, self.labels = self._collect_video_paths()
        if not self.video_paths:
            logger.error(f"No videos found for classes {selected_classes} in {split} split")
            raise ValueError("No videos found for the selected classes")

        logger.info(f"ToyDataset created with {len(self.video_paths)} samples for classes: {selected_classes}, split: {split}")

    def _collect_video_paths(self):
        """Collect video file paths and labels for the selected classes."""
        video_paths = []
        labels = []
        split_dir = os.path.join(self.output_dir, self.split)

        for class_name in self.selected_classes:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                logger.warning(f"Class directory {class_dir} not found")
                continue

            for video_dir in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_dir)
                if os.path.isdir(video_path):
                    video_paths.append(video_path)
                    labels.append(class_name)

        return video_paths, labels

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        """Return a processed video clip and its label."""
        try:
            buffer = self.load_frames(self.video_paths[idx])
            buffer = self.crop(buffer, self.clip_len, self.crop_size)
            if self.split == 'test':
                buffer = self.randomflip(buffer)
            buffer = self.normalize(buffer)
            buffer = self.to_tensor(buffer)
            label = self.class_to_idx[self.labels[idx]]
            return torch.from_numpy(buffer), torch.tensor(label)
        except Exception as e:
            logger.error(f"Error processing video at index {idx}: {e}")
            raise

    def load_frames(self, file_dir):
        """Load frames from a video directory."""
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir) if img.endswith('.jpg')])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame
        return buffer

    def randomflip(self, buffer):
        """Randomly flip video horizontally."""
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.flip(frame, 1)
        return buffer

    def normalize(self, buffer):
        """Normalize frames by subtracting mean values."""
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame
        return buffer

    def to_tensor(self, buffer):
        """Convert buffer to tensor format (channels, frames, height, width)."""
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        """Load frames from a video directory."""
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        """
        Randomly crop video clip.
        
        Args:
            buffer (np.array): Input video frames
            clip_len (int): Number of frames to extract
            crop_size (int): Size of spatial crop
        
        Returns:
            np.array: Cropped video clip
        """
        # Temporal jittering
        time_index = np.random.randint(max(buffer.shape[0] - clip_len, 1))
        
        # Spatial cropping
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop buffer
        cropped_buffer = buffer[
            time_index:time_index + clip_len,
            height_index:height_index + crop_size,
            width_index:width_index + crop_size, 
            :
        ]

        return cropped_buffer