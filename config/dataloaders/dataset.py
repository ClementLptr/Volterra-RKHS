import os
import logging
from sklearn.model_selection import train_test_split
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from config.path import Path

import logging

logger = logging.getLogger('VideoClassification')
logger.setLevel(logging.DEBUG)

class VideoDataset(Dataset):
    def __init__(self, dataset='ucf101', split='train', clip_len=16, preprocess=True):
        """
        Video Dataset Loader with improved error handling
        
        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Folder to read from (train/val/test). Defaults to 'train'.
            clip_len (int): Number of frames in each clip. Defaults to 16.
            preprocess (bool): Whether to preprocess dataset. Default is True.
        """
        try:
            # Get dataset paths
            self.root_dir, self.output_dir = Path.db_dir(dataset)
            logger.info(f'Dataset: {dataset}, Root: {self.root_dir}, Output: {self.output_dir}')
        except Exception as e:
            logger.error(f"Path configuration error: {e}")
            raise

        # Dataset parameters
        self.dataset = dataset
        self.clip_len = clip_len
        self.split = split
        
        # Image processing parameters
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        # Preprocessing and data loading
        try:
            if not self.check_integrity():
                logger.error(f'Dataset integrity check failed for {dataset}')
                raise RuntimeError('Dataset not found or corrupted.')
            
            # Force preprocessing by default
            if not preprocess or not self.check_preprocess():
                logger.info(f'Preprocessing {dataset} dataset...')
                self.preprocess()

            # Collect video paths and labels
            self.fnames, self.labels = self._collect_video_paths()

            if not self.fnames:
                logger.error(f'No video files found in {self.output_dir}/{self.split}')
                raise ValueError('No video files found in the dataset')

            # Create label mapping
            self.label2index = {label: index for index, label in enumerate(sorted(set(self.labels)))}
            self.label_array = np.array([self.label2index[label] for label in self.labels], dtype=int)

            logger.info(f'Number of {split} videos: {len(self.fnames)}')

            # Optional: Save labels mapping
            self._save_labels_mapping()
        except Exception as e:
            logger.error(f"Dataset initialization error: {e}")
            raise

    def _collect_video_paths(self):
        """Collect video file paths and their corresponding labels."""
        folder = os.path.join(self.output_dir, self.split)
        fnames, labels = [], []

        for label in sorted(os.listdir(folder)):
            label_path = os.path.join(folder, label)
            if os.path.isdir(label_path):
                for video_dir in os.listdir(label_path):
                    video_path = os.path.join(label_path, video_dir)
                    if os.path.isdir(video_path):
                        fnames.append(video_path)
                        labels.append(label)

        return fnames, labels

    def _save_labels_mapping(self):
        """Save label mappings to a text file."""
        label_files = {
            "ucf101": 'config/dataloaders/ucf_labels.txt',
            "hmdb51": 'config/dataloaders/hmdb_labels.txt'
        }
        
        filename = label_files.get(self.dataset)
        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                for id, label in enumerate(sorted(self.label2index), 1):
                    f.write(f"{id} {label}\n")

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        """
        Returns processed video clip and its label.
        
        Args:
            index (int): Index of video clip
        
        Returns:
            tuple: (video_tensor, label)
        """
        try:
            buffer = self.load_frames(self.fnames[index])
            buffer = self.crop(buffer, self.clip_len, self.crop_size)
            labels = np.array(self.label_array[index])

            if self.split == 'test':
                buffer = self.randomflip(buffer)
            
            buffer = self.normalize(buffer)
            buffer = self.to_tensor(buffer)
            
            return torch.from_numpy(buffer), torch.from_numpy(labels)
        
        except Exception as e:
            logger.error(f"Video processing error at index {index}: {e}")
            raise

    def check_integrity(self):
        """Check if the root dataset directory exists."""
        return os.path.exists(self.root_dir)

    def check_preprocess(self):
        """
        Comprehensively verify preprocessing
        
        Checks:
        - Output directory exists
        - Split subdirectory exists
        - At least one video processed for each class
        """
        logger.debug(f'Checking preprocessing for {self.dataset}')
        
        # Check output directory
        if not os.path.exists(self.output_dir):
            logger.debug(f'Output directory {self.output_dir} does not exist')
            return False
        
        # Check split directory
        split_dir = os.path.join(self.output_dir, self.split)
        if not os.path.exists(split_dir):
            logger.debug(f'Split directory {split_dir} does not exist')
            return False
        
        # Check classes in split directory
        classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        if not classes:
            logger.debug(f'No classes found in {split_dir}')
            return False
        
        # Verify at least one processed video per class
        for cls in classes:
            cls_path = os.path.join(split_dir, cls)
            videos = [v for v in os.listdir(cls_path) if os.path.isdir(os.path.join(cls_path, v))]
            
            if not videos:
                logger.debug(f'No processed videos found for class {cls}')
                return False
            
            # Check first video has frames
            first_video = os.path.join(cls_path, videos[0])
            frames = [f for f in os.listdir(first_video) if f.endswith('.jpg')]
            
            if not frames:
                logger.debug(f'No frames found in video {first_video}')
                return False
            
            # Optionally check frame dimensions
            first_frame = os.path.join(first_video, frames[0])
            try:
                img = cv2.imread(first_frame)
                if img is None or img.shape[0] != self.resize_height or img.shape[1] != self.resize_width:
                    logger.debug(f'Frame {first_frame} has incorrect dimensions')
                    return False
            except Exception as e:
                logger.debug(f'Error reading frame {first_frame}: {e}')
                return False
        
        logger.debug('Preprocessing check passed')
        return True

    def preprocess(self):
        """
        Enhanced preprocessing with detailed logging
        Split dataset and extract frames for each video
        """
        # Ensure output directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.output_dir, split), exist_ok=True)

        # Track processed videos
        processed_count = 0
        skipped_count = 0

        # Iterate through action/class folders
        for action_name in os.listdir(self.root_dir):
            action_path = os.path.join(self.root_dir, action_name)
            
            # Ensure it's a directory
            if not os.path.isdir(action_path):
                continue

            # Find video files
            video_files = [
                f for f in os.listdir(action_path) 
            ]

            # Skip if no videos
            if not video_files:
                logger.warning(f'No videos found in {action_path}')
                continue

            # Split dataset
            try:
                train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
                train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)
            except ValueError:
                logger.warning(f'Not enough videos in {action_name} to split')
                continue

            # Process videos for each split
            for split, videos in [('train', train), ('val', val), ('test', test)]:
                split_dir = os.path.join(self.output_dir, split, action_name)
                os.makedirs(split_dir, exist_ok=True)

                for video in videos:
                    try:
                        self.process_video(video, action_name, split_dir)
                        processed_count += 1
                    except Exception as e:
                        logger.error(f'Failed to process {video}: {e}')
                        skipped_count += 1

        logger.info(f'Preprocessing completed. Processed: {processed_count}, Skipped: {skipped_count}')

    def process_video(self, video, action_name, save_dir):
        """
        Process a single video:
        - Read video 
        - Extract frames
        - Resize frames
        - Save frames
        """
        video_filename = os.path.splitext(video)[0]
        video_save_path = os.path.join(save_dir, video_filename)
        os.makedirs(video_save_path, exist_ok=True)

        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))
        
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Frame extraction parameters
        EXTRACT_FREQUENCY = max(frame_count // 64, 1)  # Ensure at least 1
        
        count, i = 0, 0
        while count < frame_count:
            ret, frame = capture.read()
            if not ret:
                break

            if count % EXTRACT_FREQUENCY == 0:
                # Resize frame
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                
                # Save frame
                cv2.imwrite(
                    filename=os.path.join(video_save_path, f'frame_{i:04d}.jpg'), 
                    img=frame
                )
                i += 1
            
            count += 1

        capture.release()

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