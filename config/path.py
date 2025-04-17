import os

class Path:
    @staticmethod
    def db_dir(database):
        """
        Define root directories for different video datasets
        
        Args:
            database (str): Name of the dataset (e.g., 'ucf101', 'hmdb51')
        
        Returns:
            tuple: (root directory, output directory)
        """
        if database == 'ucf101':
            # Modify these paths according to your actual dataset locations
            root_dir = './data/UCF-101/videos'
            output_dir = '.data/UCF-101/processed'
        elif database == 'hmdb51':
            # Modify these paths according to your actual dataset locations
            root_dir = './data/HMDB51/videos'
            output_dir = './data/HMDB51/preprocessed'
        else:
            raise ValueError(f"Dataset {database} not supported")
        
        return root_dir, output_dir
    @staticmethod
    def model_dir():
        return 'models/hmdb51/c3d-pretrained.pth'