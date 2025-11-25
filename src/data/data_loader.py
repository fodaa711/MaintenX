import pandas as pd
import numpy as np
from pathlib import Path

class LocalDataLoader:
    """
    Data loader for local predictive maintenance datasets
    Loads data from ./data/raw/*.csv directory structure
    """
    
    def __init__(self, base_path="./data/raw"):
        """
        Initialize DataLoader with local file paths
        
        Parameters:
        - base_path: Root directory containing raw CSV files
        """
        self.base_path = Path(base_path)
        self.file_mapping = {
            'telemetry': 'PdM_telemetry.csv',
            'errors': 'PdM_errors.csv', 
            'maintenance': 'PdM_maint.csv',
            'failures': 'PdM_failures.csv',
            'machines': 'PdM_machines.csv'
        }
        
        # Validate that data directory exists
        if not self.base_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.base_path}")
    
    def load_data(self, data_type):
        """
        Load specific dataset from local files
        
        Parameters:
        - data_type: One of ['telemetry', 'errors', 'maintenance', 'failures', 'machines']
        
        Returns:
        - pandas DataFrame with loaded data
        """
        if data_type not in self.file_mapping:
            raise ValueError(f"Invalid data_type. Must be one of {list(self.file_mapping.keys())}")
        
        file_name = self.file_mapping[data_type]
        file_path = self.base_path / file_name
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        print(f"Loading {data_type} data from: {file_path}")
        return pd.read_csv(file_path)
    
    def load_all_data(self):
        """
        Load all predictive maintenance datasets
        
        Returns:
        - Dictionary of DataFrames with all datasets
        """
        datasets = {}
        for key in self.file_mapping.keys():
            try:
                datasets[key] = self.load_data(key)
                print(f" Successfully loaded {key}: {datasets[key].shape}")
            except Exception as e:
                print(f"Error loading {key}: {e}")
        
        return datasets
    
    def save_processed_data(self, data, filename, folder="processed"):
        """
        Save processed data to specified folder
        
        Parameters:
        - data: DataFrame to save
        - filename: Name of the file (without extension)
        - folder: Subfolder in data directory ('features' or 'processed')
        """
        save_path = self.base_path.parent / folder / f"{filename}.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_csv(save_path, index=False)
        print(f"Saved {filename} to: {save_path}")
        
        return save_path