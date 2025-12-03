#!/usr/bin/env python3

import os
import sys
import json
import h5py
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add project root to sys.path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.planning_problem_utils_2d import (
    get_block_env_configs,
    get_gap_env_configs,
    get_random_2d_env_configs,
    get_block_problem_input,
    get_gap_problem_input,
    get_random_2d_problem_input
)
from datasets.point_cloud_mask_utils import (
    generate_rectangle_point_cloud,
    ellipsoid_point_cloud_sampling
)


class PointCloudDatasetGenerator:
    """Generates point cloud datasets for 2D planning problems."""
    
    def __init__(self, config):
        """
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.output_dir = config.get('output_dir', '../NAMR-RRT/data/random_2d')
        self.n_points = config.get('n_points', 1024)  # PointNet++ standard
        self.test_size = config.get('test_size', 0.2)
        self.val_size = config.get('val_size', 0.1)
        self.random_state = config.get('random_state', 42)
        
        # Create output directories
        self.train_dir = os.path.join(self.output_dir, 'train')
        self.val_dir = os.path.join(self.output_dir, 'val')
        self.test_dir = os.path.join(self.output_dir, 'test')
        
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def generate_block_gap_datasets(self):
        """Generate datasets for block and gap environments."""
        print("Generating block and gap datasets...")
        
        # Get environment configurations
        block_configs = get_block_env_configs()
        gap_configs = get_gap_env_configs()
        
        all_problems = []
        all_point_clouds = []
        all_labels = []
        
        # Process block environments
        print("Processing block environments...")
        for i, config in enumerate(tqdm(block_configs)):
            problem = get_block_problem_input(config)
            point_cloud = self._generate_point_cloud_for_problem(problem)
            
            if point_cloud is not None:
                all_problems.append(problem)
                all_point_clouds.append(point_cloud)
                all_labels.append(0)  # 0 for block environments
        
        # Process gap environments
        print("Processing gap environments...")
        for i, config in enumerate(tqdm(gap_configs)):
            problem = get_gap_problem_input(config)
            point_cloud = self._generate_point_cloud_for_problem(problem)
            
            if point_cloud is not None:
                all_problems.append(problem)
                all_point_clouds.append(point_cloud)
                all_labels.append(1)  # 1 for gap environments
        
        # Split dataset
        self._save_datasets(all_point_clouds, all_labels, all_problems, "block_gap")
    
    def generate_random_2d_datasets(self):
        """Generate datasets for random 2D environments."""
        print("Generating random 2D datasets...")
        
        # Get environment configurations
        random_configs = get_random_2d_env_configs()
        
        all_problems = []
        all_point_clouds = []
        all_labels = []
        
        print("Processing random 2D environments...")
        for i, config in enumerate(tqdm(random_configs)):
            problem = get_random_2d_problem_input(config)
            point_cloud = self._generate_point_cloud_for_problem(problem)
            
            if point_cloud is not None:
                all_problems.append(problem)
                all_point_clouds.append(point_cloud)
                all_labels.append(2)  # 2 for random 2D environments
        
        # Split dataset
        self._save_datasets(all_point_clouds, all_labels, all_problems, "random_2d")
    
    def _generate_point_cloud_for_problem(self, problem):
        """Generate point cloud for a planning problem."""
        binary_mask = problem['binary_mask']
        start_point = np.array(problem['x_start'])
        goal_point = np.array(problem['x_goal'])
        
        try:
            # Try ellipse sampling first (focused on the region between start and goal)
            point_cloud = ellipsoid_point_cloud_sampling(
                start_point=start_point,
                goal_point=goal_point,
                max_min_ratio=2.0,  # Ellipse aspect ratio
                binary_mask=binary_mask,
                n_points=self.n_points,
                n_raw_samples=10000
            )
            
            # If ellipse sampling doesn't produce enough points, fall back to rectangle sampling
            if len(point_cloud) < self.n_points:
                point_cloud = generate_rectangle_point_cloud(
                    binary_mask=binary_mask,
                    n_points=self.n_points,
                    over_sample_scale=5
                )
            
            # Ensure we have exactly n_points
            if len(point_cloud) > self.n_points:
                # Randomly sample n_points
                indices = np.random.choice(len(point_cloud), self.n_points, replace=False)
                point_cloud = point_cloud[indices]
            elif len(point_cloud) < self.n_points:
                # Pad with zeros if not enough points
                padding = np.zeros((self.n_points - len(point_cloud), 2))
                point_cloud = np.vstack([point_cloud, padding])
            
            return point_cloud
            
        except Exception as e:
            print(f"Error generating point cloud: {e}")
            return None
    
    def _save_datasets(self, point_clouds, labels, problems, dataset_name):
        """Split and save datasets to train/val/test directories."""
        # Convert to numpy arrays
        point_clouds = np.array(point_clouds, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        
        # Add z-dimension (zeros for 2D)
        point_clouds_3d = np.zeros((point_clouds.shape[0], point_clouds.shape[1], 3), dtype=np.float32)
        point_clouds_3d[:, :, :2] = point_clouds
        
        # Split into train+val and test
        X_temp, X_test, y_temp, y_test, problems_temp, problems_test = train_test_split(
            point_clouds_3d, labels, problems,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels
        )
        
        # Split temp into train and val
        val_ratio = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val, problems_train, problems_val = train_test_split(
            X_temp, y_temp, problems_temp,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=y_temp
        )
        
        print(f"\nDataset statistics for {dataset_name}:")
        print(f"  Total samples: {len(point_clouds)}")
        print(f"  Train samples: {len(X_train)}")
        print(f"  Val samples: {len(X_val)}")
        print(f"  Test samples: {len(X_test)}")
        
        # Save datasets
        self._save_h5_dataset(X_train, y_train, problems_train, 
                             os.path.join(self.train_dir, f"{dataset_name}.h5"), 
                             "train")
        self._save_h5_dataset(X_val, y_val, problems_val,
                             os.path.join(self.val_dir, f"{dataset_name}.h5"),
                             "val")
        self._save_h5_dataset(X_test, y_test, problems_test,
                             os.path.join(self.test_dir, f"{dataset_name}.h5"),
                             "test")
    
    def _save_h5_dataset(self, point_clouds, labels, problems, filepath, split_name):
        """Save dataset in HDF5 format."""
        with h5py.File(filepath, 'w') as f:
            # Save point clouds
            f.create_dataset('data', data=point_clouds)
            
            # Save labels
            f.create_dataset('label', data=labels)
            
            # Save metadata
            f.attrs['num_samples'] = len(point_clouds)
            f.attrs['num_points'] = point_clouds.shape[1]
            f.attrs['split'] = split_name
            f.attrs['dataset_name'] = os.path.basename(filepath).replace('.h5', '')
            
            # Save problem information as JSON strings
            problem_info = []
            for problem in problems:
                problem_dict = {
                    'x_start': problem['x_start'],
                    'x_goal': problem['x_goal'],
                    'search_radius': float(problem.get('search_radius', 0.0)),
                    'best_path_len': float(problem.get('best_path_len', 0.0)) if 'best_path_len' in problem else 0.0,
                    'flank_path_len': float(problem.get('flank_path_len', 0.0)) if 'flank_path_len' in problem else 0.0
                }
                problem_info.append(json.dumps(problem_dict))
            
            # Store as variable length strings
            dt = h5py.special_dtype(vlen=str)
            dset = f.create_dataset('problem_info', (len(problem_info),), dtype=dt)
            for i, info in enumerate(problem_info):
                dset[i] = info
        
        print(f"Saved {split_name} dataset to {filepath}")


def main():
    """Main function to generate all datasets."""
    config = {
        'output_dir': '../NAMR-RRT/data/random_2d',
        'n_points': 1024,  # PointNet++ standard input size
        'test_size': 0.2,  # 20% for testing
        'val_size': 0.1,   # 10% for validation (of remaining 80%)
        'random_state': 42
    }
    
    generator = PointCloudDatasetGenerator(config)
    
    # Generate block and gap datasets
    generator.generate_block_gap_datasets()
    
    # Generate random 2D datasets
    generator.generate_random_2d_datasets()
    
    print("\nDataset generation completed!")
    print(f"Train data saved to: {generator.train_dir}")
    print(f"Val data saved to: {generator.val_dir}")
    print(f"Test data saved to: {generator.test_dir}")


if __name__ == "__main__":
    main()
