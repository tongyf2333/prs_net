from mesh2voxel import create_voxel_grid
from merge_mesh import merge_mesh
import torch
import numpy as np
from pathlib import Path
import trimesh
from torch.utils.data import Dataset, DataLoader
import os
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_nearest_points_to_voxel_centers(points, min_bound, voxel_size, resolution):
    device = points.device
    lin_coords = torch.arange(resolution, device=device) + 0.5
    grid_x, grid_y, grid_z = torch.meshgrid(lin_coords, lin_coords, lin_coords, indexing="ij")
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # [R, R, R, 3]
    centers = grid * voxel_size + min_bound  # [R, R, R, 3]

    centers_flat = centers.view(-1, 3)  # [V, 3]
    points = points.unsqueeze(0)        # [1, N, 3]
    centers_flat = centers_flat.unsqueeze(1)  # [V, 1, 3]

    dists = torch.norm(centers_flat - points, dim=2)  # [V, N]
    nearest_idx = torch.argmin(dists, dim=1)          # [V]
    nearest_points = points[0, nearest_idx]           # [V, 3]

    nearest_points_grid = nearest_points.view(resolution, resolution, resolution, 3)  # [R, R, R, 3]
    return nearest_points_grid

class MyDataset(Dataset):
    def __init__(self, data_dir, rotate=True, point_num=1000, num_workers=8):
        self.rotate = rotate
        self.point_num = point_num
        self.data_paths = glob.glob(os.path.join(data_dir, '**', '*.obj'), recursive=True)
        self.data_size = len(self.data_paths)

        self.points = []
        self.voxel_grids = []
        self.voxel_sizes = []
        self.min_bounds = []
        self.near_points = []
        
        futures = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for idx, path in enumerate(self.data_paths):
                futures.append(executor.submit(self.process_single_mesh, path, idx))

            for f in tqdm(as_completed(futures), total=len(futures), desc="Loading meshes (parallel)"):
                idx, point, voxel_grid, voxel_size, min_bound,near_point = f.result()
                self.points.append(point)
                self.voxel_grids.append(voxel_grid)
                self.voxel_sizes.append(voxel_size)
                self.min_bounds.append(min_bound)
                self.near_points.append(near_point)


    def process_single_mesh(self, path, idx):
        combined_mesh = merge_mesh(path, self.rotate)
        voxel_grid, voxel_size, min_bound = create_voxel_grid(combined_mesh, grid_size=32)
        points, _ = trimesh.sample.sample_surface(combined_mesh, self.point_num)

        point = torch.tensor(points, dtype=torch.float32)
        voxel_grid = torch.tensor(voxel_grid, dtype=torch.float32)
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        min_bound = torch.tensor(min_bound, dtype=torch.float32)
        near_point=get_nearest_points_to_voxel_centers(point, min_bound, voxel_size, 32)
        
        return idx, point, voxel_grid, voxel_size, min_bound, near_point

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return self.points[idx], self.voxel_grids[idx], self.voxel_sizes[idx], self.min_bounds[idx], self.near_points[idx]

if __name__ == "__main__":
    dataset = MyDataset('meshes')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for points, voxel_grid, voxel_size, min_bound,near_point in dataloader:
        print("Points shape:", points.shape)
        print("Voxel grid shape:", voxel_grid.shape)
        print("Voxel size:", voxel_size)
        print("Min bound:", min_bound)
        print("Near points shape:", near_point.shape)