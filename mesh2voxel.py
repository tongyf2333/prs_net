import numpy as np
import trimesh

def load_mesh(file_path):
    mesh = trimesh.load(file_path)
    return mesh

def create_voxel_grid(mesh, grid_size=32):
    min_bound, max_bound = mesh.bounds
    extent = max_bound - min_bound
    max_extent = np.max(extent)

    pitch = max_extent / grid_size

    mesh_copy = mesh.copy()
    mesh_copy.apply_translation(-min_bound)
    mesh_copy.apply_scale(1.0 / pitch)

    voxel = mesh_copy.voxelized(pitch=1.0)
    voxel_matrix = voxel.matrix.astype(bool)
    sx, sy, sz = voxel_matrix.shape

    padded = np.zeros((grid_size, grid_size, grid_size), dtype=bool)

    ox = max((grid_size - sx) // 2, 0)
    oy = max((grid_size - sy) // 2, 0)
    oz = max((grid_size - sz) // 2, 0)

    ex = min(sx, grid_size - ox)
    ey = min(sy, grid_size - oy)
    ez = min(sz, grid_size - oz)

    padded[ox:ox+ex, oy:oy+ey, oz:oz+ez] = voxel_matrix[:ex, :ey, :ez]

    return padded, pitch, min_bound


if __name__ == "__main__":
    mesh = load_mesh('meshes/test_table1/table1.obj')
    voxel_grid, voxel_size, min_bound = create_voxel_grid(mesh, grid_size=32)
    print("Voxel grid shape:", voxel_grid.shape)
    print("Voxel size:", voxel_size)
    print("Min bound:", min_bound)