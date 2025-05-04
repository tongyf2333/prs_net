import open3d as o3d
import numpy as np
import trimesh
import os
import glob
from tqdm import tqdm
from multiprocessing import Pool
import functools
import argparse

def find_nearest_points(points, mesh):
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    return scene.compute_closest_points(points)['points']

def merge_mesh(path, rotate=True):
    scene = trimesh.load(path, force='mesh')
    if isinstance(scene, trimesh.Trimesh):
        combined_mesh = scene
    else:
        combined_mesh = trimesh.util.concatenate(list(scene.geometry.values()))
    if rotate:
        random_rotation = trimesh.transformations.random_rotation_matrix()
        combined_mesh.apply_transform(random_rotation)
    return combined_mesh

def trimesh_to_open3d(mesh):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    return o3d_mesh

def mesh_to_voxel(mesh, resolution=32):
    mesh.compute_vertex_normals()
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    scale = 1.0 / max(extent)
    mesh.translate(-center)
    mesh.scale(scale, center=(0, 0, 0))
    voxel_size = 1.0 / resolution
    voxel_array = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)

    for voxel in voxel_grid.get_voxels():
        idx = voxel.grid_index
        if all(0 <= v < resolution for v in idx):
            voxel_array[tuple(idx)] = 1
            
    return voxel_array, mesh

def gen_voxelcenter():
    i, j, k = np.meshgrid(np.arange(32), np.arange(32), np.arange(32), indexing='ij')
    arr = np.stack((i, j, k), axis=-1).astype(np.float32)
    return arr / 32 + 1 / 64 - 0.5

# 每个进程处理一个文件路径
def process_one_mesh(path, voxel_center,out_dir,rotate=True):
    try:
        mesh = merge_mesh(path,rotate)
        mesh = trimesh_to_open3d(mesh)
        voxel_array, mesh = mesh_to_voxel(mesh, resolution=32)
        nearest = find_nearest_points(voxel_center.reshape(-1, 3), mesh).numpy().reshape(32, 32, 32, 3)
        sampled_points = mesh.sample_points_uniformly(number_of_points=1000)
        sampled_points = np.asarray(sampled_points.points)
        if rotate==False:
            o3d.io.write_triangle_mesh(os.path.join(out_dir,"model_normalized.obj"), mesh)
        return voxel_array, nearest, sampled_points
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None  # 标记错误项

def preprocess(data_dir, output_dir,rotate=True):
    data_paths = glob.glob(os.path.join(data_dir, '**', '*.obj'), recursive=True)
    data_len = len(data_paths)
    print("Number of meshes:", data_len)

    voxel_center = gen_voxelcenter()

    with Pool(processes=8) as pool:
        results = list(tqdm(pool.imap(functools.partial(process_one_mesh, voxel_center=voxel_center,out_dir=output_dir,rotate=rotate), data_paths), total=data_len))

    results = [r for r in results if r is not None]
    data_len = len(results)
    voxels = np.zeros((data_len, 32, 32, 32), dtype=np.float32)
    nearests = np.zeros((data_len, 32, 32, 32, 3), dtype=np.float32)
    sampled = np.zeros((data_len, 1000, 3), dtype=np.float32)

    for i, (v, n, s) in enumerate(results):
        voxels[i] = v
        nearests[i] = n
        sampled[i] = s

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'voxels.npy'), voxels)
    np.save(os.path.join(output_dir, 'nearests.npy'), nearests)
    np.save(os.path.join(output_dir, 'sampled.npy'), sampled)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser=argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='meshes/chair', help='path to data')
parser.add_argument('--output_dir', type=str, default='preprocessed/chair', help='path to save preprocessed data')
parser.add_argument('--rotate', type=str2bool, default=True, help='whether to rotate the mesh')
args=parser.parse_args()
preprocess(args.data_dir, args.output_dir, args.rotate)