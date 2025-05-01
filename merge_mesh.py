import trimesh
import numpy as np

def merge_mesh(path, rotate=True):
    scene = trimesh.load(path)
    if isinstance(scene, trimesh.Trimesh):
        combined_mesh = scene
    else:
        combined_mesh = trimesh.util.concatenate(list(scene.geometry.values()))
    if rotate:
        random_rotation = trimesh.transformations.random_rotation_matrix()
        combined_mesh.apply_transform(random_rotation)
    return combined_mesh

if __name__ == "__main__":
    path = 'chair5.obj'
    combined_mesh = merge_mesh(path)