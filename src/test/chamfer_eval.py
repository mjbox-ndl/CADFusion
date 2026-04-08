"""
Chamfer Distance evaluation using pure PyTorch (no external chamfer_distance package).
Computes per-sample CD between generated and reference point clouds.

Usage:
    python src/test/chamfer_eval.py \
        --fake exp/visual_objects/first_run \
        --real exp/visual_objects/first_run_ref
"""

import torch
import argparse
import os
import numpy as np
from tqdm import tqdm
import random
from plyfile import PlyData
from pathlib import Path

random.seed(0)
N_POINTS = 2000


def find_files(folder, extension):
    return sorted([Path(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith(extension)])


def read_ply(path):
    with open(path, 'rb') as f:
        plydata = PlyData.read(f)
        x = np.array(plydata['vertex']['x'])
        y = np.array(plydata['vertex']['y'])
        z = np.array(plydata['vertex']['z'])
        vertex = np.stack([x, y, z], axis=1)
    return vertex


def distChamfer(a, b):
    """Pure PyTorch chamfer distance. a, b: (bs, N, 3)"""
    x, y = a, b
    bs, num_points_x, _ = x.size()
    _, num_points_y, _ = y.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind_x = torch.arange(0, num_points_x, device=a.device).long()
    diag_ind_y = torch.arange(0, num_points_y, device=a.device).long()
    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(2).expand(bs, num_points_x, num_points_y)
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand(bs, num_points_x, num_points_y)
    P = rx + ry - 2 * zz  # (bs, num_points_x, num_points_y)
    dl = P.min(2)[0]  # (bs, num_points_x) - for each x point, min dist to y
    dr = P.min(1)[0]  # (bs, num_points_y) - for each y point, min dist to x
    return dl, dr


def downsample_pc(points, n):
    sample_idx = random.sample(list(range(points.shape[0])), n)
    return points[sample_idx]


def normalize_pc(points):
    scale = np.max(np.abs(points))
    points = points / scale
    return points


def collect_pc(cad_folder):
    pc_path = find_files(os.path.join(cad_folder, 'ptl'), 'final_pcd.ply')
    if len(pc_path) == 0:
        return None
    pc_path = pc_path[-1]  # last (final) pcd
    pc = read_ply(pc_path)
    if pc.shape[0] > N_POINTS:
        pc = downsample_pc(pc, N_POINTS)
    pc = normalize_pc(pc)
    return pc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", type=str, required=True, help="Path to generated visual objects")
    parser.add_argument("--real", type=str, required=True, help="Path to reference visual objects")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    fake_names = sorted(os.listdir(args.fake))
    real_names = set(os.listdir(args.real))

    cd_list = []
    skipped = 0

    for name in tqdm(fake_names, desc="Computing Chamfer Distance"):
        if name not in real_names:
            skipped += 1
            continue

        fake_folder = os.path.join(args.fake, name)
        real_folder = os.path.join(args.real, name)

        fake_pc = collect_pc(fake_folder)
        real_pc = collect_pc(real_folder)

        if fake_pc is None or real_pc is None:
            skipped += 1
            continue

        # Ensure same number of points for batched computation
        n_pts = min(fake_pc.shape[0], real_pc.shape[0])
        if fake_pc.shape[0] > n_pts:
            fake_pc = downsample_pc(fake_pc, n_pts)
        if real_pc.shape[0] > n_pts:
            real_pc = downsample_pc(real_pc, n_pts)

        fake_t = torch.tensor(fake_pc, dtype=torch.float32).unsqueeze(0).to(args.device)
        real_t = torch.tensor(real_pc, dtype=torch.float32).unsqueeze(0).to(args.device)

        with torch.no_grad():
            dl, dr = distChamfer(fake_t, real_t)
            cd = (dl.mean(dim=1) + dr.mean(dim=1)).item()

        cd_list.append(cd)

    cd_arr = np.array(cd_list)
    print(f"\n{'='*50}")
    print(f"Chamfer Distance Evaluation")
    print(f"{'='*50}")
    print(f"Valid pairs:  {len(cd_list)}")
    print(f"Skipped:      {skipped}")
    print(f"Mean CD:      {cd_arr.mean():.6f}")
    print(f"Median CD:    {np.median(cd_arr):.6f}")
    print(f"Std CD:       {cd_arr.std():.6f}")
    print(f"Min CD:       {cd_arr.min():.6f}")
    print(f"Max CD:       {cd_arr.max():.6f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
