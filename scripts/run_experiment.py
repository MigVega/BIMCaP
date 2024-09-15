import argparse
import torch
import os
import open3d as o3d
import numpy as np

import yaml
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.linalg import norm

from facap import feature_errors
from facap.data.scan import Scan
from facap.optimization import Project, Unproject, CameraParameters, FloorTerm, WallTerm, WallSegmentTerm
from facap.utils import dicts_to_torch, visualize_data
from facap.geometry.allign_walls import align_walls
from scipy.spatial.transform import Rotation as R

#todo get floorplan from BIM

def compute_pose_rmse(translations_gt, rotations_gt, translations_est, rotations_est):
    # Compute RMSE for translations
    translation_errors = np.linalg.norm(translations_gt - translations_est, axis=1)
    translation_err = np.mean(translation_errors)

    # Compute RMSE for rotations
    degree_errors = []
    for rotvec_gt, rotvec_est in zip(rotations_gt, rotations_est):
        rotation_gt = R.from_rotvec(rotvec_gt)
        rotation_est = R.from_rotvec(rotvec_est)

        cosine_similarity = np.trace(np.dot(np.transpose(rotation_est.as_matrix()), rotation_gt.as_matrix()))
        angular_distance_radians = np.arccos((cosine_similarity - 1) / 2)

        # Convert angular distance from radians to degrees
        angular_distance_degrees = angular_distance_radians * (180 / np.pi)

        angular_distance_degrees = min(angular_distance_degrees, 180-angular_distance_degrees)

        print("Angular distance between estimated and ground truth rotations:", angular_distance_degrees, "degrees")

        degree_errors.append(angular_distance_degrees)

    rotation_err = np.mean(degree_errors)

    return translation_err, rotation_err


if __name__ == '__main__':
    print('Started experiment')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="YAML configuration file")
    parser.add_argument("--device", default='cuda:0', help="Device to run")
    args = parser.parse_args()
    with open(args.config, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    #Scan should have camera poses
    scan_path = cfg["paths"]["scan_path"]
    scan = Scan(scan_path, scale=cfg["data"]["depths_scale"])# , cut_frames=70)


    save_path = cfg["paths"]["save_path"]
    os.makedirs(save_path, exist_ok=True)

    mesh, pcd = scan.make_mesh_pcd()
    o3d.io.write_triangle_mesh(f"{save_path}/source_mesh.ply", mesh)
    o3d.io.write_point_cloud(f"{save_path}/source_pcd.ply", pcd)

    print('Written Source Mesh and PointCloud')

    #Extract scan data
    data = scan.generate_ba_data(min_frame_difference=cfg["data"]["min_frame_difference"],
                                 max_initial_distance=cfg["data"]["max_initial_distance"],
                                 floor_percentiles=cfg["data"]["floor_percentiles"],
                                 wall_sparsity=cfg["data"]["wall_sparsity"],
                                 floor_sparsity=cfg["data"]["floor_sparsity"],
                                 include_columns=cfg["error"]["include_columns"]
                                 )

    print('Generated data')
    if "wall_term_type" in cfg["error"] and cfg["error"]["wall_term_type"] == "segment":
        floorplan = torch.from_numpy(np.load(f"{scan_path}/floorplan.npy"))
        alligned_walls = align_walls(data[2], floorplan, scale=scan.scale)
        data = (data[0], data[1], alligned_walls, data[3], data[4])

    visualize_data(data, save_path=save_path, scale=scan.scale)
    dicts_to_torch(data, args.device)

    #Extract left?
    left, right, wall, floor, ceiling = data

    #Get camere parameters
    camera_parameters = CameraParameters(scan.cameras).to(args.device).float()
    unproject = Unproject(camera_parameters, scale=scan.scale)
    project = Project(camera_parameters)
    cost_function = nn.MSELoss()

    use_bim = cfg["data"]["use_bim"]

    if cfg["error"]["floor_term"]:
        floor_plane = None
        if use_bim:
            floor_plane = torch.Tensor(np.asarray(o3d.io.read_point_cloud(f"{scan_path}/floorplane.ply").points).astype(np.float32))
        floor_function = FloorTerm(floor, unproject, cost_function, floor_plane)

    if cfg["error"]["ceil_term"]:
        ceil_plane = None
        if use_bim:
            ceil_plane = torch.Tensor(np.asarray(o3d.io.read_point_cloud(f"{scan_path}/ceiling.ply").points).astype(np.float32))
        ceil_function = FloorTerm(ceiling, unproject, cost_function, ceil_plane)

    if cfg["error"]["wall_term"]:
        floorplan = torch.from_numpy(np.load(f"{scan_path}/floorplan.npy"))
        if cfg["error"]["wall_term_type"] == "point":
            wall_function = WallTerm(wall, unproject, cost_function, floorplan).to(args.device).float()
        else:
            wall_function = WallSegmentTerm(wall, unproject, cost_function, floorplan).to(args.device).float()

    params = []

    #Add Camera Parameters as parameters to be optimized
    fixed_cameras = [scan._frames[i] for i in cfg["optimization"]["fixed_cameras_idx"]]
    for name, param in camera_parameters.named_parameters():
        if name.split(".")[-1] not in fixed_cameras:
            params.append(param)

    optimizer = optim.SGD(params, lr=cfg["optimization"]["lr"], momentum=cfg["optimization"]["momentum"])

    torch.save(camera_parameters.state_dict(), f"{save_path}/source_cameras.pth")

    scan_gt = Scan("../camera_refinement_BIM/scan_full/", scale=1)#, cut_frames=70)

    gt_mesh, gt_pcd = scan_gt.make_mesh_pcd()
    o3d.io.write_triangle_mesh(f"{save_path}/gt_mesh.ply", gt_mesh)
    o3d.io.write_point_cloud(f"{save_path}/gt_pcd.ply", gt_pcd)

    print('Written gt mesh and pointcloud')

    camera_parameters_gt = CameraParameters(scan_gt.cameras).to(args.device).float()
    torch.save(camera_parameters_gt.state_dict(), f"{save_path}/gt_cameras.pth")
    rotvec_gt = camera_parameters_gt.state_dict()['rotvecs']
    trans_gt = camera_parameters_gt.state_dict()['translations']

    for epoch in range(cfg["optimization"]["num_epoches"]):


        if epoch % 50 == 0:
            print(f"Epoch {epoch}")
            rotvec_est = camera_parameters.state_dict()['rotvecs']
            trans_est = camera_parameters.state_dict()['translations']
            translation_rmse, rotation_rmse = compute_pose_rmse(trans_gt.cpu().detach().numpy(),
                                                                rotvec_gt.cpu().detach().numpy(),
                                                                trans_est.cpu().detach().numpy(),
                                                                rotvec_est.cpu().detach().numpy())
            print("Translation Err:", translation_rmse)
            print("Rotation Err:", rotation_rmse)


        optimizer.zero_grad()
        error_args = {"unproject": unproject,
                      "project": project,
                      "scale": scan.scale,
                      "distance_function": cost_function,
                      **cfg["error"]}
        ba_function = getattr(feature_errors, cfg["error"]["error_type"])
        ba_term = ba_function(left, right, **error_args)

        floor_term = 0.
        wall_term = 0.
        ceil_term = 0.
        print(f"The value of the loss function on the {epoch}-iteration")
        print(f"\t\t feature-based BA term - {float(ba_term)}")

        if cfg["error"]["floor_term"]:
            floor_term = floor_function() * cfg["error"]["floor_weight"]
            print(f"\t\t floor term - {float(floor_term)}")

        if cfg["error"]["ceil_term"]:

            ceil_term = ceil_function() * cfg["error"]["ceil_weight"]
            print(f"\t\t ceil term - {float(ceil_term)}")

        if cfg["error"]["wall_term"]:
            wall_term = wall_function() * cfg["error"]["wall_weight"]
            print(f"\t\t wall term - {float(wall_term)}")

        loss = ba_term + wall_term + floor_term + ceil_term

        loss.backward()

        for name, param in camera_parameters.named_parameters():
            if torch.isnan(param.grad).any():
                param.grad[torch.isnan(param.grad)] = 0.0
        optimizer.step()


    torch.save(camera_parameters.state_dict(), f"{save_path}/cameras.pth")
    cameras = camera_parameters.get_cameras()
    scan.set_cameras(cameras)

    p_mesh, p_pcd = scan.make_mesh_pcd()
    o3d.io.write_triangle_mesh(f"{save_path}/processed_mesh.ply", p_mesh)
    o3d.io.write_point_cloud(f"{save_path}/processed_pcd.ply", p_pcd)
    print('Written processed mesh and pointcloud')




