import open3d as o3d
import numpy as np
import copy
import sys

def union_target(target_down):
    print("Make a combined point cloud")
    #pcds = [target_down]
    pcd_combined = o3d.geometry.PointCloud()
    list_filtered = []
    for i in range (len(target_down.points)):
        p = target_down.points[i]
        list_filtered.append(p)
    pcd_combined.points = o3d.utility.Vector3dVector(list_filtered)
    return pcd_combined.points

def union(source_down, target_down):
    print("Make a combined point cloud")
    pcds = [source_down, target_down]
    pcd_combined = o3d.geometry.PointCloud()
    list_filtered = []
    for i in range (len(target_down.points)):
        p = target_down.points[i]
        list_filtered.append(p)
    for j in range (len(source_down.points)):
        q = source_down.points[j]
        list_filtered.append(q)
    pcd_combined.points = o3d.utility.Vector3dVector(list_filtered)
    return pcd_combined.points

def filter_pointcloud_on_z(source, voxel_size):
    z_min = 0.02
    z_max = 0.80
    result = []
    for i in range(len(source.points)):
        _p = source.points[i]
        if (z_min <= abs(_p[2]) and abs(_p[2]) <= z_max):
            result.append(_p)
    
    result_pc = o3d.geometry.PointCloud()
    result_pc.points = o3d.utility.Vector3dVector(result)
    result_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    return result_pc

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    #source_temp.paint_uniform_color([1, 0.706, 0])
    #target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name='due separate')

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size):
    target = o3d.io.read_point_cloud("tf.ply")
    #source =  o3d.io.read_point_cloud("resources/input/CR.ply")
    source =  o3d.io.read_point_cloud("Coperchio_olioc3comp.ply")
    #source_ =  o3d.io.read_point_cloud("resources/input/CAD.ply")
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    #target = o3d.io.read_point_cloud("resources/input/cad.ply")
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size*15)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    #print(":: RANSAC registration on downsampled point clouds.")
    #print("   Since the downsampling voxel size is %.3f," % voxel_size)
    #print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    #print(":: Point-to-plane ICP registration is applied on original point")
    #print("   clouds to refine the alignment. This time we use a strict")
    #print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result