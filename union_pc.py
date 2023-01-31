import open3d as o3d
import numpy as np
import pyrealsense2 as rs
import copy
import time
import os
import sys
import cv2

def save(pipeline, i):
    colorizer = rs.colorizer()
    frames = pipeline.wait_for_frames()
    colorized = colorizer.process(frames)

    ply = rs.save_to_ply("prova_u" + str(i) + ".ply")

    ply.set_option(rs.save_to_ply.option_ply_binary, False)
    ply.set_option(rs.save_to_ply.option_ply_normals, True)

    print("Saving prova_u" + str(i) + ".ply")

    ply.process(colorized)
    print("Done")


def visualize():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        i=0
        while i<2:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))

            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.destroyAllWindows()
                save(pipeline, i)
                print("\nRepositioning the object..")
                i = i+1
            if key == ord('q'):
                break
    finally:
        pipeline.stop();

def filtra(source_down):
    ply_filtrata = o3d.geometry.PointCloud()
    min_bound = [-10, -10, -0.40]
    max_bound = [10, 10, 0]
    points_list_filtered = []
    for i in range(len(source_down.points)):
        p = source_down.points[i]
        if((min_bound[0] <= p[0] and p[0] <= max_bound[0]) and (min_bound[1] <= p[1] and p[1] <= max_bound[1])
            and (min_bound[2] <= p[2] <= max_bound[2])):
            points_list_filtered.append(p)
    ply_filtrata.clear()
    ply_filtrata.points = o3d.utility.Vector3dVector(points_list_filtered)
    return ply_filtrata

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

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def prepare_dataset(voxel_size):
    print("\nLoad two point clouds and disturb initial pose.")
    source_down = o3d.io.read_point_cloud("prova_u0.ply")
    #o3d.visualization.draw_geometries([source_down])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    target_down = o3d.io.read_point_cloud("prova_u1.ply")
    #o3d.visualization.draw_geometries([target_down])
    source = filtra(source_down)
    target = filtra(target_down)
    o3d.io.write_point_cloud("sf.ply", source, write_ascii=True)
    o3d.io.write_point_cloud("tf.ply", target, write_ascii=True)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                       zoom=0.00001, front=[0, 0, 1],
                                       lookat=[0, 0, 1],
                                       up=[0, 1, 0])

def union(source_down, target_down):
    print("Make a combined point cloud")
    pcds = [source_down, target_down]
    pcd_combined = o3d.geometry.PointCloud()
    for i in range (len(pcds)):
        pcd_combined += pcds[i]
        #pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size = 0.005)
    o3d.io.write_point_cloud('test_combined.ply', pcd_combined, write_ascii=True)
    o3d.visualization.draw_geometries([pcd_combined])

if __name__ == '__main__':
    print("+----------------------------------+")
    print("|Press 's' to save point cloud     |")
    print("|Press 'q' to close                |")
    print("+----------------------------------+")
    visualize()
    voxel_size = 0.005 
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)
    source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
    print(result_ransac)
    draw_registration_result(source_down, target_down, result_ransac.transformation)
    union(source_down.transform(result_ransac.transformation), target_down)
