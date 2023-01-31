import open3d as o3d
import numpy as np
import pyrealsense2 as rs
import copy
import time
import os
import sys
import cv2
import lib_m
import lib
import open_file

global pcd_combined

pcd_1 = o3d.geometry.PointCloud
pcd_2 = o3d.geometry.PointCloud
pcd_combined = o3d.geometry.PointCloud()

attemps = 10

def add_point_cloud(i, voxel_size, colorized):
    voxel_size = 0.005
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()

    pcd_2 = rs.save_to_ply("prova_iterative/prova_iterative" + str(i) + "_m.ply")
    pcd_2.set_option(rs.save_to_ply.option_ply_binary, False)
    pcd_2.set_option(rs.save_to_ply.option_ply_normals, True)
    pcd_2.process(colorized)


    target_down = o3d.io.read_point_cloud("prova_iterative/prova_iterative_m.ply")
    source_down = o3d.io.read_point_cloud("prova_iterative/prova_iterative" + str(i) + "_m.ply")
    source.points = lib.filtra(source_down)
    target.points = lib.filtra(target_down)

    open_file.open(target)

    source_down, source_fpfh = lib_m.preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = lib_m.preprocess_point_cloud(target, voxel_size)
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    result_ransac = lib_m.execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
    print(result_ransac)
    print(result_ransac.transformation)
    result_icp = lib_m.refine_registration(source, target, source_fpfh, target_fpfh,
                                 voxel_size, result_ransac)
    # Controllo sulla bonta' e reiterazione del metodo
    times = 1
    while(result_icp.fitness <= 0.2 and times < attemps):
        result_ransac = lib_m.execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh, voxel_size)
        #print(result_ransac)
        result_icp = lib_m.refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac)
        #print(result_icp)
        times = times + 1

    lib_m.draw_registration_result(source, target, result_icp.transformation)
    print("Number of attempts: ", times)

    #pcd_combined.points = lib.union(source_down.transform(result_ransac.transformation), target_down)
    #pcd_combined.points = lib_m.union_target(target_down)
    pcd_combined.points = lib_m.union(source.transform(result_ransac.transformation), target)
    print(len(pcd_combined.points))
    o3d.io.write_point_cloud('prova_iterative/prova_iterative_m.ply', pcd_combined, write_ascii=True)

def save_view(pipeline, i):
    colorizer = rs.colorizer()
    frames = pipeline.wait_for_frames()
    colorized = colorizer.process(frames)

    if os.path.exists("prova_iterative/prova_iterative_m.ply"):
        voxel_size = 0.005
        print("Saving the point cloud " + str(i))
        add_point_cloud(i, voxel_size, colorized)
    else:    
        pcd_1 = rs.save_to_ply("prova_iterative/prova_iterative_m.ply")
        print("Saving prova_iterative_m")
        pcd_1.set_option(rs.save_to_ply.option_ply_binary, False)
        pcd_1.set_option(rs.save_to_ply.option_ply_normals, True)
        pcd_1.process(colorized)

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
        while i<100:
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
                save_view(pipeline, i)
                print("\nRepositioning the object..")
                i = i+1
            if key == ord('b'):
                cv2.destroyAllWindows()
                print("\nThis was the final point cloud. Open prova_iterative.ply.")
                pcd_combined = o3d.io.read_point_cloud("prova_iterative/prova_iterative_m.ply")
                o3d.visualization.draw_geometries([pcd_combined], window_name='Finale')
            if key == ord('q'):
                break
    finally:
        pipeline.stop();

if __name__ == '__main__':
    print("+----------------------------------+")
    print("|Press 's' to save point cloud     |")
    print("|Press 'q' to close                |")
    print("|Press 'b' to not take point cloud |")
    print("+----------------------------------+")
    visualize()