import open3d as o3d
import cv2
import numpy as np
import os
import sys
import glob
import time
import pyrealsense2 as rs
import copy

global result

colors = [[0, 1, 0], [1, 0, 0], [0.6, 0.2, 0.2], [0.5, 1, 0.2], [0.5, 0.2, 0.4], [0.5, 0.1, 0.9], [0.4, 0, 0.8], [0.4, 0, 0.6],
        [0.3, 0.9, 0.3], [0.4, 0.1, 0.8], [0.7, 0.3, 0], [0.7, 0.4, 1], [0.8, 0.8,0], [0.9, 0.5,0], [1, 0, 0], [0.9, 1,0], [1, 0.5, 1],
        [0.4, 0.4, 1], [0.4, 0.3, 0.8], [0.4, 0.2, 0.2], [0.4, 0.1, 0.1], [0.3, 0.7, 1], [0.3, 0.3, 0.7], [0.3, 0, 0.2], [0.2, 0.1, 0.7], 
        [0.1, 1, 0.7], [0.2, 0.3, 0], [0.2, 0, 0.7], [0.2, 0.8, 0.1], [0.3, 0, 0.4], [0.3, 0, 0]]

def filtra2(source_down):
    ply_filtrata = o3d.geometry.PointCloud()
    min_bound = [-0.29, -0.29, -0.39] #-0.40 #0  #-0.10, -0.10, 0.0 #-10.10, -10.10, -0.40 #[-0.09, -0.05, -0.30] #[-10.09, -10.05, -10.50]
    max_bound = [0.29, 0.29, 0.39] #0 #0.40    #0.10, 0.10, 0.30       #10.10, 10.10, 05      #[0.15, 0.13, 0]    #[10.20, 10.15, 10.50]
    points_list_filtered = []
    print(len(source_down.points))
    for i in range(len(source_down.points)):
        p = source_down.points[i]
        if((min_bound[0] <= p[0] and p[0] <= max_bound[0]) and (min_bound[1] <= p[1] and p[1] <= max_bound[1])
            and (min_bound[2] < p[2] and p[2] <= max_bound[2])):
            points_list_filtered.append(p)
    ply_filtrata.clear()
    ply_filtrata.points = o3d.utility.Vector3dVector(points_list_filtered)
    return ply_filtrata.points

def filtra(source_down):
    ply_filtrata = o3d.geometry.PointCloud()
    min_bound = [-10.09, -10.05, -10.50] #-0.40 #0  #-0.10, -0.10, 0.0 #-10.10, -10.10, -0.40 #[-0.09, -0.05, -0.30] #[-10.09, -10.05, -10.50]
    max_bound = [10.20, 10.15, 10.50] #0 #0.40    #0.10, 0.10, 0.30       #10.10, 10.10, 05      #[0.15, 0.13, 0]    #[10.20, 10.15, 10.50]
    points_list_filtered = []
    print(len(source_down.points))
    for i in range(len(source_down.points)):
        p = source_down.points[i]
        if((min_bound[0] <= p[0] and p[0] <= max_bound[0]) and (min_bound[1] <= p[1] and p[1] <= max_bound[1])
            and (min_bound[2] < p[2] and p[2] <= max_bound[2])):
            #print(p)
            points_list_filtered.append(p)
    ply_filtrata.clear()
    ply_filtrata.points = o3d.utility.Vector3dVector(points_list_filtered)
    return ply_filtrata.points

def load_point_clouds(path_cad, voxel_size):
    pcds = []
    pcd = o3d.geometry.PointCloud()

    pcd_ = o3d.io.read_point_cloud("real_time/From_Camera.ply")
    pcd.points = filtra2(pcd_)
    o3d.visualization.draw_geometries([pcd])
    pcd_down1 = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_down1.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    pcds.append(pcd_down1)


    pcd_ = o3d.io.read_point_cloud(path_cad)
    #pcd_ = pcd_.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    pcd.points = filtra(pcd_)
    pcd_down2 = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_down2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    pcds.append(pcd_down2)
    return pcds
    
def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    #print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    result = icp_coarse
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp
    
def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        #print(source_id)
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], max_correspondence_distance_coarse, max_correspondence_distance_fine)
            #print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph
    
def main_method_multiway(path_cad):
    voxel_size = 0.003 #0.01
    t = time.time()
    pcds_down = load_point_clouds(path_cad, voxel_size)

    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5

    pose_graph = full_registration(pcds_down,
                                    max_correspondence_distance_coarse,
                                    max_correspondence_distance_fine)
                                    
    #print("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)

    o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)
            
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds_down)):
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
        print(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].paint_uniform_color(colors[point_id])
        pcd_combined += pcds_down[point_id]
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    #o3d.io.write_point_cloud("/mnt/f1e018d6-98bf-4b00-91de-f7b3f30ea52a/TESI/ColangeloMarika/ObjBlackCircle1.ply", pcd_combined_down)

    elapsed = time.time() - t
    print("MON - tempo per caricamento + ricerca delle trasformazioni + generazione della vasca totale + salvataggio")
    print(elapsed)
    frame_s = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=np.array([0., 0., 0.]))
    o3d.visualization.draw_geometries([pcd_combined, frame_s])

def modify(depth_image, color_image):
    new_image = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), np.uint8)
    for i in range(depth_image.shape[0]):
        for j in range(depth_image.shape[1]):
            if (depth_image[i, j] != 0):
                new_image[i, j, :] = color_image[i, j, :]
                #print(type(new_image))
    return new_image

def save_view(pipeline):
    colorizer = rs.colorizer()
    frames = pipeline.wait_for_frames()
    colorized = colorizer.process(frames)

    pcd_1 = rs.save_to_ply("real_time/From_Camera.ply")
    print("Saving point cloud")
    pcd_1.set_option(rs.save_to_ply.option_ply_binary, False)
    pcd_1.set_option(rs.save_to_ply.option_ply_normals, True)
    pcd_1.process(colorized)

def visualize():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    colorizer = rs.colorizer()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    align_to = rs.stream.color
    align = rs.align(align_to)


    try:
        i=0
        while i<1:
            frames = pipeline.wait_for_frames()

            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue
            
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            color_image_modify = modify(depth_image, color_image)

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            images = np.hstack([color_image_modify])

            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.destroyAllWindows()
                save_view(pipeline)
                i = i+1
            if key == ord('q'):
                break
    finally:
        pipeline.stop();

if __name__ == '__main__':
    print("+----------------------------------+")
    print("|Press s' to save point cloud     |")
    print("|Press 'q' to close                |")
    print("+----------------------------------+")
    visualize()
    names = ["45gradi/Modello_Ruotato0.ply",
            "45gradi/Modello_Ruotato1.ply",
            "45gradi//Modello_Ruotato2.ply",
            "45gradi//Modello_Ruotato3.ply",
            "45gradi//Modello_Ruotato4.ply",
            "45gradi//Modello_Ruotato5.ply",
            "45gradi//Modello_Ruotato6.ply"]

    fits = []

    for v in range(6):
        #fit = main_method_multiway(names[v])
        #fits.append(fit)
        main_method_multiway(names[v])
        
