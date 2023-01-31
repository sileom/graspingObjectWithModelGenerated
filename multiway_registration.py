import open3d as o3d
import cv2
import numpy as np
import os
import sys
import glob
import time
import lib
import pyrealsense2 as rs
import open_file
import lib_m

pcd_1 = o3d.geometry.PointCloud
pcd_2 = o3d.geometry.PointCloud

voxel_size = 0.003 #0.01
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5

attemps = 10

def modify(depth_image, color_image):
    new_image = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), np.uint8)
    for i in range(depth_image.shape[0]):
        for j in range(depth_image.shape[1]):
            if (depth_image[i, j] != 0):
                new_image[i, j, :] = color_image[i, j, :]
    return new_image

def load_point_clouds(voxel_size=0.0):
    pcds_list = glob.glob("prova_iterative/prova_iterative*_m.ply")
    print(pcds_list)
    pcds = []
    pcd = o3d.geometry.PointCloud()
    #i=1
    for i in range(len(pcds_list)):
        pcd_ = o3d.io.read_point_cloud("prova_iterative/prova_iterative%d_m.ply" % i)
        pcd.points = lib.filtra(pcd_)
        #o3d.visualization.draw_geometries([pcd])
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        pcds.append(pcd_down)
    return pcds

def pairwise_registration(source, target):
    #print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
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
                pcds[source_id], pcds[target_id])
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
    
def add_point_cloud(i, voxel_size, colorized):
    voxel_size = 0.003
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()

    pcd_2 = rs.save_to_ply("prova_iterative/prova_iterative" + str(i) + "_m.ply")
    pcd_2.set_option(rs.save_to_ply.option_ply_binary, False)
    pcd_2.set_option(rs.save_to_ply.option_ply_normals, True)
    pcd_2.process(colorized)


    target_down = o3d.io.read_point_cloud("prova_iterative/prova_iterative0_m.ply")
    source_down = o3d.io.read_point_cloud("prova_iterative/prova_iterative" + str(i) + "_m.ply")
    source.points = lib.filtra(source_down)
    target.points = lib.filtra(target_down)

    source_down, source_fpfh = lib_m.preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = lib_m.preprocess_point_cloud(target, voxel_size)
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))


def save_view(pipeline, i):
    colorizer = rs.colorizer()
    frames = pipeline.wait_for_frames()
    colorized = colorizer.process(frames)

    if os.path.exists("prova_iterative/prova_iterative0_m.ply"):
        voxel_size = 0.005
        print("Saving the point cloud " + str(i))
        add_point_cloud(i, voxel_size, colorized)
    else:    
        pcd_1 = rs.save_to_ply("prova_iterative/prova_iterative0_m.ply")
        print("Saving prova_iterative0_m")
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

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    align_to = rs.stream.color
    align = rs.align(align_to)


    try:
        i=0
        while i<100:
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
                save_view(pipeline, i)
                print("\nRepositioning the object..")
                i = i+1
            if key == ord('b'):
                cv2.destroyAllWindows()
                print("\nThis is the final point cloud.")

                t = time.time()
                pcds_down = load_point_clouds(voxel_size)

                pose_graph = full_registration(pcds_down,
                                                   max_correspondence_distance_coarse,
                                                   max_correspondence_distance_fine)
                                                   
                print("Optimizing PoseGraph ...")
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
                    pcd_combined += pcds_down[point_id]
                pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)

                o3d.io.write_point_cloud("prova_iterative/Modello_Finale.ply", pcd_combined_down)
                o3d.visualization.draw_geometries([pcd_combined_down])
                break
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
