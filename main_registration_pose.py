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
global names

colors = [[0, 1, 0], [1, 0, 0], [0.6, 0.2, 0.2], [0.5, 1, 0.2], [0.5, 0.2, 0.4], [0.5, 0.1, 0.9], [0.4, 0, 0.8], [0.4, 0, 0.6],
        [0.3, 0.9, 0.3], [0.4, 0.1, 0.8], [0.7, 0.3, 0], [0.7, 0.4, 1], [0.8, 0.8,0], [0.9, 0.5,0], [1, 0, 0], [0.9, 1,0], [1, 0.5, 1],
        [0.4, 0.4, 1], [0.4, 0.3, 0.8], [0.4, 0.2, 0.2], [0.4, 0.1, 0.1], [0.3, 0.7, 1], [0.3, 0.3, 0.7], [0.3, 0, 0.2], [0.2, 0.1, 0.7], 
        [0.1, 1, 0.7], [0.2, 0.3, 0], [0.2, 0, 0.7], [0.2, 0.8, 0.1], [0.3, 0, 0.4], [0.3, 0, 0]]
db = []
names = []

def caricaGraspingPose(filename):
    file1 = open(filename, "r")
    lines = file1.readlines() 
    for i in range(len(lines)):
        line = lines[i]
        line = line.rstrip("\n")
        array = line.split(" ")
        db.append(array)
        names.append(array[0])
    print("Db caricato")

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
            points_list_filtered.append(p)
    ply_filtrata.clear()
    ply_filtrata.points = o3d.utility.Vector3dVector(points_list_filtered)
    return ply_filtrata.points

def load_point_clouds(path_cad, voxel_size):
    pcds = []
    pcd = o3d.geometry.PointCloud()

    pcd_ = o3d.io.read_point_cloud(path_cad)
    #pcd_ = o3d.io.read_point_cloud("/home/marika/Scrivania/Algoritmi/second_step/rotation/FINAL_OBJ_BLACK/real_time/From_Camera.ply")
    pcd.points = filtra2(pcd_)
    #o3d.visualization.draw_geometries([pcd])
    pcd_down1 = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_down1.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    pcds.append(pcd_down1)


    #pcd_ = o3d.io.read_point_cloud(path_cad)
    #pcd_ = pcd_.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    pcd_ = o3d.io.read_point_cloud("/home/marika/Scrivania/Algoritmi/second_step/rotation/FINAL_OBJ_B/real_time/From_Camera.ply")
    pcd.points = filtra2(pcd_)
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
    return transformation_icp, information_icp, icp_fine
    
def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        #print(source_id)
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp, result_icp = pairwise_registration(
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
    return pose_graph, result_icp
    
def main_method_multiway(path_cad):
    voxel_size = 0.003 #0.01
    t = time.time()
    pcds_down = load_point_clouds(path_cad, voxel_size)

    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5

    pose_graph, result_icp = full_registration(pcds_down,
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
    #frame_s = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=np.array([0., 0., 0.]))
    #o3d.visualization.draw_geometries([pcds_down[0], pcds_down[1], frame_s])   
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds_down)):
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
        print(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].paint_uniform_color(colors[point_id])
        #file_name = "/home/marika/Scrivania/Algoritmi/second_step/rotation/FINAL_OBJ_B/real_time/Pc" + str(point_id) + ".ply"
        #o3d.io.write_point_cloud(file_name, pcds_down[point_id], write_ascii=True)
        pcd_combined += pcds_down[point_id]
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)

    elapsed = time.time() - t
    print("MON - tempo per caricamento + ricerca delle trasformazioni + generazione della vasca totale + salvataggio")
    print(elapsed)
    frame_s = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=np.array([0., 0., 0.]))
    
    o3d.visualization.draw_geometries([pcd_combined])

    #print("Resul_icp_fitness: " + str(result_icp.fitness))
    return result_icp.fitness, result_icp.transformation
    

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

    pcd_1 = rs.save_to_ply("/home/marika/Scrivania/Algoritmi/second_step/rotation/FINAL_OBJ_B/real_time/From_Camera.ply")
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
    print("|Press s' to save point cloud      |")
    print("|Press 'q' to close                |")
    print("+----------------------------------+")

    caricaGraspingPose("/home/marika/Scrivania/Algoritmi/second_step/rotation/FINAL_OBJ_B/File_matrix.txt")

    visualize()

    fits = []
    transformations = []

    #print(len(names))

    for v in range(len(names)):
        print("Comparison ... {} %".format((v+1)*9.1))
        [fit, transformation] = main_method_multiway(names[v])
        #o3d.visualization.draw_geometries([fit, transformation])
        fits.append(fit)
        print("Iterazione: " + str(v) + " Fitness " + str(fit))
        transformations.append(transformation)
        #if (fit >= 0.9):
            #break
    
    best_index = fits.index(max(fits)) #0-based
    print("Best index " + str(best_index))
    point_ = db[best_index][1:4]
    point = [float(i) for i in point_]
    vx_ = db[best_index][4:7]
    vx = [float(i) for i in vx_]
    vy_ = db[best_index][7:10]
    vy = [float(i) for i in vy_]
    vz_ = db[best_index][10:]
    vz = [float(i) for i in vz_]

    Ao = np.eye(4)
    Ao[:3,0] = vx
    Ao[:3,1] = vy
    Ao[:3,2] = vz
    Ao[:3,3] = point
    oAcad = transformations[best_index]

    target_down = o3d.geometry.PointCloud()
    target = o3d.io.read_point_cloud("/home/marika/Scrivania/Algoritmi/second_step/rotation/FINAL_OBJ_B/real_time/From_Camera.ply")
    target_down.points = filtra2(target)
    cad = o3d.io.read_point_cloud(names[best_index])
    target_down.transform(oAcad)
    o3d.visualization.draw_geometries([cad, target_down])

    print("Compute pose in world frame")
    '''
    wMe = np.array([[-0.0718, 0.997384, -0.00677, 0.0055],
                    [0.9969, 0.0720, 0.0291, 0.2892],
                    [0.0295, -0.0046, -0.9995, 0.2026],
                    [0, 0, 0, 1]])
                    '''
    wMe = np.array([[-0.00856505, 0.999954, -3.42984e-05, -0.00374687],
                    [0.999525, 0.00856038, -0.0292805, 0.423501],
                    [-0.0292789, -0.000285071, -0.999571, 0.133742],
                    [0, 0, 0, 1]])
    #camera interna endefector
    eMc = np.array([[0.01999945272, -0.9990596861, 0.03846772089, 0.05834485203],
                    [0.9997803621, 0.01974315714, -0.007031030191, -0.03476564525+0.02],
                    [0.006264944557, 0.03859988867, 0.999235107, -0.06760482074],
                    [0, 0, 0, 1]])

    #A_to_rob = wMe.dot(eMc.dot(oAcad.dot(Ao)))
    cA_1 = np.asarray([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])

    cA_2 = np.asarray([[0, 1, 0, 0],
                        [-1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    #A_to_rob = wMe.dot(eMc.dot(cA_1.dot(oAcad.dot(Ao))))
    #A_to_rob = wMe.dot(eMc.dot(oAcad.dot(Ao)))

    #A_to_rob = np.dot(wMe, (np.dot(eMc, (np.dot(cA_1, (np.dot(oAcad, Ao)))))))

    A_to_rob = np.dot(wMe, np.dot(np.dot(eMc, (np.dot(cA_1, (np.dot(oAcad, Ao))))), cA_2)) 


    print(np.dot(oAcad, Ao))
    print(np.dot(cA_1, (np.dot(oAcad, Ao))))
    print(np.dot(eMc, (np.dot(cA_1, (np.dot(oAcad, Ao))))))

    print("-------------------------------------------------")
    print(A_to_rob[0,3], A_to_rob[1,3], A_to_rob[2,3])
    print("Rd")
    print(A_to_rob[0,0], A_to_rob[0,1], A_to_rob[0,2])
    print(A_to_rob[1,0], A_to_rob[1,1], A_to_rob[1,2])
    print(A_to_rob[2,0], A_to_rob[2,1], A_to_rob[2,2])


    
