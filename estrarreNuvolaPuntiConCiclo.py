#!/usr/bin/python3

import open3d as o3d
import numpy as np
import copy
import cv2


def filtra(source_raw):
    ply_filtrata = o3d.geometry.PointCloud()
    min_bound = [-10, -10, -10]
    max_bound = [10, 10, 10]
    points_list_filtered = []
    for i in range(len(source_raw.points)):
        p = source_raw.points[i]
        if((min_bound[0] <= p[0] and p[0] <= max_bound[0]) and (min_bound[1] <= p[1] and p[1] <= max_bound[1])
            and (min_bound[2] <= p[2] <= max_bound[2])):
            points_list_filtered.append(p)
    ply_filtrata.clear()
    ply_filtrata.points = o3d.utility.Vector3dVector(points_list_filtered)
    return ply_filtrata

if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    source_raw = o3d.io.read_point_cloud("ply_data/obj_30.ply")
    target_raw = o3d.io.read_point_cloud("ply_data/obj_32.ply")
    #o3d.visualization.draw_geometries([source_raw])
    source = filtra(source_raw)
    #frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    #o3d.visualization.draw_geometries([source, frame])
    target = filtra(target_raw)

    #Esegue campionamento
    source = source.voxel_down_sample(voxel_size=0.001)
    target = target.voxel_down_sample(voxel_size=0.001)
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source)
    vis.add_geometry(target)
    threshold = 0.001
    icp_iteration = 100
    save_image = False
    print('Linea 32')
    for i in range(icp_iteration):
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, target, threshold, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
        source.transform(reg_p2l.transformation)
        print(reg_p2l.transformation)
        print(i)
        vis.update_geometry(source)
        vis.poll_events()
        vis.update_renderer()
        if save_image:
            vis.capture_screen_image("temp_%04d.jpg" % i)
    vis.destroy_window()   
