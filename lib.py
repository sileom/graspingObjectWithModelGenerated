import open3d as o3d
import copy

global theshold_fitness
global theshold_correspondence_set

theshold_fitness = 0.0003
theshold_correspondence_set = 50

def union(source_down, target_down):
    print("Make a combined point cloud")
    pcds = [source_down, target_down]
    pcd_combined = o3d.geometry.PointCloud()
    for i in range (len(pcds)):
        pcd_combined += pcds[i]
    #o3d.io.write_point_cloud('test_combined.ply', pcd_combined, write_ascii=True)
    #o3d.visualization.draw_geometries([pcd_combined])
    return pcd_combined.points

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    #o3d.visualization.draw_geometries([source_temp, target_temp],
                                       #zoom=0.00001, front=[0, 0, 1],
                                       #lookat=[0, 0, 1],
                                        #up=[0, 1, 0])

def prepare_dataset(voxel_size):
    print("\nLoad two point clouds and disturb initial pose.")
    source_down = o3d.io.read_point_cloud("prova_u0.ply")
    #o3d.visualization.draw_geometries([source_down])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    target_down = o3d.io.read_point_cloud("prova_u1.ply")
    #o3d.visualization.draw_geometries([target_down])
    source = filtra(source_down)
    target = filtra(target_down)
    #o3d.io.write_point_cloud("sf.ply", source, write_ascii=True)
    #o3d.io.write_point_cloud("tf.ply", target, write_ascii=True)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def filtra(source_down):
    ply_filtrata = o3d.geometry.PointCloud()
    min_bound = [-0.09, -0.05, -0.30] #-0.40 #0  #-0.10, -0.10, 0.0 #-10.10, -10.10, -0.40
    max_bound = [0.15, 0.13, 0] #0 #0.40    #0.10, 0.10, 0.30       #10.10, 10.10, 0
    points_list_filtered = []
    print(len(source_down.points))
    for i in range(len(source_down.points)):
        p = source_down.points[i]
        if((min_bound[0] <= p[0] and p[0] <= max_bound[0]) and (min_bound[1] <= p[1] and p[1] <= max_bound[1])
            and (min_bound[2] < p[2] and p[2] <= max_bound[2])):
            #print(p)
            points_list_filtered.append(p)
        #print(p)
    ply_filtrata.clear()
    ply_filtrata.points = o3d.utility.Vector3dVector(points_list_filtered)
    return ply_filtrata.points

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
    #print(":: RANSAC registration on downsampled point clouds.")
    #print("   Since the downsampling voxel size is %.3f," % voxel_size)
    #print("   we use a liberal distance threshold %.3f." % distance_threshold)
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

def combine_point_cloud(pcd_1, pcd_2):
    voxel_size = 0.0007
    #print(type(pcd_1))
    #pcd_1 = filtra(pcd_1)
    #pcd_2 = filtra(pcd_2)
    pcd_1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_1, pcd_1_fpfh = preprocess_point_cloud(pcd_1, voxel_size)
    pcd_2, pcd_2_fpfh = preprocess_point_cloud(pcd_2, voxel_size)
    result_ransac = execute_global_registration(pcd_1, pcd_2,
                                            pcd_1_fpfh, pcd_2_fpfh,
                                            voxel_size)
    print(result_ransac)
    #draw_registration_result(pcd_1, pcd_2, result_ransac.transformation)
    print(len(result_ransac.correspondence_set))
    if((result_ransac.fitness >= theshold_fitness) or (len(result_ransac.correspondence_set) >= theshold_correspondence_set)):
        print("Unione precedente avvenuta")
        return union(pcd_1.transform(result_ransac.transformation), pcd_2)
    else:
        return union(pcd_1, pcd_2)
    #o3d.visualization.draw_geometries([pcd_1])
    #update_visualization(pcd_1, pcd_2, vis)

def conversion_to_millimeters(pcd):
    lista_filtrata = o3d.geometry.PointCloud()
    points_list = []
    for i in range(len(pcd.points)):
        p = pcd.points[i]
        points_list.append(p*1000)
    lista_filtrata.clear()
    lista_filtrata.points = o3d.utility.Vector3dVector(points_list)
    return lista_filtrata.points
