import time
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import cv2
import lib

global pcd

pcd_1 = o3d.geometry.PointCloud()
pcd_2 = o3d.geometry.PointCloud()
pcd = o3d.geometry.PointCloud()
vis = o3d.visualization.Visualizer()

def main():
    # Configure depth and color stream
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
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    align_to = rs.stream.depth

    align = rs.align(align_to)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    #Create window
    vis.create_window(width=640, height=480)
    #vis.create_window()
    pc_0 = rs.pointcloud()
    points = rs.points()
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

    pc_0.map_to(color_frame)
    points = pc_0.calculate(depth_frame)

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape

    vtx_0 = np.asanyarray(points.get_vertices())

    pcd.points = o3d.utility.Vector3dVector(vtx_0.tolist())

    #Filter the point cloud
    pcd_1.points = lib.filtra(pcd)
    print(len(pcd.points))

    vis.add_geometry(pcd_1)
    vis.poll_events()
    vis.update_renderer()

    
    try:
        i=0
        while True:
            pc = rs.pointcloud()
            points = rs.points()

            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            if not depth_frame or not color_frame:
                continue
            pc.map_to(color_frame)

            points = pc.calculate(depth_frame)

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            vtx = np.asanyarray(points.get_vertices())

            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))

            pcd.points = o3d.utility.Vector3dVector(vtx.tolist())

            pcd_2.points = lib.filtra(pcd)
            pcd_1.points = lib.combine_point_cloud(pcd_1, pcd_2)
            
            #Update new point cloud
            vis.update_geometry(pcd_1)
            vis.update_renderer()
            vis.poll_events()

            i=i+1


            time.sleep(0.05)
            if i >100:
                break
    finally:
        o3d.io.write_point_cloud("model.ply", pcd_1, write_ascii=True)


if __name__ == "__main__":
    main()