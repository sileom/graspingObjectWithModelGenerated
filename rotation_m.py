import open3d as o3d 
import pyrealsense2 as rs
import numpy as np
import cv2
import lib_m
import lib
import math
import open_file
import copy

global pcd_t_1
global pcd

num = 3
radians = 1.5708
pcd_t_1 = o3d.geometry.PointCloud()
pcd = o3d.geometry.PointCloud()

def elementary_rotation(num, angle):
	#if num == 1:
		#pcd_rotate = [1, 0, 0, 0, math.cos(angle), -math.sin(angle), 0, math.sin(angolo), math.cos(angolo)]
		#o3d.visualizer.draw_geometries([pcd_rotate])
	#if num == 2:
		#R << cos(angolo), 0, sin(angolo), 0, 1, 0, -sin(angolo), 0, cos(angolo)
	if num == 3:
		matrix_rotate = np.asarray([[math.cos(angle), -math.sin(angle), 0],
						 [math.sin(angle), math.cos(angle), 0],
						 [0, 0, 1]])
		#print(matrix_rotate)
	return matrix_rotate

def rotation():
	angle = math.degrees(radians)

	pcd = o3d.io.read_point_cloud("second_step/rotation/Modello_Finale.ply")
	open_file.open(pcd)

	pcd_matrix = np.asarray(pcd.points)

	i=1
	for i in range (3):
		#Matrix construction
		A = elementary_rotation(num, angle)
		A = np.append(A,np.zeros([len(A), 1]),1)
		newrow = [0, 0, 0, 1]
		A = np.vstack([A, newrow])

		#Visualization and save rotate object
		pcd_t_1 = copy.deepcopy(pcd)
		pcd = pcd.transform(A)
		o3d.visualization.draw_geometries([pcd_t_1, pcd])
		print("Saving of.." + str(i) + "rotation")
		o3d.io.write_point_cloud('second_step/rotation/Modello_Ruotato' + str(i) + '.ply', pcd, write_ascii=True)
		i=i+1
	
if __name__ == '__main__':
	rotation()
