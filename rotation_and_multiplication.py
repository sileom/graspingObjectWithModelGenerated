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
#radians = 1.5708
radians = 0.785398
#radians = 0.523599
#radians = 3.14159
#radians = 4.71239
pcd_t_1 = o3d.geometry.PointCloud()
pcd = o3d.geometry.PointCloud()

def elementary_rotation(num, angle):
	if num == 1:
		matrix_rotate = np.asarray([[1, 0, 0, 0], 
						[math.cos(angle), -math.sin(angle), 0],
						[math.sin(angle), math.cos(angle)]])
	if num == 2:
		matrix_rotate = np.asarray([[cos(angle), 0, sin(angle)],
						[0, 1, 0], 
						[-sin(angle), 0, cos(angle)]])
	if num == 3:
		matrix_rotate = np.asarray([[np.cos(angle), -np.sin(angle), 0],
						 [np.sin(angle), np.cos(angle), 0],
						 [0, 0, 1]])
	return matrix_rotate

def rotation():
	angle = math.degrees(radians)

	pcd = o3d.io.read_point_cloud("/home/marika/Scrivania/Algoritmi/second_step/rotation/FINAL_OBJ_W/Modello_Finale.ply")
	open_file.open(pcd)

	pcd_matrix = np.asarray(pcd.points)

	B = np.zeros((4,4))

	pose_points = np.asarray([[-0.3986, -0.9168, 0, 0.0280],
   										[-0.8545, 0.3806, -0.3624, 0.0220],
   										[0.3322, -0.1211, -0.9320, -0.2430],
         								[0, 0, 0, 1]])

	file_matrix = open("/home/marika/Scrivania/Algoritmi/second_step/rotation/FINAL_OBJ_W/File_matrix.txt", "w")
	file_matrix.write("/home/marika/Scrivania/Algoritmi/second_step/rotation/FINAL_OBJ_W/Modello_Finale.ply ")
	file_matrix.write(str(pose_points[0,3]) + " " + str(pose_points[1,3]) + " " + str(pose_points[2, 3]) + " ")
	file_matrix.write(str(pose_points[0,0]) + " " + str(pose_points[1,0]) + " " + str(pose_points[2, 0]) + " ")
	file_matrix.write(str(pose_points[0,1]) + " " + str(pose_points[1,1]) + " " + str(pose_points[2, 1]) + " ")
	file_matrix.write(str(pose_points[0,2]) + " " + str(pose_points[1,2]) + " " + str(pose_points[2, 2]) + str("\n"))

	#i=1
	for i in range (10):
		#Matrix construction
		A = elementary_rotation(num, radians)
		A = np.append(A,np.zeros([len(A), 1]),1)
		newrow = [0, 0, 0, 1]
		A = np.vstack([A, newrow])

		if i == 0:
			B  = A.dot(pose_points)

			#file_matrix = open("/home/marika/Scrivania/Algoritmi/second_step/rotation/FINAL_OBJ_W/File_matrix.txt", "w")

			#file_matrix = open("/home/marika/Scrivania/Algoritmi/second_step/rotation/FINAL_OBJ_W/Modello_Finale.ply")

			file_matrix.write('/home/marika/Scrivania/Algoritmi/second_step/rotation/FINAL_OBJ_W/45gradi/Modello_Ruotato' + str(i) + '.ply ')
			file_matrix.write(str(B[0,3]) + " " + str(B[1,3]) + " " + str(B[2, 3]) + " ")
			file_matrix.write(str(B[0,0]) + " " + str(B[1,0]) + " " + str(B[2, 0]) + " ")
			file_matrix.write(str(B[0,1]) + " " + str(B[1,1]) + " " + str(B[2, 1]) + " ")
			file_matrix.write(str(B[0,2]) + " " + str(B[1,2]) + " " + str(B[2, 2]) + str("\n"))
		else:
			pose_points = B
			B  = A.dot(pose_points)
			file_matrix = open("/home/marika/Scrivania/Algoritmi/second_step/rotation/FINAL_OBJ_W/File_matrix.txt", "a")
			file_matrix.write('/home/marika/Scrivania/Algoritmi/second_step/rotation/FINAL_OBJ_W/45gradi/Modello_Ruotato' + str(i) + '.ply ')
			file_matrix.write(str(B[0,3]) + " " + str(B[1,3]) + " " + str(B[2, 3]) + " ")
			file_matrix.write(str(B[0,0]) + " " + str(B[1,0]) + " " + str(B[2, 0]) + " ")
			file_matrix.write(str(B[0,1]) + " " + str(B[1,1]) + " " + str(B[2, 1]) + " ")
			file_matrix.write(str(B[0,2]) + " " + str(B[1,2]) + " " + str(B[2, 2]) + str("\n"))
			file_matrix.close()

		#Visualization and save rotate object
		pcd_t_1 = copy.deepcopy(pcd)
		pcd = pcd.transform(A)
		o3d.visualization.draw_geometries([pcd_t_1, pcd])
		print("Saving of.." + str(i) + " rotation")
		o3d.io.write_point_cloud('/home/marika/Scrivania/Algoritmi/second_step/rotation/FINAL_OBJ_W/45gradi/Modello_Ruotato' + str(i) + '.ply', pcd, write_ascii=True)
		i=i+1
	
if __name__ == '__main__':
	rotation()
