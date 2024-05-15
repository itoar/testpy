import igl
import scipy as sp
import numpy as np
from meshplot import plot, subplot, interact
import open3d as o3d

import os
root_folder = os.getcwd()

def convertZeroToOneFunc(k):
    min = np.min(k)
    k = k - min
    MAX = np.max(k)
    k = k/MAX
    return k

def makeColorArray(h):
    col = np.array([[1.0, 1.0, 1.0]])
    color_list = np.empty((0,3), float)
    for i in h:
        color_list = np.append(color_list, i*col, axis=0)
    return color_list

def drawMesh(vertex, face, color):
    mesh = o3d.geometry.TriangleMesh()

    mesh.vertices = o3d.utility.Vector3dVector(vertex)
    mesh.triangles = o3d.utility.Vector3iVector(face)
    mesh.vertex_colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([mesh],mesh_show_back_face=True)    

def extractVertexUnderThreshold(k, vertex, threshold):
    index = np.where(k < threshold)
    new_v = np.empty((0,3), float)
    for idx in index:
        new_v = np.append(new_v, vertex[idx], axis=0)
    return new_v


v, f = igl.read_triangle_mesh("C:/sandbox/igl/libigl/build/_deps/libigl_tutorial_data-src/bumpy.off")
k = igl.gaussian_curvature(v, f)

m = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
minv = sp.sparse.diags(1 / m.diagonal())
kn = minv.dot(k)

kn = np.array(kn)
npk = convertZeroToOneFunc(kn)
color_list = makeColorArray(npk)

new_v = extractVertexUnderThreshold(npk, v, 0.2 )

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(new_v)
#pcd.colors = o3d.utility.Vector3dVector(colors)
drawMesh(v, f, color_list)
o3d.visualization.draw_geometries([pcd])


# np_vertices = v
# np_color = color_list
# np_triangles = f

# mesh.vertices = o3d.utility.Vector3dVector(np_vertices)
# mesh.triangles = o3d.utility.Vector3iVector(np_triangles)
# mesh.vertex_colors = o3d.utility.Vector3dVector(np_color)
# o3d.visualization.draw_geometries([mesh],mesh_show_back_face=True)

#plot(V, F)