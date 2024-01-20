#エッジを選択する

import maya.cmds as cmds
import random

def select_random_edge():
    # アクティブなオブジェクトの選択を取得
    selected_objects = cmds.ls(selection=True, dag=True, long=True)

    if selected_objects:
        # 選択されたオブジェクトのシェイプノードを取得
        shape_node = cmds.listRelatives(selected_objects[0], shapes=True, fullPath=True)
        
        if shape_node and cmds.nodeType(shape_node) == 'mesh':
            # メッシュのエッジを取得
            edges = cmds.polyListComponentConversion(shape_node, toEdge=True)
            edges = cmds.ls(edges, flatten=True)
            print(edges)
            edge_count = cmds.polyEvaluate(shape_node, edge=True)

            if edge_count > 0:
                # ランダムなエッジを選択
                random_edge_index = random.randint(0, edge_count - 1)
                print(edges)
                print(random_edge_index)
                print(len(edges))
                random_edge = edges[random_edge_index]
                cmds.select(random_edge)
            else:
                print("メッシュにエッジがありません。")
        else:
            print("選択されたオブジェクトはポリゴンメッシュではありません。")
    else:
        print("オブジェクトが選択されていません。")

# ランダムなエッジを選択
select_random_edge()


#----------------
edges = cmds.polyListComponentConversion(shape_node, toEdge=True)
edges = cmds.ls(edges, flatten=True) #これで*をつぶせる。

#----------------
#polySelectでelでedgeを選択する。
def select_edge_loop(object_name, edge_index):
    # オブジェクトのエッジを取得
    edges = cmds.polyListComponentConversion(object_name + '.e[' + str(edge_index) + ']', toEdge=True)
    edge_list = cmds.ls(edges, flatten=True)

    # エッジが存在するか確認
    if edge_list:
        # エッジループを選択a
        cmds.select(clear=True)
        cmds.polySelect(object_name, el=edge_index)
    else:
        print("指定されたエッジが見つかりません。")

# オブジェクトの名前とエッジインデックスを指定してエッジループを選択
object_name = "pCylinder1"
edge_index = 1545  # 適切なエッジインデックスに変更してください
select_edge_loop(object_name, edge_index)


#最小距離
import maya.cmds as cmds

def select_edge_loop(object_name, edge_index):
#    cmds.polySelect(object_name, el=edge_index)
    cmds.polySelect(object_name, shortestEdgePath=(10, 100))

object_name = "pCylinder1"
edge_index = 109  # 適切なエッジインデックスに変更してください
select_edge_loop(object_name, edge_index)


## エッジの長さを計算
import maya.cmds as cmds
import math

def calculate_selected_edge_lengths():
    # アクティブなエッジを取得
    selected_edges = cmds.ls(selection=True, flatten=True)
    print(selected_edges)

    # 選択されているエッジが存在するか確認
    if selected_edges:
        total_length = 0.0

        for edge in selected_edges:
            # エッジの両端の頂点座標を取得
            vertex_positions = cmds.polyInfo(edge, edgeToVertex=True)
            print(edge)
            print(vertex_positions)

            if vertex_positions:
                vertex_positions = vertex_positions[0].split(':')[1].strip().split()
                print(vertex_positions)
                vertex_pos = []
                vertex_pos.append(cmds.pointPosition('pCylinder1.vtx[' + vertex_positions[0] + ']', world=True))
                vertex_pos.append(cmds.pointPosition('pCylinder1.vtx[' + vertex_positions[1] + ']', world=True))
                print(vertex_pos)
                # エッジの長さを計算して合計に加算
                edge_length = math.sqrt((vertex_pos[0][0] - vertex_pos[1][0])**2 + (vertex_pos[0][1] - vertex_pos[1][1])**2 + (vertex_pos[0][2] - vertex_pos[1][2])**2)
                total_length += edge_length

        print("Total Selected Edge Length:", total_length)
    else:
        print("エッジが選択されていません。")

# 選択されているエッジの長さを計算
calculate_selected_edge_lengths()


##
import maya.cmds as cmds
import maya.mel as mel

#法線で展開
mel.eval('texNormalProjection 1 1 "" ;')
#展開
#cmds.u3dUnfold(ite=1, p=0, bi=1, tf=1, ms=128, rs=0)



##境界エッジを判定する
import maya.cmds as cmds

def is_boundary_edge(edge):
    # edgeの両端の頂点を取得
    faces = cmds.polyInfo(edge, edgeToFace=True)
    faces = faces[0].split()[2:]
    print(faces)
    if len(faces) == 1:
        print("boundary")
        return True
    else:
        print("not boundary")
        return False
        
# 選択しているエッジを取得
selected_edges = cmds.ls(selection=True, flatten=True)

if selected_edges:
    for edge in selected_edges:
        if is_boundary_edge(edge):
            print(f"{edge} は境界エッジです。")
        else:
            print(f"{edge} は境界エッジではありません。")
else:
    print("エッジが選択されていません。")

# edgeの選択
from maya import cmds

def is_border_edge(edge):
    face = cmds.ls(cmds.polyListComponentConversion(edge, tf=True), fl=True)
    return len(face) == 1
    
def is_seam_edge(edge):
    uv = cmds.ls(cmds.polyListComponentConversion(edge, tuv=True), fl=True)
    return len(uv) or is_border_edge(edge)
    
 def get_seam_edges(obj):
     seam_edges = []
     num_edge = cmds.polyEvaluate(obj, edge=True)
     for edge_num in range(num_edge):
         edge = '{}.e[{}]'.format(obj, edge_num)
         if is_seam_edge(edge):
             seam_edges.append(edge)
     return seam_edges



texture_size = 2048
desired_Tex_density = 10.24
def get_texel_density(faces, size= texture_size):
    ws_area = 0.0
    uv_area = 0.0
    for face in faces:
        ws_area += face.getArea(space='world')
        #ws_area = ws_area + face.getArea(space = "world")
    
        uv_area += face.getUVArea() # By default I hope it picks the current uv set

    # Get the square root of the area
    ws_area = math.sqrt(ws_area)
    uv_area = math.sqrt(uv_area)

    px_area = uv_area * size
    texel_density = (px_area / ws_area)
    return texel_density



from maya import cmds
obj = cmds.ls(selection=True, flatten=True)
print(obj)
uv = cmds.polyListComponentConversion( obj, toUV = True )
uv = cmds.ls(uv, flatten=True)
print(uv)
for uv_face in uv:
    print(uv_face)
    a = cmds.polyEvaluate( uv_face, boundingBox=True)
    print(a)




MayaのPythonスクリプトで、指定された三次元の点から最も近い頂点を選択するスクリプトを以下に示します。このスクリプトは、Maya APIのpointPosition関数を使用してオブジェクトの頂点座標を取得し、与えられた点との距離を計算して最も近い頂点を選択します。

python
Copy code
import maya.cmds as cmds
import maya.api.OpenMaya as om

def get_closest_vertex(target_point):
    # 選択されたオブジェクトを取得
    selected_objects = cmds.ls(selection=True, dag=True, shapes=True)

    if not selected_objects:
        cmds.warning("Please select a 3D object.")
        return

    closest_vertex = None
    min_distance = float('inf')

    # ターゲット座標をMPointに変換
    target_mpoint = om.MPoint(target_point[0], target_point[1], target_point[2])

    for obj in selected_objects:
        # オブジェクトの頂点座標を取得
        vtx_iter = om.MItMeshVertex(obj)
        while not vtx_iter.isDone():
            vertex_position = vtx_iter.position(om.MSpace.kWorld)
            distance = target_mpoint.distanceTo(vertex_position)

            if distance < min_distance:
                min_distance = distance
                closest_vertex = vtx_iter.index()

            vtx_iter.next()

    if closest_vertex is not None:
        cmds.select("{}.vtx[{}]".format(obj, closest_vertex))
        print("Closest vertex selected: {}".format(closest_vertex))
    else:
        cmds.warning("No vertices found in the selected objects.")

# 例: (x, y, z) 座標の点を指定して最も近い頂点を選択
target_point = (1.0, 2.0, 3.0)
get_closest_vertex(target_point)


import maya.cmds as cmds
import random

def toFlatten(components):
    return cmds.ls(components, fl=True, l=True)

#選択した頂点と隣接するエッジを選択
def find_adjacent_edges(vert, prev_vert):
    # エッジを含む頂点を取得
    connected_edges = toFlatten(cmds.polyListComponentConversion(vert, toEdge=True))
    connected_verts = toFlatten(cmds.polyListComponentConversion(connected_edges, toVertex=True))
    ban_list = [toFlatten(vert)[0], toFlatten(prev_vert)[0]]
    connected_verts = [elem for elem in connected_verts if elem not in ban_list]
    choiced_vert = random.choice(connected_verts)
    return choiced_vert
    

def find_connected_edge_from_vert(vert, num):

    v_list=[]
    v_list.append(toFlatten(vert)[0])
    for i in range(num):        
        if i == 0:
            v_next = find_adjacent_edges(v_list[-1], v_list[-1])
        else:           
            v_next = find_adjacent_edges(v_list[-1], v_list[-2])
        v_list.append(v_next)

    object_name = vert[0].split('.')[0]
    select_v_to_edge(v_list, object_name)
 
def select_v_to_edge(v_list, obj):
    cmds.select( clear=True )
    for i in range(len(v_list) - 1):
        index_0 = v_list[i].split('.')[1].split('[')[-1].rstrip(']')
        index_1 = v_list[i+1].split('.')[1].split('[')[-1].rstrip(']')
        edge = cmds.polySelect( obj, shortestEdgePath=(int(index_0), int(index_1) ) )
    edges = cmds.ls(selection=True, flatten=True)


selected_vert = cmds.ls(selection=True, flatten=True)
find_connected_edge_from_vert(selected_vert, 100)


This is develop