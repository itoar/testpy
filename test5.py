import numpy as np
import maya.cmds as cmds
import maya.api.OpenMaya as om2
import maya.api.OpenMayaAnim as oma2
import maya.mel as mel 
import numpy as np


def create_sphere_at_position(position, radius=1.0):
    # 指定された位置に球を生成する
    sphere = cmds.polySphere(radius=radius)[0]
    # 球の位置を設定する
    cmds.move(position[0], position[1], position[2], sphere, absolute=True)

def getObjectVertices():
    sll = om2.MGlobal.getActiveSelectionList()
    mesh = om2.MFnMesh(sll.getDagPath(0))
    x = mesh.getPoints()
    lists = np.array([])
    for i, t in enumerate(x):
        lists = np.append(lists, np.array([t[0], t[1], t[2]]))
    lists = np.reshape(lists, (-1, 3))
    return lists
    
def getNeigborVertexIdList(components):
    dict = {}
    for obj in components:
        v_center_id = int(obj.split(".vtx[")[1].split("]")[0])
        obj2 = cmds.polyListComponentConversion(obj, toEdge=True, fromVertex=True)
        faces = cmds.polyListComponentConversion(obj, toFace=True, fromVertex=True)
        obj3 = cmds.polyListComponentConversion(obj2, toVertex=True, fromEdge=True)
        obj4 = cmds.ls(obj3, fl=True, l=True)
        v_lists = []
        for i in obj4:
            string = i.split(".vtx[")[1].split("]")[0]
            if int(string) != v_center_id:
                v_lists.append(string)
        
        faces = cmds.ls(faces, fl=True, l=True)
        v_pair_list = []
        for f in faces:
            f_v = cmds.polyListComponentConversion(f, toVertex=True, fromFace=True)
            f_v = cmds.ls(f_v, fl=True, l=True)
            face_v_list = []
            for vid in f_v:
                id = vid.split(".vtx[")[1].split("]")[0]
                face_v_list.append(id)
            vset = set(face_v_list) & set(v_lists)
            
            #normalと比較して、必要なら入れ替える#
            temp   = cmds.polyInfo(f, fn=True)[0]
            temp   = temp.split(' ')
            normal = np.array([float(temp[-3]), float(temp[-2]), float(temp[-1])])
            #vertex_normal計算
            pair = list(vset)
            v_1_id = pair[0]
            v_2_id = pair[1]
    
            p1 = np.array(vert[int(v_1_id)] - vert[v_center_id])
            p2 = np.array(vert[int(v_2_id)] - vert[v_center_id])
            c = np.cross(p1, p2)
            v_normal = c/np.linalg.norm(c)
            if np.dot(normal, v_normal) < 0:
                pair = [pair[1], pair[0]]
                
            v_pair_list.append(pair)


            
        dict[v_center_id] = v_pair_list
    return dict    
    
def sortToSequenciapEdgePair(edge_vertex_list):
    sorted_list = []
    first_popped = edge_vertex_list.pop(0)
    popped = first_popped
    sorted_list.append(list(popped))
    while edge_vertex_list != []:
        for idx, v_list in enumerate(edge_vertex_list):
            if popped.isdisjoint(v_list) == False:
                sorted_list.append(list(v_list))
                popped = edge_vertex_list.pop(idx)
                break
    sorted_list.append(list(first_popped))
    return sorted_list
            
            
    
    

def getFaceIdList(vert):
    selected_object = cmds.ls(selection=True)
    obj = cmds.polyListComponentConversion(selected_object, toFace=True)
    obj = cmds.ls(obj, fl=True, l=True)
    
    for idx, face in enumerate(obj):
        print(face)
        temp   = cmds.polyInfo(face, fn=True)[0]
        temp   = temp.split(' ')
        face_normal = np.array([float(temp[-3]), float(temp[-2]), float(temp[-1])])
    
        edges = cmds.ls(cmds.polyListComponentConversion(face, toEdge=True), fl=True, l=True)
        edge_vertex_list = []
        for e in edges:
            vids = cmds.ls(cmds.polyListComponentConversion(e, toVertex=True), fl=True, l=True)
            vids = { int(v.split(".vtx[")[1].split("]")[0]) for v in vids }
            edge_vertex_list.append(vids)
        sorted_edge_pair = sortToSequenciapEdgePair(edge_vertex_list)

        print(f"list {sorted_edge_pair}")
        normal_list = []
        for idx in range(len(sorted_edge_pair) - 1):    
            normal = computeNormalVec(sorted_edge_pair[idx], sorted_edge_pair[idx+1], vert)
            if np.dot(normal, face_normal) < 0:
                sorted_edge_pair[idx+1] = [sorted_edge_pair[idx+1][1], sorted_edge_pair[idx+1][0]] 
        for edgeee in sorted_edge_pair:
            print(f"pair {edgeee}")
        center_pos = np.zeros(3)
        ave_normal = np.zeros(3)
        for idx in range(len(sorted_edge_pair) - 1):    
            center, normal = computeNormalVec2(sorted_edge_pair[idx], sorted_edge_pair[idx+1], vert)
            center_pos += center
            print(f"center {center}")
            ave_normal += normal
#        center, normal = computeNormalVec2(sorted_edge_pair[-1], sorted_edge_pair[0], vert)
#        ave_normal += normal
#        center_pos += center
        
        center_pos = center_pos/(len(sorted_edge_pair) - 1)
        print(f"center_pos {center_pos} len {len(sorted_edge_pair)}")
        ave_normal = ave_normal/(len(sorted_edge_pair) - 1)
        create_sphere_at_position(center_pos, 0.02)
        create_sphere_at_position(ave_normal*0.1 + center_pos, 0.02)
        

def computeNormalVec(edge_1, edge_2, vert):
    p1 = vert[edge_1[0]] - vert[edge_1[1]]
    p2 = vert[edge_2[0]] - vert[edge_2[1]]
    c = np.cross(p1, p2)
    norm = c**2
    norm = norm.sum()
    norm = norm**0.5
    c = c/norm
    return c
    
def computeNormalVec2(edge_1, edge_2, vert):
    p1 = vert[edge_1[0]] - vert[edge_1[1]]
    p2 = vert[edge_2[0]] - vert[edge_2[1]]
    if edge_1[0] == edge_2[0]:
        center = edge_1[0]
    else:
        center = edge_1[1]
    c = np.cross(p1, p2)
    norm = c**2
    norm = norm.sum()
    norm = norm**0.5
    c = c/norm
    center_pos = vert[center]
    print(f"in center {center_pos}")
#    create_sphere_at_position(center_pos, 0.02)
#    create_sphere_at_position(c*0.1 + center_pos, 0.02)

    return center_pos, c    

    
def computeNormal(center, PairList):
    n_list = []
    for pair in PairList:
        v_1_id = pair[0]
        v_2_id = pair[1]

        p1 = np.array(vert[int(v_1_id)] - vert[center])
        p2 = np.array(vert[int(v_2_id)] - vert[center])
        c = np.cross(p1, p2)
        c = c/np.linalg.norm(c)
        n_list.append(c)
    n_list = np.array(n_list)
    center_pos = vert[center]
    normal = [0.0, 0.0, 0.0]
    for n in n_list:
        normal += n
    normal = normal/len(n_list)
    pos = center_pos + 0.1*normal




#selected_object = cmds.ls(selection=True)
#components = cmds.polyListComponentConversion(
#                selected_object, fv=1, ff=1, fe=1, fuv=1, fvf=1, tv=1
#                )
#components = cmds.ls(components, fl=True, l=True)
#dict = getNeigborVertexIdList(components)
#computeNormal(10, dict[10])
#for key in dict.keys():
#    computeNormal(key, dict[key])
#print("end")
vert = getObjectVertices()
getFaceIdList(vert)