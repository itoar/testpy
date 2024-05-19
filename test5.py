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



coff_mat = []
for j in range(JNum):
    tmp = []
    for i in alpha:
        tmp.extend([i,i,i])
    coff_mat.append(tmp)
coff_mat = np.array(coff_mat)
print(coff_mat)

A = 100.0*np.ones_like(coff_mat)
print(A)
B = coff_mat * A
print(B)

def rotation_to_matrix(rotation):
    rx = math.radians(rotation[0])
    ry = math.radians(rotation[1])
    rz = math.radians(rotation[2])

    cos_rx = math.cos(rx)
    sin_rx = math.sin(rx)
    cos_ry = math.cos(ry)
    sin_ry = math.sin(ry)
    cos_rz = math.cos(rz)
    sin_rz = math.sin(rz)

    matrix = [
        [cos_ry * cos_rz, cos_rz * sin_rx * sin_ry - cos_rx * sin_rz, sin_rx * sin_rz + cos_rx * cos_rz * sin_ry, 0],
        [cos_ry * sin_rz, cos_rx * cos_rz + sin_rx * sin_ry * sin_rz, cos_rx * sin_ry * sin_rz - cos_rz * sin_rx, 0],
        [-sin_ry, cos_ry * sin_rx, cos_rx * cos_ry, 0],
        [0, 0, 0, 1]
    ]

    return matrix

def get_combined_local_matrix(joint_name):
    joint_names = []
    current_joint = joint_name
    while current_joint:
        joint_names.append(current_joint)
        current_joint = cmds.listRelatives(current_joint, parent=True)
        if current_joint:
            current_joint = current_joint[0]    
    return joint_names

hoge = get_combined_local_matrix(pjoint_name)
rtmat = np.identity(4)

for j in hoge:
    rot = cmds.xform(j, query=True, rotation=True, ws=False)
    rot_mat = np.array(rotation_to_matrix(rot))
    rot_mat[3,:] = np.array([0.0,0.0,0.0,1.0])
    rtmat = rot_mat @ rtmat
rotate_mat = np.array([[0.5, 0.5, 0.0, 0.0],[-0.5, 0.5, 0.0, 0.0],[0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 0.0, 1.0]])
rotate_mat = np.linalg.inv(rtmat) @ rotate_mat

cmds.xform(joint_name , matrix=rotate_mat.ravel(), worldSpace=True)


def calcRotation2Mat(rotation):
    # オイラー角を作成
    euler = om2.MEulerRotation(math.radians(rotation[0]), math.radians(rotation[1]), math.radians(rotation[2]), om2.MEulerRotation.kXYZ)
    return np.array(euler.asMatrix()).reshape([4,4])

def calcMat2Rotation(Mat):
    util = om.MScriptUtil()
    mat = om.MMatrix()
    util.createMatrixFromList(Mat.ravel().tolist(), mat)
    rot = om.MEulerRotation.decompose(mat, om.MEulerRotation.kXYZ)
    return [math.degrees(rot.x), math.degrees(rot.y), math.degrees(rot.z)]

def getJointOrient(joint_node):
    x = cmds.getAttr(joint_node + ".jointOrientX")
    y = cmds.getAttr(joint_node + ".jointOrientY")
    z = cmds.getAttr(joint_node + ".jointOrientZ")
    return [x, y, z]

def calcJointMatrix(joint_node):
    w_mat = getJointWorldMatrix(joint_node)

    print("--------------")
    print(f"{joint_node} world mat")
    print(w_mat)
    rotation = calcMat2Rotation(w_mat)
    print("rotation")
    print(rotation)
    print("--------------")


def setRotation(joint_node, rotation_input):
    base_mat = getJointWorldMatrix(joint_node)
    base_mat = base_mat.T
    rotation_mat = calcRotation2Mat(rotation_input)

    parent = cmds.listRelatives(joint_node, parent=True)
    parent_mat = getJointWorldMatrix(parent)
    parent_mat = parent_mat.T

    joint_rot = getJointOrient(joint_node)
    joint_rot_mat = calcRotation2Mat(joint_rot)
    joint_rot_mat = joint_rot_mat[0:3,0:3]

    parent_mat_rot = parent_mat[0:3,0:3]
    rotation_mat = rotation_mat[0:3,0:3]
    rot = rotation_mat @ joint_rot_mat @ parent_mat_rot

    base_mat[0:3,0:3] = rot[0:3,0:3]


    cmds.xform(joint_node, matrix=base_mat.ravel(), ws=True)

