import numpy as np
import maya.cmds as cmds
import maya.api.OpenMaya as om2
import maya.api.OpenMayaAnim as oma2
import maya.mel as mel 
import numpy as np

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
        cmds.select(faces)
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
    create_sphere_at_position(center_pos, 0.02)
    normal = [0.0, 0.0, 0.0]
    for n in n_list:
        normal += n
    normal = normal/len(n_list)
    pos = center_pos + 0.1*normal
    create_sphere_at_position(pos, 0.02)



vert = getObjectVertices()

selected_object = cmds.ls(selection=True)
components = cmds.polyListComponentConversion(
                selected_object, fv=1, ff=1, fe=1, fuv=1, fvf=1, tv=1
                )
components = cmds.ls(components, fl=True, l=True)
dict = getNeigborVertexIdList(components)
computeNormal(10, dict[10])
for key in dict.keys():
    computeNormal(key, dict[key])