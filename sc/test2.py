import numpy as np
import maya.cmds as cmds
import maya.api.OpenMaya as om2
import maya.api.OpenMayaAnim as oma2
import maya.mel as mel     # maya.melモジュールをインポートし

def test():
    print("test2")
    a = np.array([0,0,0,0])
    print(a)
    selection = cmds.ls(selection=True, dag=True, type='mesh')
    print(selection)

def getObjectVertices():
    sll = om2.MGlobal.getActiveSelectionList()
    mesh = om2.MFnMesh(sll.getDagPath(0))
    x = mesh.getPoints()
    list = np.array([])
    for i, t in enumerate(x):
        list = np.append(list, np.array([t[0], t[1], t[2]]))
        print(i, " ", t[0], t[1], t[2])
    list = np.reshape(list, (-1, 3))
    print(list)
    return list

def getWeight():
    sll = om2.MGlobal.getActiveSelectionList()
    mesh = om2.MFnMesh(sll.getDagPath(0))
    # skinNode = n(SKINCLUSTER)
    skinFn = om2.MFnSkinCluster(mesh)
    singleIdComp = om2.MFnSingleIndexedComponent()
    vertexComp = singleIdComp.create(om2.MFn.kMeshVertComponent)
    weightData = skinFn.getWeights(mesh.node(), vertexComp)
    print(weightData)

def create_skincluster_fn():
    # type: () -> (om2.MFnSkinCluster, om2.MDagPath, om2.MObject)
    active_sel_list = om2.MGlobal.getActiveSelectionList()          # type: om2.MSelectionList
    dag_path, m_object_component = active_sel_list.getComponent(0)  # type: (om2.MDagPath, om2.MObject)
    mesh_dag_path = dag_path.extendToShape()                        # type: om2.MDagPath
    skincluster_node = get_skincluster_node(mesh_dag_path.node())   # type: om2.MObject
    skincluster_fn = oma2.MFnSkinCluster(skincluster_node)          # type: om2.MFnSkinCluster
    return skincluster_fn, mesh_dag_path, m_object_component

def get_skincluster_node(mesh_node):
    # type: (om2.MObject) -> om2.MObject
    dg_iterator = om2.MItDependencyGraph(mesh_node, om2.MFn.kSkinClusterFilter, om2.MItDependencyGraph.kUpstream) # type: om2.MItDependencyGraph
    while not dg_iterator.isDone():
        m_object = dg_iterator.currentNode()
        if m_object.hasFn(om2.MFn.kSkinClusterFilter):
            return m_object
        dg_iterator.next()
    return None

def create_vertex_component():
    # type: () -> om2.MObject
    single_idx_comp_fn = om2.MFnSingleIndexedComponent()            # type: om2.MFnSingleIndexedComponent
    return single_idx_comp_fn.create(om2.MFn.kMeshVertComponent)    # type: om2.MObject

def get_influence_index_by_name(skincluster_fn, influence_name):
    # type: (oma2.MFnSkinCluster, str) -> int
    influences = skincluster_fn.influenceObjects()                      # type: om2.MDagPathArray
    for influence in influences:
        if influence.__str__() == influence_name:
            return skincluster_fn.indexForInfluenceObject(influence)    # type: int # Python2系はlong
    return None

def getSkinWeight():
    skincluster_fn, mesh_dag_path, m_object_component = create_skincluster_fn()                         # type: (om2.MFnSkinCluster, om2.MDagPath, om2.MObject)

    influences = skincluster_fn.influenceObjects()
    list = np.array([])
    is_first = True
    for influence in influences:
        index = skincluster_fn.indexForInfluenceObject(influence)
        skin_Weights = skincluster_fn.getWeights(mesh_dag_path, m_object_component, index)
        if is_first == True:
            list = np.array(skin_Weights)
            is_first = False
        else:
            tmp_list = np.array(skin_Weights)
            list = np.vstack([list, tmp_list])
    list = list.T
    print(list)

    return list

def getVertexAndWeight():
    print("getVertexAndWeight")
    vert = getObjectVertices()
    weight = getSkinWeight()
    vertAndWeight = {"vert":vert, "weight":weight}
    return vertAndWeight

def getBindVertexAndWeight():
    print("getBindVertexAndWeight")
    mel.eval("dagPose -r -g -bp joint1;")
    vert = getObjectVertices()
    weight = getSkinWeight()
    mel.eval("Undo;")
    vertAndWeight = {"vert":vert, "weight":weight}

    return vertAndWeight

def getJointList():
    selected_joints = cmds.ls(dagObjects=True, type='joint')
    return selected_joints

def getJointMatrix(joint):
    # matrix = cmds.xform(joint, query=True, matrix=True, worldSpace=True)
    matrix = cmds.xform(joint, query=True, matrix=True)
    mat = np.array(matrix)
    mat = np.reshape(mat, (4, 4))
    print(mat)
    return mat.T

def getJointWorldMatrix(joint):
    matrix = cmds.xform(joint, query=True, matrix=True, worldSpace=True)
    mat = np.array(matrix)
    mat = np.reshape(mat, (4, 4))
    print(mat)
    return mat.T


def getJointsMatrix(isWorldSpace = False):
    joints = getJointList()
    vec = np.array([0.0, 0.0, 0.0, 1.0])
    mat_list = []
    for joint in joints:
        if isWorldSpace == False:
            mat = getJointMatrix(joint)
        else:
            mat = getJointWorldMatrix(joint)
        new_vec = mat@vec
        print(new_vec)
        mat_list.append(mat)
    return mat_list



def maketest():
    #シェイプの頂点とウェイトを取得する。
    vertAndWeight = getBindVertexAndWeight()
    transformed_vert = []
    for v_idx in range(len(vertAndWeight["vert"])):
        vec_target = vertAndWeight["vert"][v_idx]
        vec_target = np.append(vec_target, 1.0)
        weight = vertAndWeight["weight"][v_idx]
        print("vec_target")
        print(vec_target)
        print("weight")
        print(weight)

        mat_list = getJointsMatrix(isWorldSpace=True)
        mat_bind_list = getBindJointMatrix()

        vec_list = []
        for i in range(len(mat_list)):
            off_mat = mat_bind_list[i]
            off_mat = np.linalg.inv(off_mat)
            tmp_vec = off_mat@vec_target
            new_vec = mat_list[i]@tmp_vec
            vec_list.append(new_vec)

        solution = np.array([0.0, 0.0, 0.0, 1.0])
        for i in range(len(weight)):
            solution += weight[i]*vec_list[i]
        transformed_vert.append(solution)
    for vert in transformed_vert:
        create_sphere_at_position(vert, radius=0.1)
def create_sphere_at_position(position, radius=1.0):
    # 指定された位置に球を生成する
    sphere = cmds.polySphere(radius=radius)[0]
    # 球の位置を設定する
    cmds.move(position[0], position[1], position[2], sphere, absolute=True)


def getBindJointMatrix():
    #バインドポーズにする
    mel.eval("dagPose -r -g -bp joint1;")
    mat_bind_list = getJointsMatrix(isWorldSpace=True)
    mel.eval("Undo;")
    # mel.eval("undo;")
    return mat_bind_list


-----------------------------------------

def get_bound_joints(mesh_name):
    skin_cluster = cmds.ls(cmds.listHistory(mesh_name), type="skinCluster")
    if skin_cluster:
        bound_joints = cmds.skinCluster(skin_cluster[0], query=True, influence=True)
        return bound_joints
    else:
        return []

# Example usage:
selected_mesh = cmds.ls(selection=True)
if selected_mesh:
    bound_joints = get_bound_joints(selected_mesh[0])
    for i in bound_joints:
        print(i)
    cmds.select(bound_joints[0])


def DuplicateForTarget():
    obj = cmds.ls(selection=True)
    mel.eval("Duplicate;")
    cmds.select(obj)
    mel.eval("ToggleVisibilityAndKeepSelection;")

def getObjectVertices():
    sll = om2.MGlobal.getActiveSelectionList()
    mesh = om2.MFnMesh(sll.getDagPath(0))
    x = mesh.getPoints()
    list = np.array([])
    for i, t in enumerate(x):
        list = np.append(list, np.array([t[0], t[1], t[2]]))
        print(i, " ", t[0], t[1], t[2])
    list = np.reshape(list, (-1, 3))
    return list
