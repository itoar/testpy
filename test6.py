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

def testtest():
    sll = om2.MGlobal.getActiveSelectionList()
    mesh = om2.MFnMesh(sll.getDagPath(0))
    sll_str = str(sll)
    r = re.findall('\[\w+\]', sll_str)  # 「英数字:」を抽出
    print(r)  # ['id:', 'mail:', 'tel:']
    indice = []
    for i in r:
        print(i)
        print(re.sub('\[|\]', '', i))
        indice.append(re.sub('\[|\]', '', i))
    print(indice)
    print(sll_str)
    print(sll.getDagPath(0))
    indice = np.array(indice)
#    points = om2.MPointArray(indice)
    print(mesh.getPoint(0))
    print(mesh.getPoints())

    list = np.array([])
    for idx in indice:
        x = mesh.getPoint(int(idx))
        list = np.append(list, np.array([x[0], x[1], x[2]]))
        print(idx, " ", x[0], x[1], x[2])
    list = np.reshape(list, (-1, 3))
    print(list)

def processWeight(K=2):
    w = getSkinWeight()
    list = np.array([])
    is_first = True
    for i in w:
        ii = np.argsort(i)
        i[ii[:-K]] = 0
        res = 1.0 - np.sum(i)
        i[ii[-K:]] += res/K
        if is_first == True:
            list = np.array(i)
            is_first = False
        else:
            tmp_list = np.array(i)
            list = np.vstack([list, tmp_list])
    print(list)

def getMeshs():
    slist = om2.MGlobal.getActiveSelectionList()
    itsl = om2.MItSelectionList(slist)
    meshPaths = []
    while not itsl.isDone():
        dagPath = itsl.getDagPath()
        itsl.next()
        if dagPath is None:
            continue
        apiType = dagPath.apiType()
        if apiType != om2.MFn.kTransform:
            continue
        for c in range(dagPath.childCount()):
            child = dagPath.child(c)
            if child.apiType() != om2.MFn.kMesh:
                continue
            path = dagPath.getAPathTo(child)
            mesh = om2.MFnMesh(path)
            if not mesh.findPlug('intermediateObject', True).asBool():
                meshPaths.append(path)
                break
    return meshPaths
#meshpath内の頂点の行列を作る。
def concatenatePointLists(meshPaths):
    retval = np.empty([0, 3])
    for path in meshPaths:
        mesh = om2.MFnMesh(path)
        points = mesh.getPoints(om2.MSpace.kWorld)
        points = np.array([[p.x, p.y, p.z] for p in points])
        retval = np.append(retval, points.reshape(-1, 3), axis=0)
    return retval

def getJoints():
    asl = om2.MGlobal.getActiveSelectionList()
    itsl = om2.MItSelectionList(asl)
    jointPaths = []
    while not itsl.isDone():
        try:
            dagPath = itsl.getDagPath()
        except:  continue
        finally: itsl.next()
        if not dagPath:
            continue
        apiType = dagPath.apiType()
        if apiType == om2.MFn.kJoint:
            jointPaths.append(dagPath)
    return jointPaths

def dup():
    cmds.duplicate(returnRootsOnly=True, name="cloneMeshName", renameChildren=True)

def tetet():
    neighbor = []
    meshPaths = getMeshs()
    for path in meshPaths:
        mesh = om2.MFnMesh(path)
        _, indices = mesh.getTriangles()
        offset = len(neighbor)
        neighbor = neighbor + [set() for v in range(mesh.numVertices)]
        print(neighbor)
        print(indices)
        print(len(indices))
        for l in range(len(indices) / 3):
            i0 = indices[l * 3 + 0] + offset
            i1 = indices[l * 3 + 1] + offset
            i2 = indices[l * 3 + 2] + offset
            neighbor[i0].add(i1)
            neighbor[i0].add(i2)
            neighbor[i1].add(i0)
            neighbor[i1].add(i2)
            neighbor[i2].add(i0)
            neighbor[i2].add(i1)
        print(neighbor)
import maya.OpenMaya as om
def set_joint_matrix(joint_name, matrix, ws_enable = True):
    """
    指定した行列でジョイントのトランスフォームを設定する関数

    Args:
        joint_name (str): 移動したいジョイントの名前
        matrix (list of list of float): 4x4のトランスフォームマトリックス
    """
    if not cmds.objExists(joint_name):
        print(f"ジョイント {joint_name} が存在しません")
        return

    # OpenMayaのMMatrixを使用して行列を設定

    # xformコマンドを使用して行列を設定
    cmds.xform(joint_name, matrix=np.array(matrix).ravel(), worldSpace=True)

def set_rotate(joint_name, rotation):
    set_joint_matrix(joint_name, rotate=rotation, ws=False)


def ThisIsATest():
    print("OK This is Test Function")
    print("This test")
    print("test is a test")

def TestOK():
    # 使用例
    joint_name = "joint4"
    # 例として単位行列を使用 (通常は他の行列を使用)
    target_matrix = [
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 1.0, 1.0, 1.0]
    ]
    target_matrix2 = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 1.0, 1.0, 1.0]
    ]



    rotate = [
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]

    set_joint_matrix(joint_name, np.array(target_matrix2))
    cmds.xform(joint_name, rotation=[0.0, 0.0, 0.0], worldSpace=False)

def TestOK2():
    matrix = cmds.xform("joint4", query=True, matrix=True, worldSpace=True)
    print(np.array(matrix).reshape(4,4))

def TestOK3():
    target_matrix = [
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0]
    ]
    m = np.array(target_matrix).ravel().tolist()
    m_matrix = om.MMatrix()
    om.MScriptUtil.createMatrixFromList(target_matrix, m_matrix)
    m_transform_matrix = om.MTransformationMatrix(m_matrix)
    translation = m_transform_matrix.translation(om.MSpace.kWorld)
    rotation = m_transform_matrix.rotation()
    rotation = [om.MAngle(angle, om.MAngle.kRadians).asDegrees() for angle in rotation]
    print(rotation)
#ボーン（ワールド）を取得して置く
#bindposeに移動
#指定ボーンをskinningに追加する
# 指定ボーンを指定されたワールドマトリックス位置(親ジョイント)に作成、指定した親に指定する。
#元のボーン行列にもどす。
#   元のボーンのそれぞれの行列（ワールド）を指定する。
#追加したボーンを指定したワールド位置に移動する。
#-----------------------------------

def raycast(v0, v1, v2, ray_start, ray_dir):
    edge_1 = v1 - v0
    edge_2 = v2 - v0
    ray_offset = ray_start - v0

    det = np.linalg.det(np.array([edge_1, ray_dir, edge_2]))
    eps = 0.000001

    u = np.linalg.det(np.array([ray_dir, edge_2, ray_offset])) / det
    if u >= 0 and u <= 1:
        v = np.linalg.det(np.array([ray_dir, ray_offset, edge_1])) / det
        if v >= 0 and v <= 1 and u + v <= 1:
            t = np.linalg.det(np.array([edge_2, ray_offset, edge_1])) / det
            if t >= 0:
                return t
    return -1

def skinWeightTransfer():
    selection = cmds.ls(selection=True)
    cmds.polyTriangulate(selection)
    mesh = selection[0]
    faces = cmds.polyListComponentConversion(mesh, toFace=True)
    faces = cmds.filterExpand(faces, selectionMask=34, expand=True)

    triangles = []
    for face in faces:
        vertices = cmds.polyListComponentConversion(face, toVertex=True)
        vertices = cmds.filterExpand(vertices, selectionMask=31, expand=True)

        # 三角形の頂点座標を取得
        if len(vertices) == 3:
            triangle = []
            for vertex in vertices:
                pos = cmds.pointPosition(vertex, world=True)
                triangle.append(pos)
            triangles.append(triangle)
    ray_start = np.array([0.0, 10.0, 0.0])
    ray_dir = np.array([0.0, -1.0, 0.0])
    sel = []
    print(len(faces))
    for i in range(len(faces)):
        t = raycast(np.array(triangles[i][0]), np.array(triangles[i][1]), np.array(triangles[i][2]), ray_start, ray_dir)
        if t >= 0:
            print(f"Intersect {ray_start + t * ray_dir}")
            sel.append(faces[i])
#        print(f" {faces[i]} : {triangles[i]}")
    cmds.select(sel)


def ray_cast2(ray_dest):
    ray_direction = om2.MFloatVector(0.0, -1.0, 0.0)
    dest_mfn_mesh = om2.MFnMesh(ray_dest)
    src_position = om2.MFloatPoint(om2.MFloatVector(0.0, 10.0, 0.0))

    hit_point, hit_ray_param, hit_face, hit_triangle, hit_bary1, hit_bary2 = dest_mfn_mesh.closestIntersection(
        src_position,
        ray_direction,
        om2.MSpace.kWorld,
        1000,
        True)
    print(hit_point)
    print(hit_ray_param)
    print(hit_face)
    cmds.select(f"pSphere1.f[{hit_face}]")
    print(hit_triangle)
    print(hit_bary1)
    print(hit_bary2)
    tris = dest_mfn_mesh.getTriangles()
    print(f"tris {tris[0]}")
    print(f"tris {tris[1]}")
    print(dest_mfn_mesh.getFaceVertexIndex(hit_face,0))
    facetri = np.array(tris[0])
    print(len(facetri))
    vert = np.array(tris[1]).reshape(-1,3)
    print(vert)

    triangles = []
    iter = 0
    for fnum in facetri:
        tmp = []
        for i in range(fnum):
            tmp.append(vert[iter + i].tolist())
        triangles.append(tmp)
        iter += 1
    print(triangles)
    for ii, tt in enumerate(triangles):
        print(f"{ii} : {tt}")

    print(f"test {triangles[hit_face][hit_triangle]}")

    # APIで移動した場合は戻すが出来ないので今回のようなスクリプトで実行するだけなら大概はxformで移動
    # 速度を重視する場合はcmdsプラグインとして移動前の状態を保管してctrl+z用の関数として作成する
    # src_mit_vtx.setPosition(om2.MPoint(hit_point), om2.MSpace.kWorld)     # SPI2.0での頂点移動
    # cmds.xform("{}.vtx[{}]".format(str(ray_src), index), t=[v for v in hit_point][0:3], worldSpace=True)

    # 衝突判定の取得情報
    """
    print("source_point:{}".format(src_position))   # 衝突した座標
    print("hit_point:{}".format(hit_point))         # 衝突した座標
    print("hit_ray_param:{}".format(hit_ray_param)) # 衝突点までの距離
    print("hit_face:{}".format(hit_face))           # 衝突したフェースのindex
    print("hit_triangle:{}".format(hit_triangle))   # 衝突した三角形フェースの相対index(ポリゴンが保持しているindex)
    print("hit_bary1:{}".format(hit_bary1))         # 衝突点が三角フェースのどの辺に位置するかどうかの比率
    print("hit_bary2:{}".format(hit_bary2))         # 衝突点が三角フェースのどの辺に位置するかどうかの比率
    """

def apply_raycast():
    sel = om2.MGlobal.getSelectionListByName("pSphere1")
    """:type transform_sel: om2.MSelectionList"""
    sel_dag = sel.getDagPath(0)
    """:type transform_dag: om2.MDagPath"""
    ray_cast2(sel_dag)