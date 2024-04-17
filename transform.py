def conputeRigidBodyTransform(A, B, enable_scale = False):
    # A,B は データ次元 x ポイント数の行列
    dim, point_num = A.shape
    center_A = A.mean(axis=1)
    center_B = B.mean(axis=1)
    A_normalized = A - np.tile(center_A, (point_num, 1)).T
    B_normalized = B - np.tile(center_B, (point_num, 1)).T

    sx = np.mean(np.sum(A_normalized**2, axis=0))
    sy = np.mean(np.sum(B_normalized**2, axis=0))
    AB = (B_normalized @ A_normalized.T) / point_num
    rank = np.linalg.matrix_rank(AB)
    if rank < dim - 2:
        return 

    # AB^tに対するSVD
    U, D, Vt = np.linalg.svd(AB, full_matrices=True, compute_uv=True)
    V = Vt.T

    S = np.eye(dim)
    # AB^t determinantで場合分ける
    if np.linalg.det(AB) < 0:
        S[dim - 1, dim - 1] = -1

    R = U @ S @ V.T

    if enable_scale:
        s = np.trace(np.diag(D) @ S) / sx
        t = center_B - s * (R @ center_A)
    else:
        t = center_B - (R @ center_A)
        s = 1.0
    return R, s, t



def create_joint_with_matrix(matrix, parent_joint=None):
    # 新しいジョイントを作成
    new_joint = cmds.joint()

    # Maya APIを使用して新しいジョイントのワールドマトリックスを設定
    m_matrix = om.MMatrix(matrix)
    m_transform = om.MTransformationMatrix(m_matrix)
    m_translation = m_transform.translation(om.MSpace.kWorld)
    m_rotation = m_transform.rotation(asQuaternion=True)

    # 新しいジョイントのトランスフォームノードを取得
    joint_transform = cmds.listRelatives(new_joint, parent=True, path=True)[0]

    # ワールドマトリックスを新しいジョイントのトランスフォームノードに適用
    cmds.xform(joint_transform, translation=m_translation, rotation=m_rotation, worldSpace=True)

    # 親ジョイントが指定されている場合は、新しいジョイントをその子に設定
    if parent_joint:
        cmds.parent(new_joint, parent_joint)

    return new_joint

# マトリックスを作成（例：単位行列）
matrix = om.MMatrix()
# 親ジョイントの名前を指定（必要な場合）
parent_joint = "parent_joint_name"

# 新しいジョイントを作成し、親ジョイントに設定
created_joint = create_joint_with_matrix(matrix, parent_joint)


def setJointRadius(name, joint_rad):
    str = name + ".radius"
    cmds.setAttr(str, joint_rad)

def addSkinCluster(skinname, jointname):
    str = "skinCluster -e -dr 4 -lw true -wt 0 -ai " + joint4 + " " + skinname + ";"
    mel.eval(str)

def getJointTransform(joint_name):
    t = cmds.xform(joint_name, query=True, translation=True, worldSpace=True)
    r = cmds.xform(joint_name, query=True, rotation=True, worldSpace=True)
    s = cmds.xform(joint_name, query=True, scale=True, worldSpace=True)
    return t, r, s

def setJointTransform(joint_name, t, r):
    cmds.xform(joint_name, translation=t, rotation=r, worldSpace=True)

def RenameJoint(joint_name, name):
    cmds.rename(joint_name, name)