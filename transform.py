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