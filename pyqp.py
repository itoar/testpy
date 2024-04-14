import time
from concurrent.futures import ProcessPoolExecutor
import os
import time

import osqp
import numpy as np
from scipy import sparse

def computeBasisMat(Mat, JNum, SNum, vertex):
    basis_mat = []
    for joint_idx in range(JNum):
        basis_tmp = []
        for sample_idx in range(SNum):
            basis = Mat[sample_idx][joint_idx] @ vertex
            basis_tmp.extend(basis[:3])
        basis_mat.append(basis_tmp)
    basis_mat = np.array(basis_mat)
    return basis_mat

def computeQPMatrix(basis, target):
    P_mat = basis@basis.T
    Q_mat = basis @ target
    return P_mat, Q_mat

def fn(v_idx):
    return v_idx*2



if __name__ == '__main__':
    rng = np.random.default_rng()

    JointNum: int = 100
    TargetNum: int = 1
    Influence: int = 4
    VertexNum : int = 100

    BindVertices = []
    for vert in range(VertexNum):
        BindVertices.append([vert, vert, vert, 1])


    SkinMatList: list = []

    weight = rng.random(JointNum)

    TargetVert: list = []
    for vert_idx in range(VertexNum):
        tmp = []
        for sample_idx in range(TargetNum):
            tmp.extend([vert_idx*(sample_idx + 1), vert_idx*(sample_idx + 1), vert_idx*(sample_idx + 1)])
        TargetVert.append(tmp)
    TargetVert = np.array(TargetVert)
        
    for target_idx in range(TargetNum):
        SkinMat: list = []
        for joint_num in range(JointNum):
            rn = rng.random((4,4))
            SkinMat.append(rn)
        SkinMatList.append(SkinMat)

    MaskList = []
    MASK_NUM = 4
    for vert_idx in range(VertexNum):
        tmp = []
        for joint_idx in range(JointNum):
            if joint_idx > MASK_NUM:
                tmp.extend([0])
            else:
                tmp.extend([1])
        MaskList.append(tmp)

    MaskedBasisLists = []
    for v_idx, v in enumerate(BindVertices):     
        mask = np.array(MaskList[v_idx])
        basis = computeBasisMat(SkinMatList, JointNum, TargetNum, v) # ジョイント数 x (3*ターゲット数)
        masked_basis = basis[mask > 0][:]
        MaskedBasisLists.append(masked_basis)

    # --計算前に計算できるMat or Vec 
    # BindVertices      バインド時の頂点座標リスト [頂点数 x 4] の行列 
    # TargetVert        ターゲットの頂点座標リスト [ 頂点数 x (3*ターゲット数)) ]の行列
    # SkinMatList       スキニング行列 [ ターゲット数 x ジョイント数 x 4 x 4 ]のリスト
    # InfluenceMask     マスク行列 [ 頂点数 x ジョイント数]
    # MaskedBasisLists  マスクされた規定マトリックス [ ジョイント数 x (3*ターゲット数)) の行列]

    def optimize(v_idx):
        masked_basis = MaskedBasisLists[v_idx]
        weight_num = masked_basis.shape[0]
        P_mat, Q_mat = computeQPMatrix(masked_basis, TargetVert[v_idx])
        A_mat = np.concatenate([np.identity(weight_num), np.full((1, weight_num), 1)])

        
        P = sparse.csc_matrix(P_mat)
        q = np.array(Q_mat)
        A = sparse.csc_matrix(A_mat)
        l = np.full((weight_num+1), 0.0)
        l[weight_num] = 1.0    
        u = np.full((weight_num+1), 1.0)
        
        # Create an OSQP object
        prob = osqp.OSQP()
        
        # Setup workspace and change alpha parameter
        prob.setup(P, q, A, l, u, alpha=1.0, verbose = False, adaptive_rho = False)
        
        # Solve problem
        res = prob.solve()



    start = time.time()
    # data = list(range(len(BindVertices)))
    # with ProcessPoolExecutor() as executor:  # -----(2)
    #     results = executor.map(optimize, data)    


    for v_idx in range(len(BindVertices)):
        optimize(v_idx)
    end = time.time()
    # for result in results:
    #     print(result)
    print(f"TIME {end - start}")