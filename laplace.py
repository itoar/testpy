import maya.api.OpenMaya as om
import maya.cmds as cmds
import scipy.sparse as sp
import numpy as np

def get_selected_mesh_dag():
    """選択中のメッシュの MDagPath を取得する"""
    sel_list = om.MGlobal.getActiveSelectionList()
    if sel_list.length() == 0:
        om.MGlobal.displayError("メッシュが選択されていません")
        return None
    dag = sel_list.getDagPath(0)
    if not dag.node().hasFn(om.MFn.kMesh):
        om.MGlobal.displayError("選択オブジェクトはメッシュではありません")
        return None
    return dag

def compute_cotangent(v1, v2, v3):
    """
    三角形の頂点 v1, v2, v3 において、v3 の角の余接値を計算する。
    :param v1, v2, v3: om.MPoint 型。v3 が角の頂点となる。
    :return: cot(θ) = (v1-v3)・(v2-v3) / ||(v1-v3)×(v2-v3)||
    """
    vec1 = om.MVector(v1 - v3)
    vec2 = om.MVector(v2 - v3)
    dot_val = vec1 * vec2
    cross_val = vec1 ^ vec2  # cross product
    norm_cross = cross_val.length()
    if norm_cross < 1e-6:
        return 0.0
    return dot_val / norm_cross

def compute_sparse_laplacian():
    """
    選択中のメッシュから、cotangent weight に基づく離散ラプラシアンを
    scipy.sparse を用いて計算する。
    sparse 行列 L は、各頂点間の関係を示す n x n 行列となる（n: 頂点数）。
    """
    dag = get_selected_mesh_dag()
    if dag is None:
        return None

    mesh_fn = om.MFnMesh(dag)
    nVerts = mesh_fn.numVertices

    # lil_matrix 形式で n x n のゼロ行列を作成
    L = sp.lil_matrix((nVerts, nVerts), dtype=np.float64)

    # エッジ毎に cotangent weight を計算するために MItMeshEdge を使用
    edge_it = om.MItMeshEdge(dag)
    while not edge_it.isDone():
        vtxId1 = edge_it.vertexId(0)
        vtxId2 = edge_it.vertexId(1)

        # エッジに隣接する面（内部エッジなら通常2枚、境界エッジなら1枚）
        faceIds = edge_it.getConnectedFaces()
        cot_sum = 0.0
        # 隣接面ごとに cotangent 値を合計
        for faceId in faceIds:
            faceVerts = mesh_fn.getPolygonVertices(faceId)
            thirdVerts = [vid for vid in faceVerts if vid not in [vtxId1, vtxId2]]
            if len(thirdVerts) != 1:
                continue
            vtxId3 = thirdVerts[0]
            pos1 = mesh_fn.getPoint(vtxId1, om.MSpace.kWorld)
            pos2 = mesh_fn.getPoint(vtxId2, om.MSpace.kWorld)
            pos3 = mesh_fn.getPoint(vtxId3, om.MSpace.kWorld)
            cot = compute_cotangent(pos1, pos2, pos3)
            cot_sum += cot

        # 内部エッジなら両側面、境界エッジなら1側面のみとなる
        weight = 0.5 * cot_sum

        # オフダイアゴナル成分: L[i,j] = -weight
        L[vtxId1, vtxId2] = -weight
        L[vtxId2, vtxId1] = -weight
        # ダイアゴナル成分は、その頂点に接するエッジの weight の和
        L[vtxId1, vtxId1] += weight
        L[vtxId2, vtxId2] += weight

        edge_it.next()

    # 必要に応じて CSR 形式に変換可能（計算や最適化時に有利）
    L_csr = L.tocsr()
    return L_csr

if __name__ == "__main__":
    laplacian_sparse = compute_sparse_laplacian()
    if laplacian_sparse is not None:
        om.MGlobal.displayInfo("scipy.sparse を用いた離散ラプラシアンの計算結果:")
        print(laplacian_sparse)


/////////////////////////////////////////////////

import maya.api.OpenMaya as om
import maya.cmds as cmds
import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg import lsqr

def get_selected_mesh_dag():
    """選択中のメッシュの MDagPath を取得する"""
    sel_list = om.MGlobal.getActiveSelectionList()
    if sel_list.length() == 0:
        om.MGlobal.displayError("メッシュが選択されていません")
        return None
    dag = sel_list.getDagPath(0)
    if not dag.node().hasFn(om.MFn.kMesh):
        om.MGlobal.displayError("選択オブジェクトはメッシュではありません")
        return None
    return dag

def compute_cotangent(v1, v2, v3):
    """
    三角形の頂点 v1, v2, v3 において、v3 の角の余接値を計算する。
    :param v1, v2, v3: om.MPoint 型（v3 が角の頂点）
    :return: cot(θ) = ((v1-v3)・(v2-v3)) / ||(v1-v3)×(v2-v3)||
    """
    vec1 = om.MVector(v1 - v3)
    vec2 = om.MVector(v2 - v3)
    dot_val = vec1 * vec2
    cross_val = vec1 ^ vec2  # 外積
    norm_cross = cross_val.length()
    if norm_cross < 1e-6:
        return 0.0
    return dot_val / norm_cross

def compute_sparse_laplacian():
    """
    選択中のメッシュから、cotangent weight に基づく離散ラプラシアン L を
    scipy.sparse を用いて計算する。L は n×n の疎行列（CSR形式）となる。
    """
    dag = get_selected_mesh_dag()
    if dag is None:
        return None
    mesh_fn = om.MFnMesh(dag)
    nVerts = mesh_fn.numVertices

    # lil_matrix は逐次更新に適している
    L = sp.lil_matrix((nVerts, nVerts), dtype=np.float64)

    # エッジ毎に cotangent weight を計算（内部エッジなら2枚、境界エッジなら1枚）
    edge_it = om.MItMeshEdge(dag)
    while not edge_it.isDone():
        vtxId1 = edge_it.vertexId(0)
        vtxId2 = edge_it.vertexId(1)

        faceIds = edge_it.getConnectedFaces()
        cot_sum = 0.0
        for faceId in faceIds:
            faceVerts = mesh_fn.getPolygonVertices(faceId)
            # エッジ上の2頂点以外の頂点を求める
            thirdVerts = [vid for vid in faceVerts if vid not in [vtxId1, vtxId2]]
            if len(thirdVerts) != 1:
                continue
            vtxId3 = thirdVerts[0]
            pos1 = mesh_fn.getPoint(vtxId1, om.MSpace.kWorld)
            pos2 = mesh_fn.getPoint(vtxId2, om.MSpace.kWorld)
            pos3 = mesh_fn.getPoint(vtxId3, om.MSpace.kWorld)
            cot = compute_cotangent(pos1, pos2, pos3)
            cot_sum += cot

        # エッジの weight = 0.5*(cotα + cotβ)（境界エッジなら1面分のみ）
        weight = 0.5 * cot_sum

        # オフダイアゴナル成分
        L[vtxId1, vtxId2] = -weight
        L[vtxId2, vtxId1] = -weight
        # ダイアゴナル成分：隣接するエッジの weight の和
        L[vtxId1, vtxId1] += weight
        L[vtxId2, vtxId2] += weight

        edge_it.next()
    return L.tocsr()

def get_constraints(nVerts):
    """
    サンプルとして制約情報を定義する。
    実際はユーザー入力に基づいて、各制約は
    { "vertices": [v1, v2, v3], "bary": [α, β, γ], "value": f, "weight": w }
    の形式で与えられると仮定する。
    """
    constraints = []
    # 制約例1: 三角形の頂点 [0, 1, 2] で、バリセンター座標 [0.3, 0.4, 0.3]、制約値 +1
    if nVerts >= 3:
        constraints.append({
            "vertices": [0, 1, 2],
            "bary": [0.3, 0.4, 0.3],
            "value": 1.0,
            "weight": 1.0
        })
    # 制約例2: 三角形の頂点 [3, 4, 5] で、バリセンター座標 [0.2, 0.5, 0.3]、制約値 -1
    if nVerts >= 6:
        constraints.append({
            "vertices": [3, 4, 5],
            "bary": [0.2, 0.5, 0.3],
            "value": -1.0,
            "weight": 1.0
        })
    return constraints

def construct_constraint_matrix(nVerts, constraints):
    """
    制約リストから、C 行列（サイズ：N_C×nVerts）と右辺 b_C を作成する。
    各制約行 i では、対象三角形の頂点 v1, v2, v3 に対して、
    C[i, v1] = weight * α, C[i, v2] = weight * β, C[i, v3] = weight * γ
    とし、b_C[i] = weight * (制約値) とする。
    """
    N_C = len(constraints)
    C = sp.lil_matrix((N_C, nVerts), dtype=np.float64)
    bC = np.zeros(N_C, dtype=np.float64)
    for i, cons in enumerate(constraints):
        vertices = cons["vertices"]
        bary = cons["bary"]
        value = cons["value"]
        weight = cons["weight"]
        bC[i] = weight * value
        for j, v in enumerate(vertices):
            if v < nVerts:
                C[i, v] = weight * bary[j]
    return C.tocsr(), bC

def solve_expanded_system():
    """
    ・まず疎行列 L を計算する。
    ・次に、ユーザー制約に基づく C 行列と右辺 b_C を作成する。
    ・拡大行列 A = [L; C] と b = [0,...,0, b_C] を構築し、
      lsqr を用いて最小二乗解 u を求める。
    """
    # 1. Laplacian の計算
    L = compute_sparse_laplacian()
    if L is None:
        return None
    nVerts = L.shape[0]

    # 2. 制約情報の取得（実際はユーザー入力から取得する）
    constraints = get_constraints(nVerts)
    C, bC = construct_constraint_matrix(nVerts, constraints)

    # 3. 拡大行列 A の構築：A = [L; C]
    A = sp.vstack([L, C])
    # 4. 右辺ベクトル b の構築：上部 nVerts 成分は 0、下部は bC
    b = np.concatenate([np.zeros(nVerts), bC])

    # 5. lsqr を用いて最小二乗解を求める
    sol = lsqr(A, b)
    u = sol[0]
    return u

if __name__ == "__main__":
    u_solution = solve_expanded_system()
    if u_solution is not None:
        om.MGlobal.displayInfo("拡大線形システムの最小二乗解が求まりました。")
        # 得られた u_solution は各頂点でのハーモニックフィールドの値
        print("ハーモニックフィールド u:")
        print(u_solution)


//////////

import maya.api.OpenMaya as om
import maya.api.OpenMayaUI as omui
from maya.OpenMayaRender import MHardwareRenderer

class ConstraintContext(omui.MPxContext):
    def __init__(self):
        super(ConstraintContext, self).__init__()
        self.setTitleString("Constraint Draw Tool")
        self.strokePoints = []  # MPoint のリスト

    def doPressEvent(self, event):
        pos = event.position  # スクリーン座標 (x, y)
        hitData = self.getIntersection(pos)
        if hitData:
            # hitData["point"] は MPoint 型
            self.strokePoints = [hitData["point"]]
        else:
            self.strokePoints = []
        self.refresh()

    def doDragEvent(self, event):
        pos = event.position
        hitData = self.getIntersection(pos)
        if hitData:
            self.strokePoints.append(hitData["point"])
        self.refresh()

    def doReleaseEvent(self, event):
        # ストロークが終了したら、必要な処理を実行（例えば制約生成など）
        self.refresh()

    def getIntersection(self, pos):
        """
        画面上の位置 pos (x, y) から、現在選択中のメッシュとの交点を計算して返す。
        戻り値は {"point": MPoint, "faceId": int, ...} の辞書など。
        詳細は前述の実装を参照。
        """
        # ここでは簡単な例として None を返す
        # 実際は M3dView.viewToWorld() などを用いて交点を求める
        return None

    def doDraw(self, view):
        """
        doDraw() メソッドをオーバーライドして、収集した strokePoints を
        ビューポート上に線として描画する。
        """
        if not self.strokePoints:
            return
        
        # OpenGL 描画のためのハードウェアレンダラーを取得
        renderer = MHardwareRenderer.theRenderer()
        glFT = renderer.glFunctionTable()
        
        view.beginGL()
        # 線の色を赤に設定
        glFT.glColor3f(1.0, 0.0, 0.0)
        glFT.glLineWidth(2.0)
        glFT.glBegin(omui.MGL_LINE_STRIP)
        for pt in self.strokePoints:
            glFT.glVertex3f(pt.x, pt.y, pt.z)
        glFT.glEnd()
        view.endGL()

# コンテキストの登録と切り替え
def initializeConstraintTool():
    ctxName = "constraintContext"
    # 既存のコンテキストがあればそれを使うか、新規に作成する
    if cmds.contextInfo(ctxName, exists=True):
        cmds.setToolTo(ctxName)
    else:
        # 新しいコンテキストのインスタンスを作成し、Maya に登録
        pluginCtx = ConstraintContext()
        omui.MPxContext.registerContext("ConstraintContext", ConstraintContext)
        cmds.setToolTo("ConstraintContext")

# ツールの起動例
initializeConstraintTool()
