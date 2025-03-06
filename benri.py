import maya.api.OpenMaya as om

def select_vertices(object_name, vertex_indices):
    """
    指定したオブジェクトの頂点インデックスリストから頂点を選択する関数。
    
    :param object_name: メッシュオブジェクトの名前（例："pCube1"）
    :param vertex_indices: 頂点インデックスのリスト（例：[0, 2, 4, 6]）
    """
    # オブジェクトのDagPathを取得するため、MSelectionListを作成して追加
    sel_list = om.MSelectionList()
    sel_list.add(object_name)
    dag_path = sel_list.getDagPath(0)
    
    # 頂点コンポーネントを作成
    mfnComp = om.MFnSingleIndexedComponent()
    vertex_comp = mfnComp.create(om.MFn.kMeshVertComponent)
    
    # 頂点インデックスをコンポーネントに追加
    for index in vertex_indices:
        mfnComp.addElement(index)
    
    # 新しい選択リストを作成し、DagPathと頂点コンポーネントを追加
    new_sel_list = om.MSelectionList()
    new_sel_list.add(dag_path, vertex_comp)
    
    # 選択リストをアクティブに設定
    om.MGlobal.setActiveSelectionList(new_sel_list)
    print("Selected vertices on '{}': {}".format(object_name, vertex_indices))

# 使用例
if __name__ == "__main__":
    # シーン内のメッシュオブジェクト名（適宜変更してください）
    object_name = "pCube1"
    
    # 選択したい頂点のインデックスリスト
    vertex_indices = [0, 2, 4, 6]
    
    select_vertices(object_name, vertex_indices)






・・
import maya.api.OpenMaya as om

def select_edges_from_vertices(object_name, vertex_indices):
    """
    指定されたメッシュオブジェクトの頂点リストに基づき、
    両端がその頂点に属するエッジを抽出し選択する関数。
    
    :param object_name: メッシュオブジェクトの名前（例："pCube1"）
    :param vertex_indices: 頂点インデックスのリスト（例：[0, 2, 4, 6]）
    """
    # 1. オブジェクトのDagPathを取得
    sel_list = om.MSelectionList()
    sel_list.add(object_name)
    dag_path = sel_list.getDagPath(0)
    
    # 2. 頂点リストをsetに変換（判定を高速化）
    vertex_set = set(vertex_indices)
    
    # 3. エッジのリストを作成するため、MItMeshEdgeを利用して全エッジをチェック
    selected_edge_indices = []
    edge_iter = om.MItMeshEdge(dag_path)
    while not edge_iter.isDone():
        # 現在のエッジの両端の頂点インデックスを取得（タプルで返る）
        verts = edge_iter.getConnectedVertices()
        # 両端の頂点が指定された頂点リストに含まれる場合、そのエッジを選択対象に追加
        if all(v in vertex_set for v in verts):
            selected_edge_indices.append(edge_iter.index())
        edge_iter.next()
    
    # 4. エッジコンポーネントを作成して、抽出したエッジインデックスを追加
    comp_fn = om.MFnSingleIndexedComponent()
    edge_comp = comp_fn.create(om.MFn.kMeshEdgeComponent)
    for edge_index in selected_edge_indices:
        comp_fn.addElement(edge_index)
    
    # 5. 新たな選択リストにDagPathとエッジコンポーネントを登録し、選択状態に設定
    new_sel_list = om.MSelectionList()
    new_sel_list.add(dag_path, edge_comp)
    om.MGlobal.setActiveSelectionList(new_sel_list)
    
    print("Selected edges on '{}': {}".format(object_name, selected_edge_indices))

# 使用例
if __name__ == "__main__":
    # シーン内のメッシュオブジェクト名（適宜変更してください）
    object_name = "pCube1"
    # 選択したい頂点のインデックスリスト（例：頂点 0, 1, 2, 3 を指定）
    vertex_indices = [0, 1, 2, 3]
    select_edges_from_vertices(object_name, vertex_indices)
