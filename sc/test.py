#エッジを選択する

import maya.cmds as cmds
import random

def select_random_edge():
    # アクティブなオブジェクトの選択を取得
    selected_objects = cmds.ls(selection=True, dag=True, long=True)

    if selected_objects:
        # 選択されたオブジェクトのシェイプノードを取得
        shape_node = cmds.listRelatives(selected_objects[0], shapes=True, fullPath=True)
        
        if shape_node and cmds.nodeType(shape_node) == 'mesh':
            # メッシュのエッジを取得
            edges = cmds.polyListComponentConversion(shape_node, toEdge=True)
            edges = cmds.ls(edges, flatten=True)
            print(edges)
            edge_count = cmds.polyEvaluate(shape_node, edge=True)

            if edge_count > 0:
                # ランダムなエッジを選択
                random_edge_index = random.randint(0, edge_count - 1)
                print(edges)
                print(random_edge_index)
                print(len(edges))
                random_edge = edges[random_edge_index]
                cmds.select(random_edge)
            else:
                print("メッシュにエッジがありません。")
        else:
            print("選択されたオブジェクトはポリゴンメッシュではありません。")
    else:
        print("オブジェクトが選択されていません。")

# ランダムなエッジを選択
select_random_edge()


#----------------
edges = cmds.polyListComponentConversion(shape_node, toEdge=True)
edges = cmds.ls(edges, flatten=True) #これで*をつぶせる。

#----------------
#polySelectでelでedgeを選択する。
def select_edge_loop(object_name, edge_index):
    # オブジェクトのエッジを取得
    edges = cmds.polyListComponentConversion(object_name + '.e[' + str(edge_index) + ']', toEdge=True)
    edge_list = cmds.ls(edges, flatten=True)

    # エッジが存在するか確認
    if edge_list:
        # エッジループを選択a
        cmds.select(clear=True)
        cmds.polySelect(object_name, el=edge_index)
    else:
        print("指定されたエッジが見つかりません。")

# オブジェクトの名前とエッジインデックスを指定してエッジループを選択
object_name = "pCylinder1"
edge_index = 1545  # 適切なエッジインデックスに変更してください
select_edge_loop(object_name, edge_index)


#最小距離
import maya.cmds as cmds

def select_edge_loop(object_name, edge_index):
#    cmds.polySelect(object_name, el=edge_index)
    cmds.polySelect(object_name, shortestEdgePath=(10, 100))

object_name = "pCylinder1"
edge_index = 109  # 適切なエッジインデックスに変更してください
select_edge_loop(object_name, edge_index)


## エッジの長さを計算
import maya.cmds as cmds
import math

def calculate_selected_edge_lengths():
    # アクティブなエッジを取得
    selected_edges = cmds.ls(selection=True, flatten=True)
    print(selected_edges)

    # 選択されているエッジが存在するか確認
    if selected_edges:
        total_length = 0.0

        for edge in selected_edges:
            # エッジの両端の頂点座標を取得
            vertex_positions = cmds.polyInfo(edge, edgeToVertex=True)
            print(edge)
            print(vertex_positions)

            if vertex_positions:
                vertex_positions = vertex_positions[0].split(':')[1].strip().split()
                print(vertex_positions)
                vertex_pos = []
                vertex_pos.append(cmds.pointPosition('pCylinder1.vtx[' + vertex_positions[0] + ']', world=True))
                vertex_pos.append(cmds.pointPosition('pCylinder1.vtx[' + vertex_positions[1] + ']', world=True))
                print(vertex_pos)
                # エッジの長さを計算して合計に加算
                edge_length = math.sqrt((vertex_pos[0][0] - vertex_pos[1][0])**2 + (vertex_pos[0][1] - vertex_pos[1][1])**2 + (vertex_pos[0][2] - vertex_pos[1][2])**2)
                total_length += edge_length

        print("Total Selected Edge Length:", total_length)
    else:
        print("エッジが選択されていません。")

# 選択されているエッジの長さを計算
calculate_selected_edge_lengths()


##
import maya.cmds as cmds
import maya.mel as mel

#法線で展開
mel.eval('texNormalProjection 1 1 "" ;')
#展開
#cmds.u3dUnfold(ite=1, p=0, bi=1, tf=1, ms=128, rs=0)