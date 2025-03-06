import maya.api.OpenMaya as om
import maya.api.OpenMayaUI as omui
import maya.cmds as cmds

# -------------------------------------------
# カスタムコンテキストの定義
# -------------------------------------------
class SphereOnMeshContext(omui.MPxContext):
    def __init__(self):
        super(SphereOnMeshContext, self).__init__()
        self.setTitleString("Sphere On Mesh Tool")
        # アイコン（任意）の設定。適当な xpm ファイルを指定できます。
        self.setImage("move.xpm", omui.MPxContext.kImage1)

    def toolOnSetup(self, *args):
        # ツール開始時に選択状態をクリア（または適宜初期化処理）
        cmds.select(clear=True)
        om.MGlobal.displayInfo("Sphere On Mesh Tool を起動しました。メッシュ上をクリックしてください。")

    def doPress(self, event):
        """
        ユーザーがビューポート上でクリックした際に呼ばれる。
        クリック位置からレイキャストを行い、選択済みメッシュとの交差点を求め、球を生成する。
        """
        # 現在の 3D ビューを取得
        view = omui.M3dView.active3dView()
        pos = event.getPosition()  # (x, y) のタプルが返る
        x, y = pos

        # クリック位置からカメラに沿ったレイの始点と終点を取得
        try:
            nearPoint, farPoint = view.viewToWorld(x, y)
        except Exception as e:
            om.MGlobal.displayError("viewToWorld の呼び出しに失敗: " + str(e))
            return

        nearPt = om.MPoint(nearPoint)
        farPt  = om.MPoint(farPoint)
        # レイの方向ベクトルを算出（正規化）
        rayDir = (farPt - nearPt).normal()

        # ※ここでは、対象となるメッシュは「選択済みの最初のメッシュ」とします。
        selList = cmds.ls(selection=True, dag=True, type="mesh")
        if not selList:
            om.MGlobal.displayWarning("対象のメッシュを選択してください。")
            return

        # 選択メッシュの DAG パスを取得
        meshDagPath = self.getDagPathFromName(selList[0])
        if meshDagPath is None:
            om.MGlobal.displayWarning("選択メッシュの DAG パスの取得に失敗しました。")
            return

        # MFnMesh を使ってレイとの交差判定を行う
        meshFn = om.MFnMesh(meshDagPath)
        try:
            # closestIntersection は、交差情報があればタプル、なければ空タプルを返す
            interResult = meshFn.closestIntersection(
                om.MFloatPoint(nearPt),
                om.MFloatVector(rayDir),
                None,     # 面のフィルタ（使わない場合は None）
                None,     # UV 情報（使わない場合は None）
                False,    # backfaceCull: False で両面
                om.MSpace.kWorld
            )
        except Exception as e:
            om.MGlobal.displayError("交差判定に失敗: " + str(e))
            return

        if not interResult:
            om.MGlobal.displayWarning("クリック位置と交差するメッシュの面が見つかりませんでした。")
            return

        # 交差情報を展開
        hitPoint, hitParam, hitFace, hitTriangle, hitBarycentric = interResult

        # 面の法線を取得（交差した面番号 hitFace を指定）
        normal = meshFn.getPolygonNormal(hitFace, om.MSpace.kWorld)
        normalVec = normal.normal()

        # 球の半径と、交差点から法線方向にオフセットする量（半径分ずらす例）
        sphereRadius = 0.5
        spherePos = hitPoint + normalVec * sphereRadius

        # Maya の cmds で球を生成し、ワールド座標に配置
        sphereName = cmds.polySphere(r=sphereRadius, name="generatedSphere")[0]
        cmds.xform(sphereName, ws=True, t=(spherePos.x, spherePos.y, spherePos.z))
        om.MGlobal.displayInfo("Sphere 作成: ({:.2f}, {:.2f}, {:.2f})".format(spherePos.x, spherePos.y, spherePos.z))

    def getDagPathFromName(self, nodeName):
        """
        指定されたノード名から DAG パスを取得する補助関数
        """
        selectionList = om.MSelectionList()
        try:
            selectionList.add(nodeName)
            dagPath = selectionList.getDagPath(0)
            return dagPath
        except Exception as e:
            om.MGlobal.displayError("DAG パスの取得に失敗: " + str(e))
            return None

# -------------------------------------------
# コンテキストコマンドの定義（ツールとして登録するため）
# -------------------------------------------
class SphereOnMeshContextCmd(omui.MPxContextCommand):
    def __init__(self):
        super(SphereOnMeshContextCmd, self).__init__()
        self._context = None

    def makeObj(self):
        self._context = SphereOnMeshContext()
        return self._context

def cmdCreator():
    return SphereOnMeshContextCmd()

# -------------------------------------------
# プラグインの初期化／終了関数
# -------------------------------------------
def initializePlugin(mobject):
    plugin = om.MFnPlugin(mobject, "YourName", "1.0", "Any")
    try:
        # "sphereOnMeshContextCmd" という名前でコンテキストコマンドを登録
        plugin.registerContextCommand("sphereOnMeshContextCmd", cmdCreator)
    except Exception as e:
        om.MGlobal.displayError("sphereOnMeshContextCmd の登録に失敗: " + str(e))

def uninitializePlugin(mobject):
    plugin = om.MFnPlugin(mobject)
    try:
        plugin.deregisterContextCommand("sphereOnMeshContextCmd")
    except Exception as e:
        om.MGlobal.displayError("sphereOnMeshContextCmd の解除に失敗: " + str(e))


import maya.cmds as cmds
cmds.loadPlugin("C:/path/to/sphereOnMeshTool.py")

cmds.setToolTo("sphereOnMeshContextCmd")