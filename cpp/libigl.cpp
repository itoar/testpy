#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>

// グラフ用定数
constexpr int cEdgeMax = 1000;
constexpr int cNodeMax = 1000;

struct Vector3
{
    float mPosx;
    float mPosy;
    float mPosz;
    void print() const
    {
        std::cout << mPosx << " " << mPosy << " " << mPosz << std::endl;               
    }
};

struct Edge
{
    int mNodeA = 0;
    int mNodeB = 0;
    Edge(int node_a, int node_b) : mNodeA(node_a), mNodeB(node_b) {}
    virtual ~Edge() {}
    virtual void print() const
    {
        std::cout << "Edge: " << mNodeA << " " << mNodeB << std::endl;       
    }
};

struct Node
{
    Vector3 mPos;
    Node(Vector3 pos) : mPos(pos) {}
    virtual ~Node() {}
    virtual void print() const
    {
        mPos.print();
    }
};

struct CustomEdge : public Edge
{
    float mDamage = 0.0f;
    CustomEdge(int node_a, int node_b, float damage) 
        : Edge(node_a, node_b), mDamage(damage) {}
    virtual ~CustomEdge() {}
    virtual void print() const override
    {
        std::cout << "CustomEdge: " << mNodeA << " " << mNodeB << " " << mDamage << std::endl;       
    }
};

struct CustomNode : public Node
{
    bool mFixed = false;
    CustomNode(Vector3 pos) 
        : Node(pos), mFixed(false) {}
    virtual ~CustomNode() {}
    virtual void print() const override
    {
        std::cout << "CustomNode (fixed = " << mFixed << "): ";
        mPos.print();
    }    
};

struct Graph
{
    Edge** mEdgeList;
    int mEdgeNum = 0;
    Node** mNodeList;
    int mNodeNum = 0;
    Graph()
    {
        mEdgeList = new Edge*[cEdgeMax];
        mNodeList = new Node*[cNodeMax];
    }
    ~Graph()
    {
        for (int i = 0; i < mEdgeNum; i++)
            delete mEdgeList[i];
        delete [] mEdgeList;
        for (int i = 0; i < mNodeNum; i++)
            delete mNodeList[i];
        delete [] mNodeList;
    }
    void AddNode(Vector3 pos)
    {
        if (mNodeNum < cNodeMax) {
            Node* node = new Node(pos);
            mNodeList[mNodeNum++] = node;
        } else {
            std::cerr << "Node capacity exceeded!" << std::endl;
        }
    }
    void AddCustomNode(Vector3 pos)
    {
        if (mNodeNum < cNodeMax) {
            Node* node = new CustomNode(pos);
            mNodeList[mNodeNum++] = node;
        } else {
            std::cerr << "Node capacity exceeded!" << std::endl;
        }
    }
    void AddEdge(int node_a, int node_b)
    {
        if (mEdgeNum < cEdgeMax) {
            Edge* edge = new Edge(node_a, node_b);
            mEdgeList[mEdgeNum++] = edge;
        } else {
            std::cerr << "Edge capacity exceeded!" << std::endl;
        }
    }
    void AddCustomEdge(int node_a, int node_b, float damage)
    {
        if (mEdgeNum < cEdgeMax) {
            Edge* edge = new CustomEdge(node_a, node_b, damage);
            mEdgeList[mEdgeNum++] = edge;
        } else {
            std::cerr << "Edge capacity exceeded!" << std::endl;
        }
    }
    void dump() const
    {
        std::cout << "->Edge" << std::endl;
        for (int i = 0; i < mEdgeNum; i++)
            mEdgeList[i]->print();
        std::cout << "->Node" << std::endl;
        for (int i = 0; i < mNodeNum; i++)
            mNodeList[i]->print();
    }
};

//
// 立方格子（キュービックグリッド）を生成する関数
// numVertical: 縦方向（X個）、numHorizontal: 横方向（Z個）、numHeight: 高さ方向（Y個）
// 座標系：x軸→横、y軸→高さ、z軸→縦
//
void GenerateCubicGrid(Graph& graph, int numVertical, int numHorizontal, int numHeight)
{
    for (int y = 0; y < numHeight; y++) {
        for (int z = 0; z < numVertical; z++) {
            for (int x = 0; x < numHorizontal; x++) {
                float xPos = static_cast<float>(x);
                float yPos = static_cast<float>(y);
                float zPos = static_cast<float>(z);
                // 固定条件を後で与えるために CustomNode を生成
                graph.AddCustomNode(Vector3{ xPos, yPos, zPos });
            }
        }
    }
    // ノードの追加順は (y, z, x) の順．
    auto index = [=](int x, int z, int y) -> int {
        return y * (numVertical * numHorizontal) + z * numHorizontal + x;
    };
    // 各軸の隣接ノードにエッジを追加（重複なく）
    for (int y = 0; y < numHeight; y++) {
        for (int z = 0; z < numVertical; z++) {
            for (int x = 0; x < numHorizontal; x++) {
                int current = index(x, z, y);
                if (x < numHorizontal - 1) {
                    int neighbor = index(x + 1, z, y);
                    graph.AddEdge(current, neighbor);
                }
                if (z < numVertical - 1) {
                    int neighbor = index(x, z + 1, y);
                    graph.AddEdge(current, neighbor);
                }
                if (y < numHeight - 1) {
                    int neighbor = index(x, z, y + 1);
                    graph.AddEdge(current, neighbor);
                }
            }
        }
    }
}


void GenerateTableGraph(Graph& graph, int numTopVertical, int numTopHorizontal, int numLegHeight)
{
    // 天板のノードを追加（y = numLegHeight）
    for (int z = 0; z < numTopVertical; z++) {
        for (int x = 0; x < numTopHorizontal; x++) {
            float xPos = static_cast<float>(x);
            float yPos = static_cast<float>(numLegHeight); // 天板の高さ
            float zPos = static_cast<float>(z);
            graph.AddCustomNode(Vector3{ xPos, yPos, zPos });
        }
    }

    // 脚のノードを追加（四隅）
    //int legOffsets[4][2] = {
    //    {0, 0}, {0, numTopVertical - 1}, 
    //    {numTopHorizontal - 1, 0}, {numTopHorizontal - 1, numTopVertical - 1}
    //};

    int legOffsets[3][2] = {
        {0, 0}, {0, numTopVertical - 1}, 
        {numTopHorizontal - 1, 0}
    };


    for (int i = 0; i < 3; i++) {
        int x = legOffsets[i][0];
        int z = legOffsets[i][1];

        for (int y = 0; y < numLegHeight; y++) {
            graph.AddCustomNode(Vector3{ static_cast<float>(x), static_cast<float>(y), static_cast<float>(z) });
        }
    }

    // インデックス関数
    auto topIndex = [=](int x, int z) -> int {
        return z * numTopHorizontal + x;
    };
    auto legIndex = [=](int leg, int y) -> int {
        return numTopVertical * numTopHorizontal + leg * numLegHeight + y;
    };

    // 天板のエッジを追加（2Dグリッド構造）
    for (int z = 0; z < numTopVertical; z++) {
        for (int x = 0; x < numTopHorizontal; x++) {
            int current = topIndex(x, z);
            if (x < numTopHorizontal - 1) {
                graph.AddEdge(current, topIndex(x + 1, z));
            }
            if (z < numTopVertical - 1) {
                graph.AddEdge(current, topIndex(x, z + 1));
            }
        }
    }

    // 脚のエッジを追加（各脚を垂直につなぐ）
    for (int i = 0; i < 3; i++) {
        int x = legOffsets[i][0];
        int z = legOffsets[i][1];
        int topNode = topIndex(x, z);

        // 脚のノードは垂直に並んでいるので、y軸方向で接続します
        for (int y = 0; y < numLegHeight; y++) {
            int current = legIndex(i, y);

            if (y == numLegHeight - 1) {
                // 脚の最上部（y == numLegHeight - 1）と接続
                graph.AddEdge(current, topNode);
            } else {
                // 下方向に接続（脚の垂直方向接続）
                graph.AddEdge(current, current + 1);
            }
        }
    }
}


//
// igl 用に、グラフ情報を Eigen 行列に変換して表示する関数
//
void entryGraph4show(Graph& graph, igl::opengl::glfw::Viewer& viewer, Vector3 col)
{
    Eigen::MatrixXd Node4Show(graph.mNodeNum, 3);
    Eigen::MatrixXd EdgeA4Show(graph.mEdgeNum, 3);
    Eigen::MatrixXd EdgeB4Show(graph.mEdgeNum, 3);
    for (int i = 0; i < graph.mNodeNum; i++) {
        Node4Show(i, 0) = graph.mNodeList[i]->mPos.mPosx;
        Node4Show(i, 1) = graph.mNodeList[i]->mPos.mPosy;
        Node4Show(i, 2) = graph.mNodeList[i]->mPos.mPosz;
    }
    for (int i = 0; i < graph.mEdgeNum; i++) {
        EdgeA4Show(i, 0) = graph.mNodeList[ graph.mEdgeList[i]->mNodeA ]->mPos.mPosx;
        EdgeA4Show(i, 1) = graph.mNodeList[ graph.mEdgeList[i]->mNodeA ]->mPos.mPosy;
        EdgeA4Show(i, 2) = graph.mNodeList[ graph.mEdgeList[i]->mNodeA ]->mPos.mPosz;
        EdgeB4Show(i, 0) = graph.mNodeList[ graph.mEdgeList[i]->mNodeB ]->mPos.mPosx;
        EdgeB4Show(i, 1) = graph.mNodeList[ graph.mEdgeList[i]->mNodeB ]->mPos.mPosy;
        EdgeB4Show(i, 2) = graph.mNodeList[ graph.mEdgeList[i]->mNodeB ]->mPos.mPosz;
    }
    viewer.data().point_size = 20;
    viewer.data().add_points(Node4Show, Eigen::RowVector3d(col.mPosx, col.mPosy, col.mPosz));
    // ここでは set_edges() を利用してエッジを表示（set_edges(V, E, C)）
    Eigen::MatrixXi E(graph.mEdgeNum, 2);
    for (int i = 0; i < graph.mEdgeNum; i++){
        E(i,0) = graph.mEdgeList[i]->mNodeA;
        E(i,1) = graph.mEdgeList[i]->mNodeB;
    }
    // ここではデフォルト色 (0,1,1) を指定（あとで displayStress() で上書き）
    Eigen::MatrixXd defaultColor = Eigen::MatrixXd::Constant(graph.mEdgeNum, 3, 0.0);
    for(int i=0;i<graph.mEdgeNum;i++){
        defaultColor.row(i) = Eigen::RowVector3d(0,1,1);
    }
    viewer.data().line_width = 10.0;
    viewer.data().set_edges(Node4Show, E, defaultColor);
}

//
// --- 以下、6自由度ビーム要素 FEM シミュレーションおよび剪断応力計算 ---
//
// グローバル変数：解ベクトル u と各要素の剪断応力
Eigen::VectorXd g_u;
Eigen::VectorXd g_elementStress;

struct FEMArg
{
    double E = 1e12;        // Young率（材料の剛性）
    double A = 0.01;       // 断面積
    double I_y = 1e-6;     // 曲げ剛性（local z方向に対する曲げ、すなわち y軸回りの曲げ）
    double I_z = 1e-6;     // 曲げ剛性（local y方向に対する曲げ、すなわち z軸回りの曲げ）
    double nu = 0.3;       // ポアソン比
    double J = 2e-6;       // ねじり定数
    double mass = 1.0;
    double scale = 1.0;
};


//
// 6自由度ビーム要素 FEM シミュレーション
// 各要素は 12×12 局所剛性行列を持ち、局所座標系からグローバル座標系へ変換して組み立てます。
// 重力は負の Y 方向に作用し、底面（Y ≒ 0）のノードは全6自由度固定とします。
//
void simulateBeamFEM6DOF(Graph& graph, FEMArg& arg)
{
    // ノード数と全体の自由度（各ノード6自由度）
    const int N = graph.mNodeNum;
    const int dof = 6 * N;
    
    // グローバル剛性行列 K（sparse）と外力ベクトル f を初期化
    Eigen::SparseMatrix<double> K(dof, dof);
    Eigen::VectorXd f = Eigen::VectorXd::Zero(dof);
    std::vector<Eigen::Triplet<double>> triplets;

    // 材料および断面パラメータ
    double E = arg.E;        // Young率（材料の剛性）
    double A = arg.A;      // 断面積
    double I_y = arg.I_y;     // 曲げ剛性（local z方向に対する曲げ、すなわち y軸回りの曲げ）
    double I_z = arg.I_z;     // 曲げ剛性（local y方向に対する曲げ、すなわち z軸回りの曲げ）
    double nu = arg.nu;       // ポアソン比
    double G = E / (2*(1+nu));  // せん断弾性率（G = E/2(1+ν)）
    double J = arg.J;       // ねじり定数
    double mass = arg.mass;     // 各ノードの質量
    double g = 9.81;       // 重力加速度

    // --- 各ビーム要素（グラフの各エッジ）に対して ---
    for (int e = 0; e < graph.mEdgeNum; e++) {
        // ① 現在のエッジから両端のノード番号を取得
        Edge* edge = graph.mEdgeList[e];
        int nodeA = edge->mNodeA;
        int nodeB = edge->mNodeB;

        // ② 各ノードのグローバル座標を取得
        Eigen::Vector3d A_pos(
            graph.mNodeList[nodeA]->mPos.mPosx,
            graph.mNodeList[nodeA]->mPos.mPosy,
            graph.mNodeList[nodeA]->mPos.mPosz);
        Eigen::Vector3d B_pos(
            graph.mNodeList[nodeB]->mPos.mPosx,
            graph.mNodeList[nodeB]->mPos.mPosy,
            graph.mNodeList[nodeB]->mPos.mPosz);
        
        // ③ 要素の方向ベクトル d とその長さ L を計算
        Eigen::Vector3d d = B_pos - A_pos;
        double L = d.norm();
        if (L < 1e-6) continue;  // 長さが非常に短い場合はスキップ
        // 要素軸方向（局所 x 軸）の単位ベクトル n_x
        Eigen::Vector3d n_x = d / L;

        // ④ 局所座標系の決定
        //    - n_x : 要素の軸方向（局所 x 軸）
        //    - グローバル Y 軸を基準に、n_x と垂直な軸 n_y, n_z を定める
        //Eigen::Vector3d temp = globalY;
        //if (std::abs(n_x.dot(globalY)) > 0.99)
        //    temp = Eigen::Vector3d(1, 0, 0); // n_x と globalY がほぼ平行な場合は別の基準を使用
        // 局所 z 軸 n_z を、n_x と temp の外積から求め、正規化
        //Eigen::Vector3d n_y = temp.cross(n_x).normalized();
        // 局所 y 軸 n_y は、n_z と n_x の外積で得る（正規化済み）
        //Eigen::Vector3d n_z = n_x.cross(n_y);

        Eigen::Vector3d n_y{-n_x.y()/std::sqrt(n_x.y()*n_x.y() + n_x.x()*n_x.x()), n_x.x()*std::sqrt(n_x.y()*n_x.y() + n_x.x()*n_x.x()), 0};
        Eigen::Vector3d n_z{-(n_x.x()*n_x.z())/std::sqrt(n_x.y()*n_x.y() + n_x.x()*n_x.x()), -(n_x.y()*n_x.z())*std::sqrt(n_x.y()*n_x.y() + n_x.x()*n_x.x()), std::sqrt(n_x.y()*n_x.y() + n_x.x()*n_x.x())};

        if( std::abs(n_x.x()) < 0.001 && std::abs(n_x.y()) < 0.001)
        {
            auto nz = n_x.z();
            n_x = Eigen::Vector3d{0.0, 0.0, nz};
            n_y = Eigen::Vector3d{nz, 0.0, 0.0};
            n_z = Eigen::Vector3d{0.0, nz, 0.0};
        }

        // ⑤ 回転行列 R の構成
        //    R の各列に局所座標軸 (n_x, n_y, n_z) を格納し、局所→グローバル変換を表現
        Eigen::Matrix3d R;
        R.col(0) = n_x;
        R.col(1) = n_y;
        R.col(2) = n_z;

        // ⑥ 各ノードの 6自由度（平行移動と回転）の変換行列 T6（6×6）
        //    T6 は、平行移動と回転それぞれに R を適用するブロック対角行列
        //Eigen::Matrix<double,6,6> T6 = Eigen::Matrix<double,6,6>::Zero();
        //T6.block<3,3>(0,0) = R;  // 平行移動成分
        //T6.block<3,3>(3,3) = R;  // 回転成分

        // ⑦ 要素全体の変換行列 T_elem（12×12）を構成（ノードA, ノードBに対して T6 を並べる）
        //Eigen::Matrix<double,12,12> T_elem = Eigen::Matrix<double,12,12>::Zero();
        //T_elem.block<6,6>(0,0) = T6;
        //T_elem.block<6,6>(6,6) = T6;

        // ⑥ 各ノードの 6自由度（平行移動と回転）の変換行列 T6（6×6）
        //    T6 は、平行移動と回転それぞれに R を適用するブロック対角行列
        Eigen::Matrix<double,6,6> T6 = Eigen::Matrix<double,6,6>::Zero();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                T6(i, j) = R(i, j);         // 平行移動成分に R を代入
                T6(i + 3, j + 3) = R(i, j);   // 回転成分にも R を代入
            }
        }

        // ⑦ 要素全体の変換行列 T_elem（12×12）を構成（ノードA, ノードBに対して T6 を並べる）
        Eigen::Matrix<double,12,12> T_elem = Eigen::Matrix<double,12,12>::Zero();
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                T_elem(i, j) = T6(i, j);         // ノードAの部分
                T_elem(i + 6, j + 6) = T6(i, j);   // ノードBの部分
            }
        }


        // ⑧ 局所剛性行列 k_local の組み立て（12×12）
        //     ここでは、Euler–Bernoulli ビーム要素の近似式に基づく各項を設定
        //     - 軸方向剛性: k_axial = E*A/L
        //     - 曲げ剛性: k_bend_y, k_bend_z（L³ に反比例）
        //     - せん断項: k_shear_y, k_shear_z（L² に反比例）
        //     - ねじり剛性: k_torsion = G*J/L

        Eigen::Matrix<double,12,12> k_local = Eigen::Matrix<double,12,12>::Zero();
        double k_axial = E * A / L;
        double k_bend_y = 12 * E * I_z / (L * L * L);
        double k_shear_y = 6 * E * I_z / (L * L);
        double k_bend_z = 12 * E * I_y / (L * L * L);
        double k_shear_z = 6 * E * I_y / (L * L);
        double k_torsion = G * J / L;
        
        // 以下、局所座標系における剛性行列の各要素を組み立てる（対称性を利用）
        // DOF の順序：ノードA：[u_x, u_y, u_z, theta_x, theta_y, theta_z]、ノードB：同じ順序
        // 軸方向項（u_x 間の相互作用）
        k_local(0,0)   =  k_axial;   k_local(0,6)  = -k_axial;
        k_local(6,0)   = -k_axial;   k_local(6,6)  =  k_axial;

        // 曲げ・せん断項（ノードAの u_y, theta_z など）
        k_local(1,1)   =  k_bend_y;  k_local(1,5)  =  k_shear_y;
        k_local(1,7)   = -k_bend_y;  k_local(1,11) =  k_shear_y;
        k_local(5,1)   =  k_shear_y; k_local(5,5)  =  4*E*I_z/L;
        k_local(5,7)   = -k_shear_y; k_local(5,11) =  2*E*I_z/L;
        k_local(7,1)   = -k_bend_y;  k_local(7,5)  = -k_shear_y;
        k_local(7,7)   =  k_bend_y;  k_local(7,11) =  k_shear_y;
        k_local(11,1)  =  k_shear_y; k_local(11,5) =  2*E*I_z/L;
        k_local(11,7)  =  k_shear_y; k_local(11,11)= 4*E*I_z/L;

        // 同様に、曲げ・せん断項（u_z, theta_y 間の相互作用）
        k_local(2,2)   =  k_bend_z;  k_local(2,4)  = -k_shear_z;
        k_local(2,8)   = -k_bend_z;  k_local(2,10) = -k_shear_z;
        k_local(4,2)   = -k_shear_z; k_local(4,4)  =  4*E*I_y/L;
        k_local(4,8)   =  k_shear_z; k_local(4,10) =  2*E*I_y/L;
        k_local(8,2)   = -k_bend_z;  k_local(8,4)  =  k_shear_z;
        k_local(8,8)   =  k_bend_z;  k_local(8,10) =  k_shear_z;
        k_local(10,2)  = -k_shear_z; k_local(10,4) =  2*E*I_y/L;
        k_local(10,8)  =  k_shear_z; k_local(10,10)= 4*E*I_y/L;

        // ねじり項（theta_x 間の相互作用）
        k_local(3,3)   =  k_torsion; k_local(3,9)  = -k_torsion;
        k_local(9,3)   = -k_torsion; k_local(9,9)  =  k_torsion;

        // 対称性を明示的に保証するため、上三角成分を下三角にコピー
        for (int i = 0; i < 12; i++) {
            for (int j = i+1; j < 12; j++) {
                k_local(j,i) = k_local(i,j);
            }
        }

        // ⑨ 局所剛性行列 k_local をグローバル座標系に変換する：
        //     K_e = T_elem^T * k_local * T_elem
        Eigen::Matrix<double,12,12> K_e = T_elem.transpose() * k_local * T_elem;

        // ⑩ 要素のグローバル自由度のマッピング：
        //     ノードAは自由度 6*nodeA ～ 6*nodeA+5、ノードBは 6*nodeB ～ 6*nodeB+5
        int global_dofs[12];
        for (int i = 0; i < 6; i++) {
            global_dofs[i]    = 6 * nodeA + i;
            global_dofs[i+6]  = 6 * nodeB + i;
        }
        // ⑪ 各要素の剛性行列 K_e の寄与を、global triplets に追加して組み立てる
        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 12; j++) {
                triplets.push_back(Eigen::Triplet<double>(global_dofs[i], global_dofs[j], K_e(i,j)));
            }
        }
    }
    // グローバル剛性行列 K を triplets から組み立てる
    K.setFromTriplets(triplets.begin(), triplets.end());

    // ⑫ 重力荷重の適用：各ノードの Y 方向（DOF: 6*i + 1）に、負の方向へ質量×g の力を加える
    for (int i = 0; i < N; i++) {
        int dof_y = 6 * i + 1;
        f(dof_y) -= mass * g;
    }
    
    // ⑬ 境界条件（固定条件）の適用：
    //     Y 座標がほぼ0のノードを全6自由度で拘束する。
    std::vector<int> fixed_dofs;
    int fixed_num = 0;
    for (int i = 0; i < N; i++) {
        Vector3 pos = graph.mNodeList[i]->mPos;
        if (pos.mPosy <= 2.1) {
            if (CustomNode* c_node = dynamic_cast<CustomNode*>(graph.mNodeList[i]))
                c_node->mFixed = true;
            for (int j = 0; j < 6; j++)
                fixed_dofs.push_back(6 * i + j);
            fixed_num++;
        }
    }
    std::cout << "Number of fixed nodes: " << fixed_num << std::endl;
    // 各固定自由度について、対応する行・列をゼロにし、対角項を1に設定、外力もゼロにする
    for (int dof_idx : fixed_dofs) {
    // 行をゼロにする
        for (Eigen::SparseMatrix<double>::InnerIterator it(K, dof_idx); it; ++it) {
            it.valueRef() = 0.0;
        }

        // 列をゼロにする
        for (int row = 0; row < K.rows(); ++row) {
            if (K.coeff(row, dof_idx) != 0.0) {
                K.coeffRef(row, dof_idx) = 0.0;
            }
        }

        // 対角成分を1にする
        K.coeffRef(dof_idx, dof_idx) = 1.0;

        // 外力ベクトルを0にする
        f(dof_idx) = 0.0;
    }

    // ⑭ 線形系 K * u = f を解く（ここでは SimplicialLDLT を使用）
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(K);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Failed to decompose the stiffness matrix K." << std::endl;
        return;
    }
    Eigen::VectorXd u = solver.solve(f);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Failed to solve for displacement vector u." << std::endl;
        return;
    }
    
    // ⑮ FEM 解 u をグローバル変数に保存
    g_u = u;
    
    // ⑯ 各要素の剪断応力の計算
    //     ここでは、各エッジについて、両端ノードの平行移動変位（最初の3DOF）を取得し、
    //     要素の元の軸（A_orig から B_orig を計算）を推定します。
    //     その後、相対変位から軸方向成分を除去した残差が剪断変位となり、
    //     剪断ひずみはその大きさを要素長 L で割ったもの、剪断応力は G * 剪断ひずみ です。
    {
        int numElem = graph.mEdgeNum;
        g_elementStress.resize(numElem);
        for (int e = 0; e < numElem; e++){
            Edge* edge = graph.mEdgeList[e];
            int nodeA = edge->mNodeA;
            int nodeB = edge->mNodeB;
            // ノードA, ノードBの平行移動変位（最初の3DOF）
            Eigen::Vector3d uA = u.segment(6 * nodeA, 3);
            Eigen::Vector3d uB = u.segment(6 * nodeB, 3);
            // 近似的に元の位置を求める：現在位置から変位分を引く
            Eigen::Vector3d A_orig(
                graph.mNodeList[nodeA]->mPos.mPosx - static_cast<float>(uA(0)),
                graph.mNodeList[nodeA]->mPos.mPosy - static_cast<float>(uA(1)),
                graph.mNodeList[nodeA]->mPos.mPosz - static_cast<float>(uA(2))
            );
            Eigen::Vector3d B_orig(
                graph.mNodeList[nodeB]->mPos.mPosx - static_cast<float>(uB(0)),
                graph.mNodeList[nodeB]->mPos.mPosy - static_cast<float>(uB(1)),
                graph.mNodeList[nodeB]->mPos.mPosz - static_cast<float>(uB(2))
            );
            // 元の要素軸ベクトルと長さ
            Eigen::Vector3d d_orig = B_orig - A_orig;
            double L = d_orig.norm();
            if (L < 1e-6) { g_elementStress(e) = 0; continue; }
            Eigen::Vector3d n_x = d_orig / L;
            // 両端ノード間の平行移動変位の差（相対変位）
            Eigen::Vector3d du = uB - uA;
            // 軸方向成分を除去： (du.dot(n_x))*n_x が軸方向変位なので、残差が剪断変位
            Eigen::Vector3d shear_disp = du - (du.dot(n_x)) * n_x;
            // 剪断ひずみは、剪断変位の大きさを要素長で割る
            double shear_strain = shear_disp.norm() / L;
            // 剪断応力は、せん断弾性率 G と剪断ひずみの積
            double shear_stress = G * shear_strain;
            g_elementStress(e) = shear_stress;
        }
    }
    
    // ⑰ 各ノードの位置更新： FEM 解のうち、平行移動成分（最初の3DOF）を加算して新しい位置を計算
    for (int i = 0; i < N; i++) {
        int base = 6 * i;
        double du = u(base + 0);
        double dv = u(base + 1);
        double dw = u(base + 2);
        graph.mNodeList[i]->mPos.mPosx += arg.scale*static_cast<float>(du);
        graph.mNodeList[i]->mPos.mPosy += arg.scale*static_cast<float>(dv);
        graph.mNodeList[i]->mPos.mPosz += arg.scale*static_cast<float>(dw);
    }
    std::cout << "Norm of displacement solution: " << u.norm() << std::endl;
}


//
// displayStress: 各エッジの剪断応力に基づいてエッジの色を設定して表示する関数
// V: 各ノードの座標行列, E: エッジ接続情報行列, C: エッジごとの色行列（剪断応力に応じた色）
// ここでは、set_edges() を使用して表示します。
//
void displayStress(const Graph& graph, igl::opengl::glfw::Viewer& viewer, const Eigen::VectorXd& stress)
{
    // 頂点行列 V を作成
    Eigen::MatrixXd V(graph.mNodeNum, 3);
    for (int i = 0; i < graph.mNodeNum; i++){
        V(i,0) = graph.mNodeList[i]->mPos.mPosx;
        V(i,1) = graph.mNodeList[i]->mPos.mPosy;
        V(i,2) = graph.mNodeList[i]->mPos.mPosz;
    }
    // エッジ接続情報行列 E を作成
    Eigen::MatrixXi E(graph.mEdgeNum, 2);
    for (int i = 0; i < graph.mEdgeNum; i++){
        E(i,0) = graph.mEdgeList[i]->mNodeA;
        E(i,1) = graph.mEdgeList[i]->mNodeB;
    }
    // stress の最小／最大値を算出し、各エッジの色を線形補間で決定
    double smin = stress.minCoeff();
    double smax = stress.maxCoeff();
    Eigen::MatrixXd edgeColors(graph.mEdgeNum, 3);
    for (int i = 0; i < graph.mEdgeNum; i++){
        double t = (stress(i) - smin) / (smax - smin + 1e-8);
        // t = 0 → 青 (0,0,1), t = 1 → 赤 (1,0,0)
        edgeColors.row(i) = Eigen::RowVector3d(t, 0, 1-t);
    }
    // set_edges(V, E, C) でエッジ表示
    viewer.data().line_width = 10.0;
    viewer.data().set_edges(V, E, edgeColors);
}

//
// シミュレーション実行用パラメータ
//
struct SimArg
{
    int mGridX = 5;
    int mGridY = 5;
    int mGridZ = 5;
};

//
// runSim: 立方格子生成、シミュレーション実施、剪断応力表示を行う
//
void runSim(igl::opengl::glfw::Viewer& viewer, SimArg& sim_arg, FEMArg& fem_arg)
{
    Graph graph;
    // 立方格子生成：縦方向 sim_arg.mGridX 個、横方向 sim_arg.mGridZ 個、高さ sim_arg.mGridY 個
    //GenerateCubicGrid(graph, sim_arg.mGridX, sim_arg.mGridZ, sim_arg.mGridY);
    GenerateTableGraph(graph, sim_arg.mGridX, sim_arg.mGridZ, sim_arg.mGridY);

    // 初期状態（固定前）のグラフを緑色で表示
    entryGraph4show(graph, viewer, Vector3{0.0f, 1.0f, 0.0f});
    // 6自由度モデルで FEM シミュレーション実施
    simulateBeamFEM6DOF(graph, fem_arg);
    // シミュレーション後のグラフ（ノード）は赤色で表示
    entryGraph4show(graph, viewer, Vector3{1.0f, 0.0f, 0.0f});
    // 剪断応力に応じたエッジの色表示
    displayStress(graph, viewer, g_elementStress);
}

static SimArg simArg;
static FEMArg femArg;
//
// キー入力処理：キー '1' でシミュレーション再実行、'2' でクリア
//
bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
    std::cout << "Key: " << key << " (" << (unsigned int)key << ")" << std::endl;
    if (key == '1')
    {
        viewer.data().clear();
        runSim(viewer, simArg, femArg);
    }
    else if (key == '2')
    {
        viewer.data().clear();
    }
    return false;
}

//
// main: igl::Viewer および imGui メニューを用いてシミュレーション実行
//
int main(int argc, char *argv[])
{
    igl::opengl::glfw::Viewer viewer;
    viewer.callback_key_down = &key_down;

    // imGui のプラグインとメニューを追加
    igl::opengl::glfw::imgui::ImGuiPlugin plugin;
    viewer.plugins.push_back(&plugin);
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    plugin.widgets.push_back(&menu);

    // imGui メニューの設定（グリッド数の調整など）
    menu.callback_draw_viewer_menu = [&]()
    {
        menu.draw_viewer_menu();
        if (ImGui::CollapsingHeader("Simulation Settings", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::SliderInt("GridX", &simArg.mGridX, 2, 20);
            ImGui::SliderInt("GridY", &simArg.mGridY, 2, 20);
            ImGui::SliderInt("GridZ", &simArg.mGridZ, 2, 20);
            ImGui::InputDouble("E", &femArg.E);
            ImGui::InputDouble("A", &femArg.A);
            ImGui::InputDouble("I_y", &femArg.I_y);
            ImGui::InputDouble("I_z", &femArg.I_z);
            ImGui::InputDouble("j", &femArg.J);
            ImGui::InputDouble("mass", &femArg.mass);
            ImGui::InputDouble("nu", &femArg.nu);
            ImGui::InputDouble("scale", &femArg.scale);


            if (ImGui::Button("Run Simulation", ImVec2(-1,0)))
            {
                viewer.data().clear();
                runSim(viewer, simArg, femArg);
            }
        }
    };

    runSim(viewer, simArg, femArg);
    viewer.launch();
    
    return 0;
}
