#include <iostream>
#include <vector>
#include <Eigen/Dense>

// ======= Element クラス群 =======

// 要素の基底クラス：全要素共通のインターフェース
class Element {
public:
    // この例では nodeIDs は「節点番号」を保持します（各節点は 6 自由度）
    std::vector<int> nodeIDs;
    
    Element() {}
    virtual ~Element() {}

    // 局所剛性マトリクスを返す純粋仮想関数
    virtual Eigen::MatrixXd getLocalStiffnessMatrix() const = 0;
};

// BeamElement クラス：Element を継承し、BeamFEM 用のパラメータを保持  
// 生成時に局所剛性マトリクス (12×12) を計算してメンバ変数に保持します。
class BeamElement : public Element {
public:
    double E;   // ヤング率
    double A;   // 断面積
    double Iy;  // y 軸回り断面二次モーメント
    double Iz;  // z 軸回り断面二次モーメント
    double G;   // せん断弾性係数
    double J;   // ねじり定数
    double L;   // 要素長さ

    // 局所剛性マトリクス（12×12）を保持
    Eigen::MatrixXd localStiffnessMatrix;

    // コンストラクタ: nodeIDs は 2 つの節点番号（例： {0, 1}）を指定
    BeamElement(const std::vector<int>& nodeIDs_, double E_, double A_, double Iy_, double Iz_, double G_, double J_, double L_)
      : E(E_), A(A_), Iy(Iy_), Iz(Iz_), G(G_), J(J_), L(L_)
    {
        if (nodeIDs_.size() != 2) {
            std::cerr << "BeamElement requires 2 node IDs." << std::endl;
        }
        nodeIDs = nodeIDs_;

        // 12×12 の局所剛性マトリクスを生成
        localStiffnessMatrix = Eigen::MatrixXd::Zero(12, 12);

        // 定数計算
        double EA  = E * A;
        double GJ  = G * J;
        double EIy = E * Iy;
        double EIz = E * Iz;

        // ----- Axial (u_x) -----
        // 節点 1: DOF 0, 節点 2: DOF 6
        localStiffnessMatrix(0,0) = EA / L;
        localStiffnessMatrix(0,6) = -EA / L;
        localStiffnessMatrix(6,0) = -EA / L;
        localStiffnessMatrix(6,6) = EA / L;

        // ----- Torsion (θ_x) -----
        // 節点 1: DOF 3, 節点 2: DOF 9
        localStiffnessMatrix(3,3) = GJ / L;
        localStiffnessMatrix(3,9) = -GJ / L;
        localStiffnessMatrix(9,3) = -GJ / L;
        localStiffnessMatrix(9,9) = GJ / L;

        // ----- Bending about z (曲げ：y 方向変位 u_y と θ_z) -----
        // 節点 1: u_y (DOF 1), θ_z (DOF 5); 節点 2: u_y (DOF 7), θ_z (DOF 11)
        localStiffnessMatrix(1,1)   = 12 * EIz / (L*L*L);
        localStiffnessMatrix(1,5)   = 6  * EIz / (L*L);
        localStiffnessMatrix(1,7)   = -12 * EIz / (L*L*L);
        localStiffnessMatrix(1,11)  = 6  * EIz / (L*L);

        localStiffnessMatrix(5,1)   = 6  * EIz / (L*L);
        localStiffnessMatrix(5,5)   = 4  * EIz / L;
        localStiffnessMatrix(5,7)   = -6  * EIz / (L*L);
        localStiffnessMatrix(5,11)  = 2  * EIz / L;

        localStiffnessMatrix(7,1)   = -12 * EIz / (L*L*L);
        localStiffnessMatrix(7,5)   = -6  * EIz / (L*L);
        localStiffnessMatrix(7,7)   = 12 * EIz / (L*L*L);
        localStiffnessMatrix(7,11)  = -6  * EIz / (L*L);

        localStiffnessMatrix(11,1)  = 6  * EIz / (L*L);
        localStiffnessMatrix(11,5)  = 2  * EIz / L;
        localStiffnessMatrix(11,7)  = -6  * EIz / (L*L);
        localStiffnessMatrix(11,11) = 4  * EIz / L;

        // ----- Bending about y (曲げ：z 方向変位 u_z と θ_y) -----
        // 節点 1: u_z (DOF 2), θ_y (DOF 4); 節点 2: u_z (DOF 8), θ_y (DOF 10)
        localStiffnessMatrix(2,2)   = 12 * EIy / (L*L*L);
        localStiffnessMatrix(2,4)   = -6  * EIy / (L*L);
        localStiffnessMatrix(2,8)   = -12 * EIy / (L*L*L);
        localStiffnessMatrix(2,10)  = -6  * EIy / (L*L);

        localStiffnessMatrix(4,2)   = -6  * EIy / (L*L);
        localStiffnessMatrix(4,4)   = 4  * EIy / L;
        localStiffnessMatrix(4,8)   = 6  * EIy / (L*L);
        localStiffnessMatrix(4,10)  = 2  * EIy / L;

        localStiffnessMatrix(8,2)   = -12 * EIy / (L*L*L);
        localStiffnessMatrix(8,4)   = 6  * EIy / (L*L);
        localStiffnessMatrix(8,8)   = 12 * EIy / (L*L*L);
        localStiffnessMatrix(8,10)  = 6  * EIy / (L*L);

        localStiffnessMatrix(10,2)  = -6  * EIy / (L*L);
        localStiffnessMatrix(10,4)  = 2  * EIy / L;
        localStiffnessMatrix(10,8)  = 6  * EIy / (L*L);
        localStiffnessMatrix(10,10) = 4  * EIy / L;
    }

    virtual ~BeamElement() {}

    // 既に計算済みの局所剛性マトリクスを返す
    virtual Eigen::MatrixXd getLocalStiffnessMatrix() const override {
        return localStiffnessMatrix;
    }
};

// ======= FEMSimulator クラス群 =======

// FEM シミュレーターの基底クラス（グローバル剛性マトリクス、変位、荷重を保持）
class FEMSimulator {
protected:
    Eigen::MatrixXd K; // Global stiffness matrix
    Eigen::VectorXd u; // Displacement vector
    Eigen::VectorXd F; // Load vector
public:
    FEMSimulator(int numDOF) {
        K = Eigen::MatrixXd::Zero(numDOF, numDOF);
        u = Eigen::VectorXd::Zero(numDOF);
        F = Eigen::VectorXd::Zero(numDOF);
    }
    virtual ~FEMSimulator() {}
    virtual void assemble() = 0;
    virtual void applyBoundaryConditions() {}
    virtual void solve() {
        u = K.partialPivLu().solve(F);
    }
    virtual void printResults() const {
        std::cout << "Displacement vector u:\n" << u << std::endl;
    }
    Eigen::VectorXd& getLoadVector() { return F; }
};

// BeamFEMSimulator クラス：FEMSimulator を継承し、Beam 要素特有の組み立てを実装
class BeamFEMSimulator : public FEMSimulator {
public:
    BeamFEMSimulator(int numDOF) : FEMSimulator(numDOF) {}
    virtual ~BeamFEMSimulator() {}

    // パラメータ無しの assemble は利用せず警告出力
    virtual void assemble() override {
        std::cerr << "BeamFEMSimulator::assemble() には Element* のリストを渡してください。" << std::endl;
    }

    // Element*（BeamElement を含む）のリストからグローバル剛性マトリクスを組み立てる
    // 各要素は 2 節点で、各節点は 6 自由度（計 12 DOF）の情報となる
    void assemble(const std::vector<Element*>& elements) {
        K.setZero();
        for (const auto& elem : elements) {
            if (elem->nodeIDs.size() != 2) {
                std::cerr << "Error: Element does not have 2 nodes." << std::endl;
                continue;
            }
            int node1 = elem->nodeIDs[0];
            int node2 = elem->nodeIDs[1];
            // 各節点 6 自由度 → 要素全体では 12 DOF
            std::vector<int> dofIndices(12);
            for (int i = 0; i < 6; ++i) {
                dofIndices[i] = node1 * 6 + i;
                dofIndices[i + 6] = node2 * 6 + i;
            }
            Eigen::MatrixXd k_local = elem->getLocalStiffnessMatrix();
            // 局所剛性マトリクスの各項を対応するグローバル位置に加算
            for (int i = 0; i < 12; ++i) {
                for (int j = 0; j < 12; ++j) {
                    K(dofIndices[i], dofIndices[j]) += k_local(i, j);
                }
            }
        }
    }

    // 境界条件の適用例：節点 0 を完全に固定（その節点の 6 自由度を 0 に固定）
    virtual void applyBoundaryConditions() override {
        int fixedNode = 0;
        for (int i = 0; i < 6; ++i) {
            int globalIndex = fixedNode * 6 + i;
            K.row(globalIndex).setZero();
            K.col(globalIndex).setZero();
            K(globalIndex, globalIndex) = 1.0;
            F(globalIndex) = 0.0;
        }
    }
};

//
// ======= main 関数 =======
//

int main() {
    // 例：3 節点，2 要素（節点 0-1, 1-2）の 3 次元 Beam FEM
    // 各節点は 6 自由度を持つため、全体自由度数は 3 * 6 = 18
    int numNodes = 3;
    int numDOF = numNodes * 6;
    BeamFEMSimulator simulator(numDOF);

    // 材料および断面パラメータ（例）
    double E  = 210e9;    // ヤング率 [Pa]
    double A  = 0.01;     // 断面積 [m^2]
    double Iy = 8.1e-6;    // y 軸回り断面二次モーメント [m^4]
    double Iz = 8.1e-6;    // z 軸回り断面二次モーメント [m^4]
    double G  = 80e9;     // せん断弾性係数 [Pa]（例）
    double J  = 1e-5;     // ねじり定数 [m^4]（例）
    double L  = 2.0;      // 要素長さ [m]

    // Element* のリストを作成（例：節点 0-1, 1-2 で接続）
    std::vector<Element*> beamElements;
    beamElements.push_back(new BeamElement({0, 1}, E, A, Iy, Iz, G, J, L));
    beamElements.push_back(new BeamElement({1, 2}, E, A, Iy, Iz, G, J, L));

    // Element のリストを渡してグローバル剛性マトリクスを組み立て
    simulator.assemble(beamElements);

    // 荷重の設定：例として、節点 2 においてグローバル z 方向（u_z）のみに下向き荷重を作用
    // 節点 2 の DOF はインデックス 12～17 のうち、u_z は 14 番目（0-indexed）と仮定
    simulator.getLoadVector()(14) = -1000.0;  // [N]

    // 境界条件の適用：ここでは節点 0 を完全に固定
    simulator.applyBoundaryConditions();

    // 連立方程式 K * u = F の解を求める
    simulator.solve();

    // 結果の表示
    simulator.printResults();

    // 動的に確保した要素の解放
    for (auto elem : beamElements) {
        delete elem;
    }

    return 0;
}
