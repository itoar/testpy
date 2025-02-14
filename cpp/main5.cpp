#include <cstdio>
#include <cmath>
#include <cassert>

//-----------------------------------------
// CSRMatrixクラス（必要な部分のみ抜粋）
//-----------------------------------------
template<typename T>
class CSRMatrix {
public:
    int mRowNum, mColNum;
    T* mVal;
    int* mColIndex;
    int* mRowPtr;
    int mNonZeroNum;

    CSRMatrix(int r, int c) : mRowNum(r), mColNum(c), mVal(nullptr), mColIndex(nullptr), mRowPtr(nullptr), mNonZeroNum(0) {}

    ~CSRMatrix() {
        delete[] mVal;
        delete[] mColIndex;
        delete[] mRowPtr;
    }

    // CSR形式の疎行列とベクトルの積を計算する関数
    void multiply(const T* vec, T* result) const {
        for (int i = 0; i < mRowNum; ++i) {
            result[i] = static_cast<T>(0);
            for (int idx = mRowPtr[i]; idx < mRowPtr[i + 1]; ++idx) {
                result[i] += mVal[idx] * vec[mColIndex[idx]];
            }
        }
    }
};

//-----------------------------------------
// ベクトル演算ヘルパー関数
//-----------------------------------------

// 内積計算
template<typename T>
T dot(const T* a, const T* b, int size) {
    T sum = static_cast<T>(0);
    for (int i = 0; i < size; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// ベクトルのスカラー倍と加算: result = a + alpha * b
template<typename T>
void axpy(T alpha, const T* b, const T* a, T* result, int size) {
    for (int i = 0; i < size; ++i) {
        result[i] = a[i] + alpha * b[i];
    }
}

// ベクトルのノルム（L2ノルム）
template<typename T>
T norm(const T* vec, int size) {
    return std::sqrt(dot(vec, vec, size));
}

//-----------------------------------------
// 共役勾配法 (Conjugate Gradient Method)
//-----------------------------------------
template<typename T>
void conjugateGradient(const CSRMatrix<T>& A, const T* b, T* x, int maxIter, T tol) {
    int n = A.mRowNum;
    T* r = new T[n];      // 残差ベクトル r = b - Ax
    T* p = new T[n];      // 検索方向ベクトル
    T* Ap = new T[n];     // A * p の結果を保存

    // 初期残差 r = b - A*x
    A.multiply(x, r);
    for (int i = 0; i < n; ++i) {
        r[i] = b[i] - r[i]; // 残差計算
        p[i] = r[i];        // 初期方向 p_0 = r_0
    }

    T rsold = dot(r, r, n); // 初期残差の二乗ノルム

    for (int k = 0; k < maxIter; ++k) {
        A.multiply(p, Ap);  // Ap = A * p
        T alpha = rsold / dot(p, Ap, n); // α_k = (r_k^T r_k) / (p_k^T A p_k)

        // x_k+1 = x_k + α_k * p_k
        axpy(alpha, p, x, x, n);

        // r_k+1 = r_k - α_k * A p_k
        axpy(-alpha, Ap, r, r, n);

        T rsnew = dot(r, r, n); // 残差ノルムの更新

        // 収束判定
        if (std::sqrt(rsnew) < tol) {
            break;
        }

        // β_k = (r_k+1^T r_k+1) / (r_k^T r_k)
        T beta = rsnew / rsold;

        // p_k+1 = r_k+1 + β_k * p_k
        axpy(beta, p, r, p, n);

        rsold = rsnew; // 残差更新
    }

    // メモリ解放
    delete[] r;
    delete[] p;
    delete[] Ap;
}

//-----------------------------------------
// メイン関数（サンプル実行）
//-----------------------------------------
int main() {
    const int n = 3;
    
    // --- 1. 正定値疎行列 A の CSR 形式 ---
    CSRMatrix<double> A(n, n);
    A.mNonZeroNum = 5;
    A.mVal = new double[A.mNonZeroNum]{ 4.0, -1.0, -1.0, 4.0, -1.0 };
    A.mColIndex = new int[A.mNonZeroNum]{ 0, 1, 0, 1, 2 };
    A.mRowPtr = new int[n + 1]{ 0, 2, 4, 5 };

    // --- 2. 右辺ベクトル b ---
    double b[n] = { 2.0, 1.0, 2.0 };

    // --- 3. 初期解 x (ゼロベクトル) ---
    double x[n] = { 0.0, 0.0, 0.0 };

    // --- 4. CG 法の実行 ---
    int maxIter = 100;
    double tol = 1e-6;
    conjugateGradient(A, b, x, maxIter, tol);

    // --- 5. 結果表示 ---
    printf("解 x:\n");
    for (int i = 0; i < n; ++i) {
        printf("x[%d] = %f\n", i, x[i]);
    }

    return 0;
}
