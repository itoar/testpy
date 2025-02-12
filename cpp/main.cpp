#include <cstdio>

// CSR形式の疎行列クラス（バッファを使用）
// テンプレートパラメータ T により、要素の型を指定できるようになっています。
template<typename T>
class CSRMatrix {
public:
    int mRowNum, mColNum;       // 行数、列数
    T*      mVal;               // 非ゼロ要素の値のバッファ
    int*    mColIndex;          // 非ゼロ要素の列インデックスのバッファ
    int*    mRowPtr;            // 各行の開始位置（非ゼロ要素の累積個数）のバッファ
    int     mNonZeroNum;        // 非ゼロ要素数

    // コンストラクタ：行数・列数を指定。各ポインタは初期状態ではnullptrにしておく
    CSRMatrix(const int r, const int c)
        : mRowNum(r), mColNum(c), mVal(nullptr), mColIndex(nullptr), mRowPtr(nullptr), mNonZeroNum(0) {}

    // デストラクタ：確保したバッファを解放する
    ~CSRMatrix() {
        delete[] mVal;
        delete[] mColIndex;
        delete[] mRowPtr;
    }

    // 密行列からCSR形式へ変換する関数
    void buildFromDense(const T* dense) {
        // まず非ゼロ要素数 (mNonZeroNum) を数える
        mNonZeroNum = 0;
        for (int i = 0; i < mRowNum; ++i) {
            for (int j = 0; j < mColNum; ++j) {
                if (dense[i * mColNum + j] != 0.0) {
                    ++mNonZeroNum;
                }
            }
        }

        // 必要なサイズ分、各バッファのメモリを確保
        mVal        = new T[mNonZeroNum];
        mColIndex   = new int[mNonZeroNum];
        mRowPtr     = new int[mRowNum + 1];

        int count = 0;
        mRowPtr[0] = 0;
        // 各行ごとに走査し、非ゼロ要素の場合に値と列番号を記録
        for (int i = 0; i < mRowNum; ++i) {
            for (int j = 0; j < mColNum; ++j) {
                T val = dense[i * mColNum + j];  // 必要なら型変換が行われる
                if (val != 0.0) {
                    mVal[count] = val;
                    mColIndex[count] = j;
                    ++count;
                }
            }
            mRowPtr[i + 1] = count;  // i行目までの非ゼロ要素数
        }
    }

    // CSR形式の疎行列とベクトルの積を計算する関数
    // vec: 列数と同じ長さの配列
    // 戻り値: 行数分の結果を格納した新たに確保した配列（呼び出し側で delete[] する必要あり）
    T* multiply(const T* vec) const {
        T* result = new T[mRowNum];
        for (int i = 0; i < mRowNum; ++i) {
            result[i] = 0.0;
            // i行目の非ゼロ要素は、mRowPtr[i] から mRowPtr[i+1]-1 まで
            for (int idx = mRowPtr[i]; idx < mRowPtr[i + 1]; ++idx) {
                result[i] += mVal[idx] * vec[mColIndex[idx]];
            }
        }
        return result;
    }

    // CSR形式の内部データを出力する関数
    void printCSR() const {
        std::printf("mVal: ");
        for (int i = 0; i < mNonZeroNum; ++i)
            std::printf("%f ", mVal[i]);  // Tがdoubleの場合は問題ありません
        std::printf("\n");

        std::printf("mColIndex: ");
        for (int i = 0; i < mNonZeroNum; ++i)
            std::printf("%d ", mColIndex[i]);
        std::printf("\n");

        std::printf("mRowPtr: ");
        for (int i = 0; i < mRowNum + 1; ++i)
            std::printf("%d ", mRowPtr[i]);
        std::printf("\n");
    }
};

int main() {
    // サンプルの密行列（3行4列）を一次元配列（row-major order）として定義
    // 例として、以下の行列:
    //    1 0 0 2
    //    0 3 0 0
    //    4 0 5 6
    const int mRowNum = 3;
    const int mColNum = 4;
    float dense[mRowNum * mColNum] = {
         1, 0, 0, 2,
         0, 3, 0, 0,
         4, 0, 5, 6
    };

    // CSR形式の疎行列オブジェクトを作成し、密行列から変換する
    CSRMatrix<float> csr(mRowNum, mColNum);
    csr.buildFromDense(dense);

    std::printf("CSR形式の内部データ:\n");
    csr.printCSR();

    // 行列とベクトルの積を計算
    // 例として、ベクトル v = [1, 1, 1, 1]
    float vec[mColNum] = {1, 1, 1, 1};
    float* result = csr.multiply(vec);

    std::printf("行列とベクトルの積:\n");
    for (int i = 0; i < mRowNum; ++i) {
        std::printf("%f ", result[i]);
    }
    std::printf("\n");

    // multiplyで確保したメモリを解放
    delete[] result;

    return 0;
}
