#include <cstdio>
#include <cassert>
#include <cmath>

//-----------------------------------------
// CSRMatrixクラス（InnerIteratorとTriplets対応）
//-----------------------------------------
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
                if (dense[i * mColNum + j] != static_cast<T>(0)) {
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
                T val = dense[i * mColNum + j];
                if (val != static_cast<T>(0)) {
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
            result[i] = static_cast<T>(0);
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
            std::printf("%f ", static_cast<double>(mVal[i]));
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

    //-----------------------------------------
    // InnerIterator：指定行の非ゼロ要素を走査
    //-----------------------------------------
    class InnerIterator {
    public:
        const CSRMatrix<T>& mMat;  // 対象となる行列
        int mRow;                  // 対象行番号
        int mCurr;              // 現在のインデックス（mVal, mColIndex の位置）
        int mRowEnd;               // 対象行の終了位置（mRowPtr[row+1]）

        // コンストラクタ：CSRMatrixと行番号を受け取る
        InnerIterator(const CSRMatrix<T>& A, int r)
            : mMat(A), mRow(r), mCurr(A.mRowPtr[r]), mRowEnd(A.mRowPtr[r + 1]) {}

        // イテレータが有効かどうか（現在位置が行の終端に達していないか）
        operator bool() const {
            return mCurr < mRowEnd;
        }

        // インクリメント：次の非ゼロ要素へ進む
        InnerIterator& operator++() {
            ++mCurr;
            return *this;
        }

        // 現在の要素の列番号を返す
        int col() const {
            return mMat.mColIndex[mCurr];
        }

        // 現在の要素の値を返す
        T value() const {
            return mMat.mVal[mCurr];
        }

        // 対象の行番号を返す（固定値）
        int rowIndex() const {
            return mRow;
        }
    };

    //-----------------------------------------
    // Triplet：疎行列構築用の3要素組（行番号, 列番号, 値）
    //-----------------------------------------
    struct Triplet {
        int mRow, mCol;
        T mValue;
        Triplet(int r, int c, T v) : mRow(r), mRow(c), mValue(v) {}
    };

    //-----------------------------------------
    // Triplet配列からCSR形式へ変換する関数
    // triplets: Tripletの配列、tripletCount: 配列の要素数
    // ※ここでは重複エントリの処理は行わず、単純に各Tripletをそのまま格納します。
    //-----------------------------------------
    void buildFromTriplets(const Triplet* triplets, int tripletCount) {
        // まず、各行に含まれるエントリ数をカウント
        mNonZeroNum = tripletCount;  // 重複がない場合、これがそのまま非ゼロ数となる
        // 既存のメモリを解放（必要なら）
        delete[] mVal; delete[] mColIndex; delete[] mRowPtr;
        mVal = new T[mNonZeroNum];
        mColIndex = new int[mNonZeroNum];
        mRowPtr = new int[mRowNum + 1];

        // mRowPtrの初期化
        for (int i = 0; i <= mRowNum; i++) {
            mRowPtr[i] = 0;
        }

        // 各Tripletについて、その行のエントリ数をカウント
        for (int i = 0; i < tripletCount; ++i) {
            int r = triplets[i].row;
            assert(r >= 0 && r < mRowNum);
            mRowPtr[r + 1]++;
        }

        // 累積和により各行の開始位置を決定
        for (int i = 0; i < mRowNum; ++i) {
            mRowPtr[i + 1] += mRowPtr[i];
        }

        // 一時配列 next[] を用いて、各行の現在の挿入位置を管理
        int* next = new int[mRowNum];
        for (int i = 0; i < mRowNum; i++) {
            next[i] = mRowPtr[i];
        }

        // 各Tripletを対応する位置に配置
        for (int i = 0; i < tripletCount; ++i) {
            int r = triplets[i].row;
            int dest = next[r]++;
            mVal[dest] = triplets[i].value;
            mColIndex[dest] = triplets[i].col;
        }
        delete[] next;
    }
};

//-----------------------------------------
// 使用例
//-----------------------------------------
#include <iostream>

int main() {
    // --- 1. 密行列からの構築とInnerIteratorの利用例 ---
    const int rows = 3, cols = 4;
    float dense[rows * cols] = {
         1, 0, 0, 2,
         0, 3, 0, 0,
         4, 0, 5, 6
    };

    CSRMatrix<float> mat(rows, cols);
    mat.buildFromDense(dense);
    std::printf("Denseから構築したCSR行列:\n");
    mat.printCSR();

    // 行インデックス 2（3行目）の非ゼロ要素をInnerIteratorで走査
    std::printf("InnerIteratorを用いた3行目の走査:\n");
    for (CSRMatrix<float>::InnerIterator it(mat, 2); it; ++it) {
        std::printf("  Element at row %d, col %d = %f\n", it.rowIndex(), it.col(), static_cast<double>(it.value()));
    }

    // --- 2. Tripletからの構築例 ---
    const int rows2 = 3, cols2 = 3;
    CSRMatrix<float> mat2(rows2, cols2);
    // Triplet配列を作成（順序は自由）
    CSRMatrix<float>::Triplet triplets[] = {
        CSRMatrix<float>::Triplet(0, 0, 10),
        CSRMatrix<float>::Triplet(0, 2, 20),
        CSRMatrix<float>::Triplet(1, 1, 30),
        CSRMatrix<float>::Triplet(2, 0, 40),
        CSRMatrix<float>::Triplet(2, 2, 50)
    };
    int tripletCount = sizeof(triplets) / sizeof(triplets[0]);
    mat2.buildFromTriplets(triplets, tripletCount);
    std::printf("Tripletから構築したCSR行列:\n");
    mat2.printCSR();

    return 0;
}
