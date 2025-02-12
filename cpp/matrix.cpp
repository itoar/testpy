#include <iostream>
#include <cassert>
#include <utility>  // std::move のため

using TSize = std::size_t;

// 正方行列専用クラス
// テンプレートパラメータ T: 要素の型, N: 行数（＝列数）
template<typename T, TSize N>
class Matrix {
public:
    static constexpr TSize NumElements = N * N;
    
    // デフォルトコンストラクタ：全要素を T() で初期化
    Matrix() {
        for (TSize i = 0; i < NumElements; ++i) {
            buffer[i] = T();
        }
    }
    
    // コピーコンストラクタ
    Matrix(const Matrix<T, N>& other) {
        for (TSize i = 0; i < NumElements; ++i) {
            buffer[i] = other.buffer[i];
        }
    }
    
    // コピー代入演算子
    Matrix<T, N>& operator=(const Matrix<T, N>& other) {
        if (this != &other) {
            for (TSize i = 0; i < NumElements; ++i) {
                buffer[i] = other.buffer[i];
            }
        }
        return *this;
    }
        
    // デストラクタ
    ~Matrix() {
        // 固定サイズ配列のため特に解放処理は不要
    }
    
    // 書き込み用アクセス演算子：行と列を指定して要素にアクセス
    T& operator()(TSize row, TSize col) {
        assert(row < N && col < N);
        return buffer[row * N + col];
    }
    
    // 読み出し用アクセス演算子：constオブジェクト用
    const T& operator()(TSize row, TSize col) const {
        assert(row < N && col < N);
        return buffer[row * N + col];
    }
        
    // 行列のサイズ（正方行列なので辺の長さ）を返す
    TSize size() const {
        return N;
    }
    
    // 全要素をゼロに設定する
    void makeZero() {
        for (TSize i = 0; i < NumElements; ++i) {
            buffer[i] = T(0);
        }
    }
    
    // 単位行列にする API
    // 対角成分を 1, それ以外を 0 に設定する
    void makeIdentity() {
        for (TSize i = 0; i < N; ++i) {
            for (TSize j = 0; j < N; ++j) {
                (*this)(i, j) = (i == j) ? T(1) : T(0);
            }
        }
    }
    
    // 行列積を計算する API
    // サイズが異なる matrix が入力された場合はアサートでチェック（ここでは N が固定のため通常ありえません）
    Matrix<T, N> multiply(const Matrix<T, N>& other) const {
        assert(this->size() == other.size());
        Matrix<T, N> result;
        result.makeZero();
        for (TSize i = 0; i < N; ++i) {
            for (TSize j = 0; j < N; ++j) {
                for (TSize k = 0; k < N; ++k) {
                    result(i, j) += (*this)(i, k) * other(k, j);
                }
            }
        }
        return result;
    }
    
    // + 演算子のオーバーロード（行列の要素ごとの加算）
    Matrix<T, N> operator+(const Matrix<T, N>& other) const {
        Matrix<T, N> result;
        for (TSize i = 0; i < NumElements; ++i) {
            result.buffer[i] = buffer[i] + other.buffer[i];
        }
        return result;
    }
    
    // - 演算子のオーバーロード（行列の要素ごとの減算）
    Matrix<T, N> operator-(const Matrix<T, N>& other) const {
        Matrix<T, N> result;
        for (TSize i = 0; i < NumElements; ++i) {
            result.buffer[i] = buffer[i] - other.buffer[i];
        }
        return result;
    }
    
    // transpose() の API
    // 行と列を入れ替えた転置行列を返す
    Matrix<T, N> transpose() const {
        Matrix<T, N> result;
        for (TSize i = 0; i < N; ++i) {
            for (TSize j = 0; j < N; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }
    
private:
    // 固定サイズの内部バッファ。サイズは NumElements
    T buffer[NumElements];
};

// 行列の内容をプリントするヘルパー関数
template<typename T, TSize N>
void printMatrix(const Matrix<T, N>& mat) {
    for (TSize i = 0; i < mat.size(); ++i) {
        for (TSize j = 0; j < mat.size(); ++j) {
            std::cout << mat(i, j) << "\t";
        }
        std::cout << std::endl;
    }
}

//
// テスト用 main 関数
//
int main() {
    const TSize dim = 10;
    std::cout << "=== Matrix テスト開始 ===" << std::endl;
    
    // 1. デフォルトコンストラクタのテスト (初期状態は全要素 T()、ここでは 0)
    Matrix<int, dim> A;
    std::cout << "\n-- A: デフォルトコンストラクタで作成（初期値は 0） --" << std::endl;
    printMatrix(A);
    
    // 2. makeZero() のテスト
    // ※既に全要素 0 だが、念のため makeZero() を呼び出す
    A.makeZero();
    std::cout << "\n-- A: makeZero() 呼び出し後 --" << std::endl;
    printMatrix(A);
    
    // 3. makeIdentity() のテスト
    A.makeIdentity();
    std::cout << "\n-- A: makeIdentity() 呼び出し後 --" << std::endl;
    printMatrix(A);
    
    // 4. 明示的に値を設定した行列 B を作成 (例: 1,2,...,9)
    Matrix<int, dim> B;
    int value = 1;
    for (TSize i = 0; i < B.size(); ++i) {
        for (TSize j = 0; j < B.size(); ++j) {
            B(i, j) = value++;
        }
    }
    std::cout << "\n-- B: 手動で値を設定 (1～9) --" << std::endl;
    printMatrix(B);
    
    // 5. operator+ のテスト (C = A + B)
    Matrix<int, dim> C = A + B;
    std::cout << "\n-- C = A + B --" << std::endl;
    printMatrix(C);
    
    // 6. operator- のテスト (D = B - A)
    Matrix<int, dim> D = B - A;
    std::cout << "\n-- D = B - A --" << std::endl;
    printMatrix(D);
    
    // 7. multiply() のテスト (E = A.multiply(B))
    //  A は単位行列なので、E は B と同じはず
    Matrix<int, dim> E = A.multiply(B);
    std::cout << "\n-- E = A.multiply(B) (A は単位行列なので E は B と同じ) --" << std::endl;
    printMatrix(E);
    
    // 8. コピーコンストラクタのテスト (F を B からコピー)
    Matrix<int, dim> F(B);
    std::cout << "\n-- F: コピーコンストラクタで作成 (F = B) --" << std::endl;
    printMatrix(F);
    
    // 9. コピー代入演算子のテスト (G に B を代入)
    Matrix<int, dim> G;
    G = B;
    std::cout << "\n-- G: コピー代入演算子で作成 (G = B) --" << std::endl;
    printMatrix(G);
    
    // 10. transpose() のテスト (H = B.transpose())
    Matrix<int, dim> H = B.transpose();
    std::cout << "\n-- H = B.transpose() --" << std::endl;
    printMatrix(H);
    
    std::cout << "\n=== Matrix テスト終了 ===" << std::endl;
    return 0;
}
