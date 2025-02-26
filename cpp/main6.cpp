#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cassert>

// 3D空間内の位置を表す構造体
struct Vector3 {
    float x, y, z;
};

// Point 構造体（ランダムな位置と一意のIDを持つ）
struct Point {
    float x;
    float y;
    float z;
    int   id;
};

// 3D用のセルキー。グリッド上の整数座標で表現する。
struct SpatialHashKey3D {
    int x, y, z;
    bool operator==(const SpatialHashKey3D &other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

// ユーザー指定のハッシュ関数
// hash = (xi * 92837111) ^ (yi * 689287499) ^ (zi * 283923481)
// 絶対値を返す
inline int hashCoords(int xi, int yi, int zi) {
    int hash = (xi * 92837111) ^ (yi * 689287499) ^ (zi * 283923481);
    return hash < 0 ? -hash : hash;
}

// --- 固定容量バッファ ---
// オブジェクトへのポインタを格納するシンプルなコンテナです。
template<typename T>
class BufferVector {
public:
    T* buffer;       // 外部で確保されたメモリ領域へのポインタ（T はここではポインタ型）
    size_t capacity; // 格納可能な最大要素数
    size_t size;     // 現在の格納数

    BufferVector() : buffer(nullptr), capacity(0), size(0) {}

    // 要素を追加。成功すれば true を返す。
    bool push_back(const T &value) {
        if (size < capacity) {
            buffer[size] = value;
            ++size;
            return true;
        }
        return false;
    }
    
    // ポインタ同士の比較で対象を削除
    bool remove(const T &value) {
        for (size_t i = 0; i < size; i++) {
            if (buffer[i] == value) { // ポインタの等価性で比較
                buffer[i] = buffer[size - 1];
                --size;
                return true;
            }
        }
        return false;
    }
};

// --- 自前のハッシュテーブルを用いた SpatialHash3D ---
// T：セルに格納するオブジェクトの型（ここでは Point* を想定）
template<typename T>
class SpatialHash3D {
private:
    float cellSize;         // セルのサイズ（空間を cellSize 単位のグリッドに分割）
    size_t maxCells;        // 事前に確保するセル数
    size_t maxItemsPerCell; // 1セルあたりの最大格納可能数
    size_t nextCellIndex;   // 事前確保セルの次に使用するインデックス

    // 事前確保されたセル用バッファ（外部メモリ）
    BufferVector<T>* cellPool;
    // 事前確保された全セル分のアイテム格納用メモリ。サイズは (maxCells * maxItemsPerCell)
    T* itemsBuffer;
    
    // ハッシュテーブルのバケット数
    size_t tableSize;
    // 各エントリはセルキーと、そのセルへのポインタを保持する
    struct HashEntry {
         SpatialHashKey3D key;
         BufferVector<T>* value;
         HashEntry* next;
    };
    // ハッシュテーブルは HashEntry* の配列（各要素は連結リストの先頭ポインタ）
    HashEntry** table;

    // 新しいハッシュエントリを生成
    HashEntry* createHashEntry(const SpatialHashKey3D &key, BufferVector<T>* value) {
         HashEntry* entry = new HashEntry;
         entry->key = key;
         entry->value = value;
         entry->next = nullptr;
         return entry;
    }

public:
    // コンストラクタ
    // cellPool: 事前に maxCells 個分の BufferVector<T> 配列（外部で確保済み）
    // itemsBuffer: 事前に maxCells * maxItemsPerCell 個分の T 配列（T は Point* の配列となる）
    // tableSize: ハッシュテーブルのバケット数（例：素数など適切な値）
    SpatialHash3D(float cellSize, size_t maxCells, size_t maxItemsPerCell,
                  BufferVector<T>* cellPool, T* itemsBuffer, size_t tableSize)
        : cellSize(cellSize), maxCells(maxCells), maxItemsPerCell(maxItemsPerCell),
          nextCellIndex(0), cellPool(cellPool), itemsBuffer(itemsBuffer),
          tableSize(tableSize)
    {
         table = new HashEntry*[tableSize];
         for (size_t i = 0; i < tableSize; i++)
             table[i] = nullptr;
    }
    
    // デストラクタ：ハッシュテーブル内のエントリを解放する
    ~SpatialHash3D() {
         for (size_t i = 0; i < tableSize; i++) {
             HashEntry* entry = table[i];
             while(entry) {
                 HashEntry* next = entry->next;
                 delete entry;
                 entry = next;
             }
         }
         delete[] table;
         // 動的にセルを確保しないため、動的セル管理は不要
    }

    // 入力位置からセルキーを算出する
    SpatialHashKey3D getKey(const Vector3 &pos) const {
         SpatialHashKey3D key;
         key.x = static_cast<int>(std::floor(pos.x / cellSize));
         key.y = static_cast<int>(std::floor(pos.y / cellSize));
         key.z = static_cast<int>(std::floor(pos.z / cellSize));
         return key;
    }

    // ハッシュテーブル内から指定キーに対応するセルを検索
    BufferVector<T>* findCell(const SpatialHashKey3D &key) {
         int h = hashCoords(key.x, key.y, key.z);
         size_t index = h % tableSize;
         HashEntry* entry = table[index];
         while(entry) {
             if (entry->key == key)
                return entry->value;
             entry = entry->next;
         }
         return nullptr;
    }

    // ハッシュテーブルに新たなセルを登録
    BufferVector<T>* insertCell(const SpatialHashKey3D &key, BufferVector<T>* cell) {
         int h = hashCoords(key.x, key.y, key.z);
         size_t index = h % tableSize;
         HashEntry* newEntry = createHashEntry(key, cell);
         newEntry->next = table[index];
         table[index] = newEntry;
         return cell;
    }

    // オブジェクト（ポインタ）を指定位置に挿入する。成功すれば true、失敗なら false を返す。
    bool insert(const T &obj, const Vector3 &pos) {
         SpatialHashKey3D key = getKey(pos);
         BufferVector<T>* cell = findCell(key);
         if (!cell) {
              // セルが存在しなければ新たに作成
              assert(nextCellIndex < maxCells && "maxCells exceeded");
              cell = &cellPool[nextCellIndex];
              cell->buffer   = itemsBuffer + (nextCellIndex * maxItemsPerCell);
              cell->capacity = maxItemsPerCell;
              cell->size     = 0;
              nextCellIndex++;
              insertCell(key, cell);
         }
         return cell->push_back(obj);
    }

    // 指定位置に対応するセルからオブジェクト（ポインタ）を削除する
    bool remove(const T &obj, const Vector3 &pos) {
         SpatialHashKey3D key = getKey(pos);
         BufferVector<T>* cell = findCell(key);
         if (!cell)
             return false;
         return cell->remove(obj);
    }

    // オブジェクト（ポインタ）の位置更新
    // 旧位置 oldPos と新位置 newPos に基づいて、セルが異なる場合は旧セルから削除し、新セルに挿入します。
    bool update(const T &obj, const Vector3 &oldPos, const Vector3 &newPos) {
         SpatialHashKey3D oldKey = getKey(oldPos);
         SpatialHashKey3D newKey = getKey(newPos);
         if (oldKey == newKey) {
              // 同じセル内なら更新の必要はありません（ポインタ自体は変わらないため）
              return true;
         } else {
              bool removed = remove(obj, oldPos);
              bool inserted = insert(obj, newPos);
              return removed && inserted;
         }
    }

    // 指定位置に対応するセル内のデータを返す（セルが存在しなければ nullptr を返す）
    const BufferVector<T>* query(const Vector3 &pos) const {
         SpatialHashKey3D key = getKey(pos);
         int h = hashCoords(key.x, key.y, key.z);
         size_t index = h % tableSize;
         HashEntry* entry = table[index];
         while(entry) {
             if (entry->key == key)
                 return entry->value;
             entry = entry->next;
         }
         return nullptr;
    }
};

//
// --- 使用例 ---
// ランダムな位置を持つ 100 個の Point を生成し、SpatialHash3D<Point*> に格納。
// その後、特定のポイントの位置更新（移動）も実演します。
//
int main() {
    std::srand(static_cast<unsigned int>(std::time(0)));
    const size_t numPoints = 100;
    Point points[numPoints];

    // ランダムな座標（0～100）と ID を設定
    for (size_t i = 0; i < numPoints; i++) {
        points[i].x = static_cast<float>(std::rand() % 101);
        points[i].y = static_cast<float>(std::rand() % 101);
        points[i].z = static_cast<float>(std::rand() % 101);
        points[i].id = static_cast<int>(i);
    }

    // SpatialHash3D のパラメータ
    const float cellSize = 50.0f;
    const size_t maxCells = 200;          // 事前に確保するセル数
    const size_t maxItemsPerCell = 10;     // 1セルあたりの最大アイテム数
    const size_t tableSize = 53;           // ハッシュテーブルのバケット数（例：素数）

    // 事前確保メモリの確保
    BufferVector<Point*>* cellPool = new BufferVector<Point*>[maxCells];
    // itemsBuffer は、各セルが格納する Point* の配列（全体では maxCells * maxItemsPerCell 個分）
    Point** itemsBuffer = new Point*[maxCells * maxItemsPerCell];

    // SpatialHash3D<Point*> のインスタンス生成
    SpatialHash3D<Point*> spatialHash(cellSize, maxCells, maxItemsPerCell, cellPool, itemsBuffer, tableSize);

    // 100 個の Point のアドレス（ポインタ）を SpatialHash3D に挿入
    for (size_t i = 0; i < numPoints; i++) {
         Vector3 pos = { points[i].x, points[i].y, points[i].z };
         spatialHash.insert(&points[i], pos);
    }

    // 例: ポイント 0 の位置を更新（移動）
    Point* p = &points[0];
    Vector3 oldPos = { p->x, p->y, p->z };
    // ここでは、x 座標に +15、y 座標に +5、z 座標に +10 移動する例
    p->x += 15.0f;
    p->y += 5.0f;
    p->z += 10.0f;
    Vector3 newPos = { p->x, p->y, p->z };
    bool updated = spatialHash.update(p, oldPos, newPos);
    std::cout << "Point pointer " << p << " update " << (updated ? "succeeded." : "failed.") << std::endl;

    // 例として、(50,50,50) 周辺のセルの内容を出力
    Vector3 queryPos = {50.0f, 50.0f, 50.0f};
    const BufferVector<Point*>* cell = spatialHash.query(queryPos);
    if (cell) {
         std::cout << "Cell (" << queryPos.x << ", " << queryPos.y << ", " << queryPos.z 
                   << ") contains " << cell->size << " Point pointers:" << std::endl;
         for (size_t i = 0; i < cell->size; i++) {
              Point* pt = cell->buffer[i];
              std::cout << "  ID:" << pt->id << " (" << pt->x << ", " << pt->y << ", " << pt->z << ")" << std::endl;
         }
    } else {
         std::cout << "No cell at query position (" << queryPos.x << ", " << queryPos.y << ", " << queryPos.z << ")." << std::endl;
    }

    // 使用後は、事前確保したメモリを解放（動的セルは assert により使用されない）
    delete[] cellPool;
    delete[] itemsBuffer;

    return 0;
}
