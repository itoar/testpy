int pointCount = npoints(1);
for (int i = 0; i < pointCount; i++)
{
    // Input1から対象プリミティブの"myattr"属性値を取得
    int prim_idx = point(1, "prim", i);
    int cluster = point(1, "cluster", i);
    // Input0のプリミティブに"myattr"属性値を書き込み
    setprimattrib(0, "cluster", prim_idx, cluster, "set");
}


/////////////////////////////


// 現在のプリミティブ番号
int prim = @primnum;

// 現在のプリミティブの点リスト（順序付き）
int pts[] = primpoints(0, prim);
int npts = len(pts);

// 隣接プリミティブ番号を格納する配列（エッジを共有しているもの）
int adjacent_prims[];

// プリミティブの各エッジについて処理（最後の点と最初の点もエッジ）
for (int i = 0; i < npts; i++)
{
    int pt0 = pts[i];
    int pt1 = pts[(i+1) % npts];
    
    // 各点を共有しているプリミティブを取得
    int prims0[] = pointprims(0, pt0);
    int prims1[] = pointprims(0, pt1);
    
    // pt0に関して、pt1も共有している場合＝エッジを共有している
    foreach (int p; prims0)
    {
         if (p != prim && find(prims1, p) >= 0 && find(adjacent_prims, p) < 0)
         {
              append(adjacent_prims, p);
         }
    }
}

// 隣接プリミティブの数を保存
int adjacent_count = len(adjacent_prims);
setprimattrib(0, "adjacent", prim, adjacent_prims, "set");
setprimattrib(0, "adjacent_count", prim, adjacent_count, "set");

// 現在のプリミティブの法線を取得（あらかじめ法線属性 "N" がある前提）
vector myN = prim(0, "N", prim);

// 隣接プリミティブの法線と内積が0.8より大きいものをカウント
int similar_count = 0;
float temp = 0.0;
foreach (int adj; adjacent_prims)
{
    vector adjN = prim(0, "N", adj);
    float dp = dot(myN, adjN);
    if (dp > 0.95)
    {
         similar_count++;
    }
}
setprimattrib(0, "similar_adjacent_count", prim, similar_count, "set");
setprimattrib(0, "dp", prim, temp, "set");


///////////////////////


// クラスタリングのためのラベル伝搬法 (Detail Wrangle)
// 全プリミティブ数
int npr = nprimitives(0);

// 各プリミティブのラベルを格納する配列（初期値0：未割り当て）
int labels[] = array();
for (int i = 0; i < npr; i++){
    append(labels, 0);
}

// 各プリミティブの重心Y座標を計算し、配列に格納
float primY[] = array();
for (int i = 0; i < npr; i++){
    int pts[] = primpoints(0, i);
    vector centroid = {0,0,0};
    foreach (int p; pts)
        centroid += point(0, "P", p);
    centroid /= len(pts);
    append(primY, centroid.y);
}

// メッシュ全体の上端と下端のY座標を取得
float maxY = primY[0];
float minY = primY[0];
for (int i = 0; i < npr; i++){
    if (primY[i] > maxY) maxY = primY[i];
    if (primY[i] < minY) minY = primY[i];
}

// 上端・下端を判定するための許容値（全体の高さの1%程度）
float tol = 0.01 * (maxY - minY);

// 初期ラベルの設定：上端に近いプリミティブは1、下端に近いプリミティブは2
for (int i = 0; i < npr; i++){
    if (abs(primY[i] - maxY) < tol)
        labels[i] = 1;
    else if (abs(primY[i] - minY) < tol)
        labels[i] = 2;
    // その他は0 (未割り当て)
}

// ラベル伝搬の反復回数（必要に応じて調整）
int iterations = 100;

// 反復ループによるラベル伝搬
for (int iter = 0; iter < iterations; iter++){
    int newLabels[] = labels; // 現在のラベルをコピーして更新用に使用
    // 各プリミティブについて処理
    for (int i = 0; i < npr; i++){
        // 既に初期シードとして設定されている場合は変更しない
        if (labels[i] != 0)
            continue;
        
        // 現在のプリミティブの法線 (属性"N"がある前提)
        vector n_i = prim(0, "N", i);
        
        // 隣接プリミティブリスト ("adjacent"属性) を取得
        int adj[] = prim(0, "adjacent", i);
        
        // 隣接プリミティブから、法線の内積(dp)が0.8より大きいもののラベルを票としてカウント
        int vote1 = 0;
        int vote2 = 0;
        foreach (int j; adj){
            vector n_j = prim(0, "N", j);
            float dp = dot(n_i, n_j);
            if (dp > 0.8) {
                int lab = labels[j];
                if (lab == 1)
                    vote1++;
                else if (lab == 2)
                    vote2++;
            }
        }
        // 多数決でラベルを決定（票がなければ変更なし）
        if (vote1 > vote2)
            newLabels[i] = 1;
        else if (vote2 > vote1)
            newLabels[i] = 2;
    }
    // 更新
    labels = newLabels; 
}

// 最終的なクラスタラベルを "cluster" というプリミティブアトリビュートに保存
for (int i = 0; i < npr; i++){
    setprimattrib(0, "cluster2", i, labels[i], "set");
}



//
// 現在のプリミティブのラベルを取得
int myLabel = prim(0, "cluster2", @primnum);

// 現在のプリミティブに含まれる点（順序付き）を取得
int pts[] = primpoints(0, @primnum);
int npts = len(pts);

// 各エッジ（連続する2点）についてループ
for (int i = 0; i < npts; i++)
{
    // エッジの両端の点
    int pt0 = pts[i];
    int pt1 = pts[(i+1)%npts];
    
    // pt0とpt1それぞれを共有しているプリミティブを取得
    int prims_pt0[] = pointprims(0, pt0);
    int prims_pt1[] = pointprims(0, pt1);
    
    // このエッジを共有している隣接プリミティブのうち、
    // ラベルが異なるものがあるかを調べるフラグ
    int different = 0;
    
    // pt0を使っているプリミティブをチェック
    foreach (int p; prims_pt0)
    {
         // 自身は除外
         if (p == @primnum)
             continue;
         
         // pt1も共有していれば、エッジを共有している
         if (find(prims_pt1, p) >= 0)
         {
              // 隣接プリミティブのラベルを取得し、異なる場合フラグを立てる
              int neighborLabel = prim(0, "cluster2", p);
              if (neighborLabel != myLabel)
              {
                  different = 1;
                  break;
              }
         }
    }
    
    // 隣接するプリミティブの中にラベルが異なるものがあれば、このエッジをエッジグループ "diff_edge" に追加
    if (different)
    {
         // setedgegroup(geometry, groupname, primnum, edgeindex, value)
         // エッジはプリミティブ内で、各頂点の開始位置をエッジインデックスとして扱えます
         setedgegroup(0, "diff_edge2", pt0, pt1, 1);
    }
}
