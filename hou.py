int pointCount = npoints(1);
for (int i = 0; i < pointCount; i++)
{
    // Input1から対象プリミティブの"myattr"属性値を取得
    int prim_idx = point(1, "prim", i);
    int cluster = point(1, "cluster", i);
    // Input0のプリミティブに"myattr"属性値を書き込み
    setprimattrib(0, "cluster", prim_idx, cluster, "set");
}