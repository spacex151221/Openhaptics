#include <windows.h> 
#include <HD/hd.h>
#include <math.h>
#include <algorithm> // std::nth_element, std::sort
#include <vector>
#include <stdio.h>

// [AVX2 & OpenMP] 헤더 포함
#include <immintrin.h> 
#ifdef _OPENMP
#include <omp.h>
#endif

#define DllExport __declspec(dllexport)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --------------------------------------------------------
// [Data Structures]
// --------------------------------------------------------
struct Vec3 { double x, y, z; };

// [Modified] Centroid for BVH balancing
struct Triangle { 
    Vec3 v0, v1, v2; 
    int id; 
    Vec3 centroid; 
};

// [NEW] KD-Tree Point Structure
struct KDPoint {
    Vec3 pos;
    int triIdx; // 이 정점이 속한 삼각형 인덱스
    int id;     // 고유 ID
};

// [NEW] KD-Tree Node
struct KDNode {
    int left = -1;
    int right = -1;
    KDPoint point;
    int axis;
    // Range Pruning을 위한 AABB
    Vec3 min, max; 
};

struct AABB {
    Vec3 min, max;
    void init() {
        min = { 1e20, 1e20, 1e20 };
        max = { -1e20, -1e20, -1e20 };
    }
    void expand(const Vec3& p) {
        if(p.x < min.x) min.x = p.x; if(p.y < min.y) min.y = p.y; if(p.z < min.z) min.z = p.z;
        if(p.x > max.x) max.x = p.x; if(p.y > max.y) max.y = p.y; if(p.z > max.z) max.z = p.z;
    }
};

struct BVHNode {
    AABB box;
    BVHNode* left = nullptr;
    BVHNode* right = nullptr;
    std::vector<int> triIndices;
    bool isLeaf = false;
};

struct FlatBVHNode {
    AABB box;
    int rightChildIdx; 
    int triStart;      
    int triCount;      
};

struct DeviceData {
    double posX, posY, posZ;      
    double rotYaw, rotPitch, rotRoll; 
    double jointA1;
    double planeAngle;
    int buttons; 
};

// --------------------------------------------------------
// [Global Variables]
// --------------------------------------------------------
HHD hHD = HD_INVALID_HANDLE;
HDSchedulerHandle gCallback = HD_INVALID_HANDLE;
DeviceData gCurrentData = { 0 };

double gOffsetX = 0.0, gOffsetY = 0.0, gOffsetZ = 0.0;
double gStiffness = 0.3, gRadius = 40.0, gGuideForce = 0.8; 
bool gForceMode = false, gTrackMode = false;  

// [Thread Safety] 데이터 교체 중 접근 방지 플래그
volatile bool gIsDataReady = false;

double* gMeshVertices = NULL;
int gTriCount = 0;
double gLastForce[3] = {0, 0, 0}; 

// BVH Data
std::vector<Triangle> gTriangles;
std::vector<Triangle> gOrderedTriangles; 
std::vector<FlatBVHNode> gFlatNodes;     

// [NEW] KD-Tree Data
std::vector<KDNode> gKDTree;
std::vector<KDPoint> gKDPoints; 

// --------------------------------------------------------
// [Math Helpers] - AVX2 Optimized
// --------------------------------------------------------
inline Vec3 sub(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 mul(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline double lenSq(Vec3 a) { return dot(a, a); }

inline double distSqPointAABB_AVX(Vec3 p, const AABB& b) {
    __m256d p_vec = _mm256_set_pd(0.0, p.z, p.y, p.x);
    __m256d min_vec = _mm256_set_pd(0.0, b.min.z, b.min.y, b.min.x);
    __m256d max_vec = _mm256_set_pd(0.0, b.max.z, b.max.y, b.max.x);
    __m256d d1 = _mm256_sub_pd(min_vec, p_vec);
    __m256d d2 = _mm256_sub_pd(p_vec, max_vec);
    __m256d zero = _mm256_setzero_pd();
    d1 = _mm256_max_pd(d1, zero);
    d2 = _mm256_max_pd(d2, zero);
    __m256d dist = _mm256_add_pd(d1, d2);
    __m256d sqDist = _mm256_mul_pd(dist, dist);
    alignas(32) double temp[4];
    _mm256_store_pd(temp, sqDist);
    return temp[0] + temp[1] + temp[2];
}

// KD-Tree용 단순 AABB 거리 계산
inline double distSqPointBox(const Vec3& p, const Vec3& min, const Vec3& max) {
    double d = 0.0;
    if (p.x < min.x) d += (min.x - p.x) * (min.x - p.x);
    if (p.x > max.x) d += (p.x - max.x) * (p.x - max.x);
    if (p.y < min.y) d += (min.y - p.y) * (min.y - p.y);
    if (p.y > max.y) d += (p.y - max.y) * (p.y - max.y);
    if (p.z < min.z) d += (min.z - p.z) * (min.z - p.z);
    if (p.z > max.z) d += (p.z - max.z) * (p.z - max.z);
    return d;
}

Vec3 closestPointOnTriangle(Vec3 p, Vec3 a, Vec3 b, Vec3 c) {
    Vec3 ab = sub(b, a); Vec3 ac = sub(c, a); Vec3 ap = sub(p, a);
    double d1 = dot(ab, ap); double d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return a;
    Vec3 bp = sub(p, b); double d3 = dot(ab, bp); double d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return b;
    double vc = d1*d4 - d3*d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return add(a, mul(ab, d1 / (d1 - d3)));
    Vec3 cp = sub(p, c); double d5 = dot(ab, cp); double d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return c;
    double vb = d5*d2 - d1*d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return add(a, mul(ac, d2 / (d2 - d6)));
    double va = d3*d6 - d5*d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) return add(b, mul(sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))));
    double denom = 1.0 / (va + vb + vc);
    return add(a, add(mul(ab, vb * denom), mul(ac, vc * denom)));
}

// [MISSING FIXED] 햅틱 루프에서 사용하는 삼각형 판별 함수 복구
bool isPointInTriangle(double* p, double* a, double* b, double* c) {
    double v0[3] = { c[0]-a[0], c[1]-a[1], c[2]-a[2] };
    double v1[3] = { b[0]-a[0], b[1]-a[1], b[2]-a[2] };
    double v2[3] = { p[0]-a[0], p[1]-a[1], p[2]-a[2] };
    double dot00 = v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2];
    double dot01 = v0[0]*v1[0] + v0[1]*v1[1] + v0[2]*v1[2];
    double dot02 = v0[0]*v2[0] + v0[1]*v2[1] + v0[2]*v2[2];
    double dot11 = v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2];
    double dot12 = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
    double invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    double u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    double v = (dot00 * dot12 - dot01 * dot02) * invDenom;
    return (u >= -0.1) && (v >= -0.1) && (u + v <= 1.1);
}

// [MISSING FIXED] 햅틱 장치 행렬 분해 함수 복구
void decomposeMatrix(const double* m) {
    gCurrentData.posX = m[12] - gOffsetX;
    gCurrentData.posY = m[13] - gOffsetY;
    gCurrentData.posZ = m[14] - gOffsetZ;
    double fX = -m[8]; double fY = m[9]; double fZ = m[10];
    double groundDist = sqrt(fX * fX + fZ * fZ);
    if (groundDist < 0.005) { gCurrentData.rotPitch = (fY >= 0) ? (M_PI / 2.0) : (-M_PI / 2.0); }
    else { gCurrentData.rotPitch = atan2(fY, groundDist); }
    double nX = fZ; double nZ = -fX;
    gCurrentData.planeAngle = atan2(nX, -nZ);
    gCurrentData.rotYaw = atan2(fX, -fZ);
    double joints[6]; hdGetDoublev(HD_CURRENT_JOINT_ANGLES, joints);
    gCurrentData.jointA1 = joints[0];
    if (fabs(m[9]) < 0.999) { gCurrentData.rotRoll = atan2(-m[1], m[5]); }
    else { gCurrentData.rotRoll = atan2(m[4], m[0]); }
}

// --------------------------------------------------------
// [NEW] KD-Tree Construction & Search
// --------------------------------------------------------
int buildKDTreeRec(std::vector<int>& indices, int depth) {
    if (indices.empty()) return -1;

    int axis = depth % 3;
    size_t mid = indices.size() / 2;

    // Median Split using nth_element (Fast O(N))
    std::nth_element(indices.begin(), indices.begin() + mid, indices.end(),
        [&](int a, int b) {
            if (axis == 0) return gKDPoints[a].pos.x < gKDPoints[b].pos.x;
            else if (axis == 1) return gKDPoints[a].pos.y < gKDPoints[b].pos.y;
            else return gKDPoints[a].pos.z < gKDPoints[b].pos.z;
        });

    int nodeIdx = (int)gKDTree.size();
    gKDTree.push_back({});
    KDNode& node = gKDTree[nodeIdx];
    
    node.point = gKDPoints[indices[mid]];
    node.axis = axis;
    
    // Compute AABB for Pruning
    node.min = {1e20, 1e20, 1e20};
    node.max = {-1e20, -1e20, -1e20};
    for(int idx : indices) {
        Vec3 p = gKDPoints[idx].pos;
        if(p.x < node.min.x) node.min.x = p.x; if(p.x > node.max.x) node.max.x = p.x;
        if(p.y < node.min.y) node.min.y = p.y; if(p.y > node.max.y) node.max.y = p.y;
        if(p.z < node.min.z) node.min.z = p.z; if(p.z > node.max.z) node.max.z = p.z;
    }

    std::vector<int> leftIndices(indices.begin(), indices.begin() + mid);
    std::vector<int> rightIndices(indices.begin() + mid + 1, indices.end());

    node.left = buildKDTreeRec(leftIndices, depth + 1);
    node.right = buildKDTreeRec(rightIndices, depth + 1);

    return nodeIdx;
}

// Find K Nearest Triangles (via Vertices)
void searchKDTree(int nodeIdx, const Vec3& target, int k, std::vector<int>& results, std::vector<double>& dists) {
    if (nodeIdx == -1) return;
    const KDNode& node = gKDTree[nodeIdx];

    // Pruning
    if (distSqPointBox(target, node.min, node.max) > (dists.size() == k ? dists.back() : 1e20)) return;

    double dSq = lenSq(sub(target, node.point.pos));
    
    // Insert Logic (Sorted)
    if (results.size() < k || dSq < dists.back()) {
        int i = 0;
        while(i < results.size() && dists[i] < dSq) i++;
        results.insert(results.begin() + i, node.point.triIdx);
        dists.insert(dists.begin() + i, dSq);
        if (results.size() > k) {
            results.pop_back();
            dists.pop_back();
        }
    }

    double diff = (node.axis == 0 ? target.x - node.point.pos.x : 
                  (node.axis == 1 ? target.y - node.point.pos.y : target.z - node.point.pos.z));

    int nearChild = diff < 0 ? node.left : node.right;
    int farChild = diff < 0 ? node.right : node.left;

    searchKDTree(nearChild, target, k, results, dists);
    
    if (diff * diff < (dists.size() == k ? dists.back() : 1e20)) {
        searchKDTree(farChild, target, k, results, dists);
    }
}

// --------------------------------------------------------
// [BVH Logic] - Object Median Split
// --------------------------------------------------------
void deleteBVH(BVHNode* node) {
    if (!node) return;
    deleteBVH(node->left); deleteBVH(node->right); delete node;
}

BVHNode* buildBVH(std::vector<int>& triIndices, int depth) {
    BVHNode* node = new BVHNode();
    node->box.init();
    for (int idx : triIndices) {
        Triangle& t = gTriangles[idx];
        node->box.expand(t.v0); node->box.expand(t.v1); node->box.expand(t.v2);
    }
    if (triIndices.size() <= 8) {
        node->isLeaf = true; node->triIndices = triIndices; return node;
    }
    Vec3 size = sub(node->box.max, node->box.min);
    int axis = (size.y > size.x && size.y > size.z) ? 1 : (size.z > size.x && size.z > size.y ? 2 : 0);
    size_t mid = triIndices.size() / 2;
    std::nth_element(triIndices.begin(), triIndices.begin() + mid, triIndices.end(),
        [&](int a, int b) {
            Vec3& ca = gTriangles[a].centroid; Vec3& cb = gTriangles[b].centroid;
            if (axis == 0) return ca.x < cb.x; else if (axis == 1) return ca.y < cb.y; else return ca.z < cb.z;
        });
    std::vector<int> leftIndices(triIndices.begin(), triIndices.begin() + mid);
    std::vector<int> rightIndices(triIndices.begin() + mid, triIndices.end());
    
    #if defined(_OPENMP)
    if (depth < 4) {
        #pragma omp parallel sections
        {
            #pragma omp section 
            { node->left = buildBVH(leftIndices, depth + 1); }
            #pragma omp section 
            { node->right = buildBVH(rightIndices, depth + 1); }
        }
    } else { node->left = buildBVH(leftIndices, depth + 1); node->right = buildBVH(rightIndices, depth + 1); }
    #else
    node->left = buildBVH(leftIndices, depth + 1); node->right = buildBVH(rightIndices, depth + 1);
    #endif
    return node;
}

int flattenBVH(BVHNode* node, int& currentTriCount) {
    if (!node) return -1;
    int idx = (int)gFlatNodes.size(); gFlatNodes.push_back({}); gFlatNodes[idx].box = node->box;
    if (node->isLeaf) {
        gFlatNodes[idx].triStart = currentTriCount; gFlatNodes[idx].triCount = (int)node->triIndices.size(); gFlatNodes[idx].rightChildIdx = -1;
        for (int triIdx : node->triIndices) gOrderedTriangles[currentTriCount++] = gTriangles[triIdx];
    } else {
        gFlatNodes[idx].triCount = 0;
        flattenBVH(node->left, currentTriCount);
        int rightIdx = flattenBVH(node->right, currentTriCount);
        gFlatNodes[idx].rightChildIdx = rightIdx;
    }
    return idx;
}

// --------------------------------------------------------
// [Search Functions]
// --------------------------------------------------------
void findClosestInLinearBVH(int nodeIdx, Vec3 p, double& bestDistSq, Vec3& bestPoint) {
    if (nodeIdx == -1) return;
    const FlatBVHNode& node = gFlatNodes[nodeIdx];
    if (distSqPointAABB_AVX(p, node.box) >= bestDistSq) return;
    if (node.triCount > 0) { 
        for (int i = 0; i < node.triCount; ++i) {
            const Triangle& t = gOrderedTriangles[node.triStart + i];
            Vec3 c = closestPointOnTriangle(p, t.v0, t.v1, t.v2);
            double dSq = lenSq(sub(c, p));
            if (dSq < bestDistSq) { bestDistSq = dSq; bestPoint = c; }
        }
    } else { 
        findClosestInLinearBVH(nodeIdx + 1, p, bestDistSq, bestPoint);
        findClosestInLinearBVH(node.rightChildIdx, p, bestDistSq, bestPoint);
    }
}

Vec3 getClosest(Vec3 p) {
    if (gFlatNodes.empty()) return p;
    double bestDistSq = 1e20; Vec3 bestPoint = p;
    findClosestInLinearBVH(0, p, bestDistSq, bestPoint);
    return bestPoint;
}

// [NEW] Hybrid Search (KD-Tree -> Local BVH)
Vec3 getClosestHybrid(Vec3 p) {
    // 1. KD-Tree가 없으면 기존 BVH로 Fallback
    if (gKDTree.empty()) return getClosest(p);

    std::vector<int> candidateTriIndices;
    std::vector<double> dists;
    candidateTriIndices.reserve(12);
    dists.reserve(12);

    // 2. KD-Tree로 가장 가까운 정점 8개 탐색
    searchKDTree(0, p, 8, candidateTriIndices, dists);

    if (candidateTriIndices.empty()) return getClosest(p);

    double bestDistSq = 1e20;
    Vec3 bestPoint = p;

    // 3. 후보 정점이 포함된 삼각형들에 대해서만 정밀 투영 (Local Refinement)
    for (int triIdx : candidateTriIndices) {
        // gKDPoints가 가리키는 triIdx는 원본 gTriangles의 인덱스임
        const Triangle& t = gTriangles[triIdx]; 
        Vec3 c = closestPointOnTriangle(p, t.v0, t.v1, t.v2);
        double dSq = lenSq(sub(c, p));
        if (dSq < bestDistSq) {
            bestDistSq = dSq;
            bestPoint = c;
        }
    }
    return bestPoint;
}

// --------------------------------------------------------
// [Export Functions]
// --------------------------------------------------------
extern "C" {
    DllExport bool isDeviceConnected() {
        if (hHD == HD_INVALID_HANDLE) return false;
        return (hdGetError().errorCode == HD_SUCCESS);
    }

    DllExport void printOpenMPStatus() {
        printf("\n============================================\n");
        #ifdef _OPENMP
            printf("[ C++ DLL ] OpenMP: ACTIVE (%d Threads)\n", omp_get_max_threads());
        #endif
        printf("[ C++ DLL ] Mode: Hybrid Acceleration (KD-Tree + BVH)\n");
        printf("[ C++ DLL ] Status: Thread Safe & Optimized\n");
        printf("============================================\n\n");
    }

    DllExport void setOffset(double x, double y, double z) { gOffsetX=x; gOffsetY=y; gOffsetZ=z; }

    DllExport void updateMesh(double* data, int triCount) {
        // [Safety] 햅틱 스레드 접근 차단
        gIsDataReady = false; 
        Sleep(2); // Wait for ongoing haptic loop

        if (gMeshVertices) free(gMeshVertices);
        gTriangles.clear(); gOrderedTriangles.clear(); gFlatNodes.clear();
        gKDTree.clear(); gKDPoints.clear();
        gTriCount = triCount;
        
        if (triCount > 0) {
            gMeshVertices = (double*)malloc(sizeof(double) * triCount * 9);
            if (gMeshVertices) memcpy(gMeshVertices, data, sizeof(double) * triCount * 9);
            
            gTriangles.resize(triCount);
            gOrderedTriangles.resize(triCount);
            std::vector<int> indices(triCount);
            
            // KD-Tree Point Reservation
            gKDPoints.reserve(triCount * 3);

            #pragma omp parallel for
            for(int i=0; i<triCount; ++i) {
                int idx = i*9;
                gTriangles[i].v0 = {data[idx], data[idx+1], data[idx+2]};
                gTriangles[i].v1 = {data[idx+3], data[idx+4], data[idx+5]};
                gTriangles[i].v2 = {data[idx+6], data[idx+7], data[idx+8]};
                gTriangles[i].id = i;
                gTriangles[i].centroid.x = (gTriangles[i].v0.x + gTriangles[i].v1.x + gTriangles[i].v2.x) / 3.0;
                gTriangles[i].centroid.y = (gTriangles[i].v0.y + gTriangles[i].v1.y + gTriangles[i].v2.y) / 3.0;
                gTriangles[i].centroid.z = (gTriangles[i].v0.z + gTriangles[i].v1.z + gTriangles[i].v2.z) / 3.0;
                indices[i] = i;
            }

            // Fill KD-Tree Points (Main Thread)
            for(int i=0; i<triCount; ++i) {
                gKDPoints.push_back({gTriangles[i].v0, i, i*3+0});
                gKDPoints.push_back({gTriangles[i].v1, i, i*3+1});
                gKDPoints.push_back({gTriangles[i].v2, i, i*3+2});
            }

            // Build BVH (Haptics)
            BVHNode* root = buildBVH(indices, 0);
            gFlatNodes.reserve(triCount * 2);
            int currentTriCount = 0;
            flattenBVH(root, currentTriCount);
            deleteBVH(root);

            // Build KD-Tree (Geometry)
            std::vector<int> kdIndices(gKDPoints.size());
            for(size_t i=0; i<kdIndices.size(); ++i) kdIndices[i] = i;
            gKDTree.reserve(gKDPoints.size());
            buildKDTreeRec(kdIndices, 0);
            
            // [Safety] 데이터 준비 완료
            gIsDataReady = true;
        } else {
            gMeshVertices = NULL;
            gIsDataReady = false;
        }
    }

    DllExport void getClosestPointOnMesh(double x, double y, double z, double* outPos) {
        if (!gIsDataReady || gFlatNodes.empty()) { outPos[0]=x; outPos[1]=y; outPos[2]=z; return; }
        // 일반 쿼리는 정밀도가 중요하므로 기존 BVH 사용
        Vec3 p = getClosest({x, y, z}); outPos[0]=p.x; outPos[1]=p.y; outPos[2]=p.z;
    }

    DllExport void computePatch(double* boundaryPts, int count, double* outVerts, int* outVertCount) {
        if (!gIsDataReady || gFlatNodes.empty() || count < 3) { *outVertCount = 0; return; }
        
        std::vector<Vec3> boundary; Vec3 center = {0,0,0};
        for(int i=0; i<count; ++i) { Vec3 p={boundaryPts[i*3],boundaryPts[i*3+1],boundaryPts[i*3+2]}; boundary.push_back(p); center=add(center, p); }
        center = mul(center, 1.0/count); 
        // Start center snap with Hybrid
        center = getClosestHybrid(center); 

        int rings=40; int totalPoints=(rings+1)*count;
        std::vector<Vec3> grid(totalPoints); std::vector<Vec3> tempGrid(totalPoints);
        
        #pragma omp parallel for
        for(int idx=0; idx<totalPoints; ++idx) {
            int r=idx/count; int i=idx%count; double t=(double)r/rings;
            grid[idx] = add(mul(center, 1.0-t), mul(boundary[i], t));
        }

        // Iteration Loop (Originally 40)
        for(int iter=0; iter<40; ++iter) {
            double totalMove=0.0;
            #pragma omp parallel for reduction(+:totalMove)
            for(int idx=count; idx<rings*count; ++idx) {
                int r=idx/count; int i=idx%count;
                Vec3 in=grid[(r-1)*count+i]; Vec3 out=grid[(r+1)*count+i];
                Vec3 left=grid[r*count+((i-1+count)%count)]; Vec3 right=grid[r*count+((i+1)%count)];
                Vec3 avg={(in.x+out.x+left.x+right.x)*0.25, (in.y+out.y+left.y+right.y)*0.25, (in.z+out.z+left.z+right.z)*0.25};
                
                double p=(double)iter/40.0; double sw=0.5*(1.0-p)+0.05*p;
                Vec3 rel=add(mul(grid[idx], 0.2), mul(avg, 0.8));
                
                // [Hybrid Acceleration] Here is the speedup!
                Vec3 snap = getClosestHybrid(rel);
                
                Vec3 res=add(mul(snap, 1.0-sw), mul(rel, sw));
                double dx=res.x-grid[idx].x; double dy=res.y-grid[idx].y; double dz=res.z-grid[idx].z;
                totalMove += sqrt(dx*dx+dy*dy+dz*dz); tempGrid[idx]=res;
            }
            #pragma omp parallel for
            for(int idx=count; idx<rings*count; ++idx) grid[idx]=tempGrid[idx];
            if(iter>10 && totalMove < 0.05*count) break;
        }
        
        // Final Snap
        #pragma omp parallel for
        for(int idx=count; idx<rings*count; ++idx) grid[idx]=getClosestHybrid(grid[idx]);
        
        int vIdx=0;
        for(int r=0; r<rings; ++r) {
            for(int i=0; i<count; ++i) {
                int n=(i+1)%count;
                Vec3 p1=grid[r*count+i]; Vec3 p2=grid[r*count+n];
                Vec3 p3=grid[(r+1)*count+n]; Vec3 p4=grid[(r+1)*count+i];
                outVerts[vIdx++]=p1.x; outVerts[vIdx++]=p1.y; outVerts[vIdx++]=p1.z;
                outVerts[vIdx++]=p2.x; outVerts[vIdx++]=p2.y; outVerts[vIdx++]=p2.z;
                outVerts[vIdx++]=p4.x; outVerts[vIdx++]=p4.y; outVerts[vIdx++]=p4.z;
                outVerts[vIdx++]=p2.x; outVerts[vIdx++]=p2.y; outVerts[vIdx++]=p2.z;
                outVerts[vIdx++]=p3.x; outVerts[vIdx++]=p3.y; outVerts[vIdx++]=p3.z;
                outVerts[vIdx++]=p4.x; outVerts[vIdx++]=p4.y; outVerts[vIdx++]=p4.z;
            }
        }
        *outVertCount = vIdx/3;
    }
    
    DllExport void computePatchPoints(double* b, int c, double* o, int* oc) { *oc = 0; }

    HDCallbackCode HDCALLBACK IntegratedCallback(void* pUserData) {
        hdBeginFrame(hdGetCurrentDevice());
        double m[16]; hdGetDoublev(HD_CURRENT_TRANSFORM, m); decomposeMatrix(m);
        HDint b; hdGetIntegerv(HD_CURRENT_BUTTONS, &b); gCurrentData.buttons = (int)b;
        double pos[3] = { gCurrentData.posX, gCurrentData.posY, gCurrentData.posZ };
        double force[3] = {0, 0, 0};

        // [Safety Check]
        if (gIsDataReady) {
            if (gTrackMode) {
                 double distance = sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]);
                 if (distance > 0.001) {
                     if (distance > gRadius + 5.0) { force[0] = -(pos[0] / distance) * gGuideForce; force[1] = -(pos[1] / distance) * gGuideForce; force[2] = -(pos[2] / distance) * gGuideForce; }
                     else { double penetration = gRadius - distance; force[0] = (pos[0] / distance) * penetration * gStiffness; force[1] = (pos[1] / distance) * penetration * gStiffness; force[2] = (pos[2] / distance) * penetration * gStiffness; }
                 }
            } else if (gForceMode) {
                 double distance = sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]);
                 if (distance < gRadius) {
                     double penetration = gRadius - distance;
                     if (distance > 0.001) {
                         force[0] = (pos[0] / distance) * penetration * gStiffness;
                         force[1] = (pos[1] / distance) * penetration * gStiffness;
                         force[2] = (pos[2] / distance) * penetration * gStiffness;
                     }
                 }
            } else if (gFlatNodes.size() > 0 && gTriCount > 0) { 
                int nodeStack[128]; int stackPtr = 0; nodeStack[stackPtr++] = 0; 
                double minDist = 10000.0; double bestNormal[3] = {0, 0, 0}; bool found = false;
                Vec3 p = {pos[0], pos[1], pos[2]};
                double cullDistSq = 36.0; 

                while(stackPtr > 0) {
                    int nodeIdx = nodeStack[--stackPtr];
                    const FlatBVHNode& node = gFlatNodes[nodeIdx];
                    if (distSqPointAABB_AVX(p, node.box) > cullDistSq) continue; 

                    if (node.triCount > 0) { 
                        for (int i = 0; i < node.triCount; ++i) {
                            const Triangle& t = gOrderedTriangles[node.triStart + i];
                            double v[3][3] = {{t.v0.x,t.v0.y,t.v0.z},{t.v1.x,t.v1.y,t.v1.z},{t.v2.x,t.v2.y,t.v2.z}};
                            double e1[3] = {v[1][0]-v[0][0], v[1][1]-v[0][1], v[1][2]-v[0][2]};
                            double e2[3] = {v[2][0]-v[0][0], v[2][1]-v[0][1], v[2][2]-v[0][2]};
                            double n[3] = {e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]};
                            double mag = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
                            if (mag < 1e-6) continue;
                            n[0]/=mag; n[1]/=mag; n[2]/=mag;
                            double pa[3] = {pos[0]-v[0][0], pos[1]-v[0][1], pos[2]-v[0][2]};
                            double d = pa[0]*n[0] + pa[1]*n[1] + pa[2]*n[2];
                            if (d < 2.0 && d > -5.0) {
                                double projP[3] = {pos[0]-d*n[0], pos[1]-d*n[1], pos[2]-d*n[2]};
                                if (isPointInTriangle(projP, v[0], v[1], v[2])) {
                                    if (fabs(d) < fabs(minDist)) { minDist = d; bestNormal[0]=n[0]; bestNormal[1]=n[1]; bestNormal[2]=n[2]; found = true; }
                                }
                            }
                        }
                    } else { 
                        if (node.rightChildIdx != -1) nodeStack[stackPtr++] = node.rightChildIdx;
                        nodeStack[stackPtr++] = nodeIdx + 1; 
                    }
                }
                if (found && minDist <= 1.0) {
                    double pen = 1.0 - minDist;
                    force[0] = bestNormal[0] * pen * 0.8; force[1] = bestNormal[1] * pen * 0.8; force[2] = bestNormal[2] * pen * 0.8;
                }
            }
        }

        double alpha = 0.4; 
        for(int i=0; i<3; i++) {
            force[i] = alpha * force[i] + (1.0 - alpha) * gLastForce[i];
            gLastForce[i] = force[i];
            if (force[i] > 3.0) force[i] = 3.0;
            if (force[i] < -3.0) force[i] = -3.0;
        }
        hdSetDoublev(HD_CURRENT_FORCE, force);
        hdEndFrame(hdGetCurrentDevice());
        return HD_CALLBACK_CONTINUE;
    }

    DllExport void setForceMode(bool enable) { gForceMode = enable; }
    DllExport void setTrackMode(bool enable) { gTrackMode = enable; }
    
    DllExport int startHaptics() {
        if (hHD != HD_INVALID_HANDLE) {
            hdStopScheduler();
            hdDisableDevice(hHD);
            hHD = HD_INVALID_HANDLE;
        }
        hHD = hdInitDevice(HD_DEFAULT_DEVICE);
        if (hHD == HD_INVALID_HANDLE) return -1;
        
        hdEnable(HD_FORCE_OUTPUT);
        gCallback = hdScheduleAsynchronous(IntegratedCallback, 0, HD_MAX_SCHEDULER_PRIORITY);
        hdStartScheduler();
        return 0;
    }
    
    DllExport void stopHaptics() {
        if (hHD != HD_INVALID_HANDLE) {
            hdStopScheduler();
            if (gCallback != HD_INVALID_HANDLE) hdUnschedule(gCallback);
            hdDisableDevice(hHD);
            hHD = HD_INVALID_HANDLE;
            gCallback = HD_INVALID_HANDLE;
        }
    }
    
    DllExport void getDeviceData(DeviceData* pOutData) { if (pOutData) *pOutData = gCurrentData; }
}