#include <windows.h> 
#include <HD/hd.h>
#include <math.h>
#include <algorithm>

#define DllExport __declspec(dllexport)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct DeviceData {
    double posX, posY, posZ;      
    double rotYaw, rotPitch, rotRoll; 
    double jointA1;
    double planeAngle;
    int buttons; 
};

// Global variables
HHD hHD = HD_INVALID_HANDLE;
HDSchedulerHandle gCallback = HD_INVALID_HANDLE;
DeviceData gCurrentData = { 0 };

double gOffsetX = 0.0;
double gOffsetY = 0.0;
double gOffsetZ = 0.0;

double gStiffness = 0.6; 
double gRadius = 40.0; 
double gGuideForce = 0.8; 
bool gForceMode = false;  
bool gTrackMode = false;  

double* gMeshVertices = NULL;
int gTriCount = 0;
double gLastForce[3] = {0, 0, 0}; 

extern "C" {
    // 연결 상태 확인 로직 강화
    DllExport bool isDeviceConnected() {
        // 핸들이 없으면 연결 안 됨
        if (hHD == HD_INVALID_HANDLE) return false;
        
        // 에러 스택을 확인하여 통신 장애가 있는지 체크
        HDErrorInfo error = hdGetError();
        if (error.errorCode != HD_SUCCESS) {
            if (error.errorCode == HD_DEVICE_FAULT || error.errorCode == HD_COMM_ERROR) {
                return false;
            }
        }
        
        // 장치가 여전히 유효한지 쿼리 시도
        HDint modelType;
        try {
            hdGetIntegerv(HD_DEVICE_MODEL_TYPE, &modelType);
        } catch (...) {
            return false;
        }
        
        return true;
    }

    DllExport void setOffset(double x, double y, double z) {
        gOffsetX = x;
        gOffsetY = y;
        gOffsetZ = z;
    }

    // 새로 넣는 stl 파일을 인지하는 부분
    DllExport void updateMesh(double* data, int triCount) {
        if (gMeshVertices) free(gMeshVertices);
        gTriCount = triCount;
        if (triCount > 0) {
            gMeshVertices = (double*)malloc(sizeof(double) * triCount * 9);
            if (gMeshVertices) memcpy(gMeshVertices, data, sizeof(double) * triCount * 9);
        } else {
            gMeshVertices = NULL;
        }
    }

    // 좌표 및 회전 각도 계산
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
        else { gCurrentData.rotRoll = atan2(m[4], m[0]); } //짐벌락 해결
    }

    //메쉬 인식
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

    //햅틱 서보 루틴 (1KHz로 힘 연산 및 각 모드에서 계산할 것 정의)
    HDCallbackCode HDCALLBACK IntegratedCallback(void* pUserData) {
        hdBeginFrame(hdGetCurrentDevice());
        double matrix[16];
        hdGetDoublev(HD_CURRENT_TRANSFORM, matrix);
        decomposeMatrix(matrix);
        HDint nButtons;
        hdGetIntegerv(HD_CURRENT_BUTTONS, &nButtons);
        gCurrentData.buttons = (int)nButtons;
        double pos[3] = { gCurrentData.posX, gCurrentData.posY, gCurrentData.posZ };
        double force[3] = {0, 0, 0};

        //트래킹 모드 (구 테스트)
        if (gTrackMode) {
            double distance = sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]);
            if (distance > 0.001) {
                if (distance > gRadius + 5.0) {
                    force[0] = -(pos[0] / distance) * gGuideForce;
                    force[1] = -(pos[1] / distance) * gGuideForce;
                    force[2] = -(pos[2] / distance) * gGuideForce;
                } else {
                    double penetration = gRadius - distance;
                    force[0] = (pos[0] / distance) * penetration * gStiffness;
                    force[1] = (pos[1] / distance) * penetration * gStiffness;
                    force[2] = (pos[2] / distance) * penetration * gStiffness;
                }
            }
        }
        //포스 모드 (구 테스트)
        else if (gForceMode) {
            double distance = sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]);
            if (distance < gRadius) {
                double penetration = gRadius - distance;
                if (distance > 0.001) {
                    force[0] = (pos[0] / distance) * penetration * gStiffness;
                    force[1] = (pos[1] / distance) * penetration * gStiffness;
                    force[2] = (pos[2] / distance) * penetration * gStiffness;
                }
            }
        }
        //메쉬 인식 후 힘부여
        else if (gMeshVertices && gTriCount > 0) {
            double minDist = 10000.0;
            double bestNormal[3] = {0, 0, 0};
            bool found = false;
            for (int i = 0; i < gTriCount; ++i) {
                int idx = i * 9;
                double v[3][3] = {
                    {gMeshVertices[idx], gMeshVertices[idx+1], gMeshVertices[idx+2]},
                    {gMeshVertices[idx+3], gMeshVertices[idx+4], gMeshVertices[idx+5]},
                    {gMeshVertices[idx+6], gMeshVertices[idx+7], gMeshVertices[idx+8]}
                };
                double e1[3] = { v[1][0]-v[0][0], v[1][1]-v[0][1], v[1][2]-v[0][2] };
                double e2[3] = { v[2][0]-v[0][0], v[2][1]-v[0][1], v[2][2]-v[0][2] };
                double n[3] = { e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0] };
                double mag = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
                if (mag < 1e-6) continue;
                n[0] /= mag; n[1] /= mag; n[2] /= mag;
                double pa[3] = { pos[0]-v[0][0], pos[1]-v[0][1], pos[2]-v[0][2] };
                double d = pa[0]*n[0] + pa[1]*n[1] + pa[2]*n[2];
                if (d < 2.0 && d > -5.0) {
                    double projP[3] = { pos[0] - d*n[0], pos[1] - d*n[1], pos[2] - d*n[2] };
                    if (isPointInTriangle(projP, v[0], v[1], v[2])) {
                        if (fabs(d) < fabs(minDist)) {
                            minDist = d;
                            bestNormal[0] = n[0]; bestNormal[1] = n[1]; bestNormal[2] = n[2];
                            found = true;
                        }
                    }
                }
            }
            if (found && minDist <= 1.0) {
                double penetration = 1.0 - minDist;
                double stlStiffness = 0.8; 
                force[0] = bestNormal[0] * penetration * stlStiffness;
                force[1] = bestNormal[1] * penetration * stlStiffness;
                force[2] = bestNormal[2] * penetration * stlStiffness;
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
        // 기존 장치가 있다면 완전히 닫고 새로 시작 시도
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
    
    DllExport void getDeviceData(DeviceData* pOutData) { if (pOutData) *pOutData = gCurrentData; }
    
    DllExport void stopHaptics() {
        if (hHD != HD_INVALID_HANDLE) {
            hdStopScheduler();
            if (gCallback != HD_INVALID_HANDLE) hdUnschedule(gCallback);
            hdDisableDevice(hHD);
            hHD = HD_INVALID_HANDLE;
            gCallback = HD_INVALID_HANDLE;
        }
    }
}