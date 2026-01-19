#include <windows.h> 
#include <HD/hd.h>
#include <math.h>

#define DllExport __declspec(dllexport)

extern "C" {
    HHD hHD = HD_INVALID_HANDLE;
    HDSchedulerHandle gCallback = HD_INVALID_HANDLE;
    double gStiffness = 0.5; 
    double gRadius = 40.0; 

    HDCallbackCode HDCALLBACK SphereCallback(void* pUserData) {
        hdBeginFrame(hdGetCurrentDevice());
        
        double pos[3];
        hdGetDoublev(HD_CURRENT_POSITION, pos); 


        double distance = sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]);


        if (distance < gRadius) {
            double force[3] = {0, 0, 0};
            

            double penetration = gRadius - distance;
            

            if (distance > 0.001) {
                force[0] = (pos[0] / distance) * penetration * gStiffness;
                force[1] = (pos[1] / distance) * penetration * gStiffness;
                force[2] = (pos[2] / distance) * penetration * gStiffness;
            }

            for(int i=0; i<3; i++) {
                if (force[i] > 2.5) force[i] = 2.5;
                if (force[i] < -2.5) force[i] = -2.5;
            }

            hdSetDoublev(HD_CURRENT_FORCE, force);  
        } else {
            double zero[3] = {0, 0, 0};
            hdSetDoublev(HD_CURRENT_FORCE, zero);
        }

        hdEndFrame(hdGetCurrentDevice());
        return HD_CALLBACK_CONTINUE;
    }

    DllExport int startHaptics() {
        hHD = hdInitDevice(HD_DEFAULT_DEVICE);
        if (hHD == HD_INVALID_HANDLE) return -1;

        hdEnable(HD_FORCE_OUTPUT);
        gCallback = hdScheduleAsynchronous(SphereCallback, 0, HD_MAX_SCHEDULER_PRIORITY);
        hdStartScheduler();
        return 0;
    }

    DllExport void stopHaptics() {
        if (hHD != HD_INVALID_HANDLE) {
            hdStopScheduler();
            if (gCallback != HD_INVALID_HANDLE) hdUnschedule(gCallback);
            hdDisableDevice(hHD);
        }
    }

}
