import ctypes
import time

# 필수 상수 정의
HD_FORCE_OUTPUT = 0x0800
HD_CALIBRATION_INKWELL = 0x2302
HD_CALIBRATION_STATUS = 0x2303

class HapticAdminTest:
    def __init__(self):
        self.hd = ctypes.WinDLL("./hd.dll")
        
        # --- [중요] Access Violation 방지를 위한 리턴 타입 정의 ---
        self.hd.hdGetError.restype = ctypes.c_uint # p.97 기반
        self.hd.hdInitDevice.restype = ctypes.c_uint
        self.hd.hdGetCurrentDevice.restype = ctypes.c_uint
        
        # 인자 타입 정의
        self.hd.hdInitDevice.argtypes = [ctypes.c_char_p]
        self.hd.hdGetIntegerv.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_int)]

    def run(self):
        handle = self.hd.hdInitDevice(None)
        if handle == 0xFFFF:
            print("❌ 장치 연결 실패")
            return

        self.hd.hdStartScheduler() # p.88 스케줄러 시작
        
        # 파란 불 깜빡임 해결을 위해 거치대 기반 캘리브레이션 명령
        print("💡 펜을 거치대에 넣으세요...")
        time.sleep(1)
        self.hd.hdUpdateCalibration(HD_CALIBRATION_INKWELL) # p.94
        
        self.hd.hdEnable(HD_FORCE_OUTPUT) # 힘 출력 허용

        # 상태 확인
        enabled = ctypes.c_int(0)
        self.hd.hdGetIntegerv(HD_FORCE_OUTPUT, ctypes.byref(enabled))
        
        print(f"Force Enabled: {bool(enabled.value)}")
        
        # 에러 체크
        err = self.hd.hdGetError()
        if err != 0:
            print(f"에러 코드: {hex(err)}")

        self.hd.hdStopScheduler()
        self.hd.hdDisableDevice(handle)

if __name__ == "__main__":
    HapticAdminTest().run()