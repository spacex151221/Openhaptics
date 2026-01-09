import ctypes
from Coordinate import CoordinateManager
from Button import ButtonManager

class HapticEngine:
    def __init__(self):
        # 1. DLL 로드 및 초기 설정
        try:
            # hd.dll이 파일과 같은 폴더에 있어야 합니다.
            self.hd = ctypes.WinDLL("./hd.dll")
            
            # 함수 타입 정의 (OpenHaptics HDAPI 규격)
            self.hd.hdInitDevice.argtypes = [ctypes.c_char_p]
            self.hd.hdInitDevice.restype = ctypes.c_uint
            self.hd.hdStartScheduler.argtypes = []
            self.hd.hdBeginFrame.argtypes = [ctypes.c_uint]
            self.hd.hdGetDoublev.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_double)]
            self.hd.hdGetIntegerv.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_int)]
            self.hd.hdEndFrame.argtypes = [ctypes.c_uint]
            self.hd.hdStopScheduler.argtypes = []
            self.hd.hdDisableDevice.argtypes = [ctypes.c_uint]
            
            # 힘 제어를 위해 미리 정의해두는 함수 (가상 벽 구현 시 사용)
            self.hd.hdSetDoublev.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_double)]
            self.hd.hdEnable.argtypes = [ctypes.c_uint]
            
        except Exception as e:
            print(f"❌ DLL 로드 실패: {e}")
            raise

        # 2. 장치 초기화 (Default Device)
        self.handle = self.hd.hdInitDevice(None)
        if self.handle == 0xFFFF:
            raise Exception("❌ 장치 연결 실패: Touch 장치가 연결되어 있는지 확인하세요.")

        # 3. 햅틱 스케줄러 시작
        self.hd.hdStartScheduler()
        
        # 각 데이터 관리 모듈 인스턴스 생성
        self.coord_mgr = CoordinateManager(self.hd)
        self.btn_mgr = ButtonManager(self.hd)

    def get_data(self):
        """
        현재 장치의 좌표와 버튼 상태를 한 번에 가져옵니다.
        CoordinateManager에서 [X, Z, Y] 순서로 교정된 데이터를 받습니다.
        """
        self.hd.hdBeginFrame(self.handle)
        
        # [Device_X, Device_Z, Device_Y] 순서의 리스트 반환
        pos = self.coord_mgr.get_position(self.handle)
        # 버튼 상태 문자열 반환
        btn = self.btn_mgr.get_button_state()
        
        self.hd.hdEndFrame(self.handle)
        
        return pos, btn

    def close(self):
        """장치 연결을 안전하게 해제합니다."""
        self.hd.hdStopScheduler()
        self.hd.hdDisableDevice(self.handle)
        print("✅ 햅틱 엔진이 안전하게 종료되었습니다.")

# 단독 테스트용 코드
if __name__ == "__main__":
    try:
        engine = HapticEngine()
        print("엔진 가동 중... (Ctrl+C로 종료)")
        import time
        while True:
            p, b = engine.get_data()
            print(f"\rX:{p[0]:>7.2f} Z:{p[1]:>7.2f} Y:{p[2]:>7.2f} | {b}", end="")
            time.sleep(0.01)
    except KeyboardInterrupt:
        engine.close()