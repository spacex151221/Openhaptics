import ctypes

# 사용자님의 성공 코드 상수 유지
HD_CURRENT_POSITION = 0x2050

class CoordinateManager:
    def __init__(self, hd_lib):
        self.hd = hd_lib
        self.pos = (ctypes.c_double * 3)()

    def get_position(self, handle):
        """현재 장치의 [X, Y, Z] 좌표 리스트를 반환합니다."""
        self.hd.hdGetDoublev(HD_CURRENT_POSITION, self.pos)
        return [self.pos[0], self.pos[2], self.pos[1]]