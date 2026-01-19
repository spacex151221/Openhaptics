import ctypes

# 사용자님의 성공 코드 상수 유지
HD_CURRENT_BUTTONS = 0x2000
HD_DEVICE_BUTTON_1 = 1 << 0  # 1
HD_DEVICE_BUTTON_2 = 1 << 1  # 2

class ButtonManager:
    def __init__(self, hd_lib):
        self.hd = hd_lib
        self.buttons = ctypes.c_int(0)

    def get_button_state(self):
        """현재 버튼 상태 문자열을 반환합니다."""
        self.hd.hdGetIntegerv(HD_CURRENT_BUTTONS, ctypes.byref(self.buttons))
        val = self.buttons.value
        
        b1 = (val & HD_DEVICE_BUTTON_1) != 0
        b2 = (val & HD_DEVICE_BUTTON_2) != 0
        
        if b1 and b2: return "BOTH PUSHED"
        elif b1:     return "BUTTON 1   "
        elif b2:     return "BUTTON 2   "

        else:        return "RELEASED   "
