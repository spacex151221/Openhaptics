import ctypes
import subprocess
import os
import time

def build_cpp_dll():
    """파이썬이 직접 MSVC 컴파일러(cl.exe)를 호출하여 DLL을 빌드합니다."""
    print("C++ 엔진 빌드를 시작합니다...")
    
    # 1. 경로 설정 (사용자 환경에 맞게 확인 필요)
    cpp_file = "HapticEngine.cpp"
    output_dll = "HapticEngine.dll"
    include_path = r"C:\OpenHaptics\Developer\3.5.0\include"
    lib_path = r"C:\OpenHaptics\Developer\3.5.0\lib\x64"
    
    # 2. 컴파일 명령어 구성
    # /LD: DLL 생성, /I: 헤더 경로, /link: 라이브러리 경로 및 파일 지정
    compile_cmd = [
        "cl.exe", "/LD", cpp_file,
        f"/I{include_path}",
        "/link", f"/LIBPATH:{lib_path}", "hd.lib",
        f"/OUT:{output_dll}"
    ]
    
    try:
        # 3. 명령어 실행 (Developer PowerShell 환경이 필요하므로 shell=True 권장)
        # 중요: 이 코드를 실행하는 터미널 자체가 'x64 Native Tools' 상태여야 합니다.
        result = subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
        print("빌드 성공!")
        return True
    except subprocess.CalledProcessError as e:
        print("빌드 실패!")
        print(f"에러 메시지:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print("오류: cl.exe를 찾을 수 없습니다. 'x64 Native Tools Command Prompt'에서 VS Code를 실행했는지 확인하세요.")
        return False

def run_haptics():
    # 먼저 빌드를 수행합니다.
    if not build_cpp_dll():
        return

    # 빌드된 DLL 로드
    dll_path = os.path.abspath("HapticEngine.dll")
    try:
        haptic = ctypes.CDLL(dll_path)
        
        if haptic.startHaptics() == 0:
            print("\n" + "="*40)
            print("햅틱 엔진 가동 중 (구체 모드)")
            print("종료: Ctrl+C")
            print("="*40)
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                haptic.stopHaptics()
                print("\n엔진 안전 종료.")
    except Exception as e:
        print(f"DLL 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    run_haptics()