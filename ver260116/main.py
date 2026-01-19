import sys
import ctypes
import os
import math
import time
import subprocess
from collections import deque

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QOpenGLWidget, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog
from OpenGL.GL import *
from OpenGL.GLU import *
import stl
from stl import mesh

stl.stl.MAX_COUNT = 10**10 

#.cpp로 .dll 생성 (base_path 수정 필요)
def build_dll():
    cpp_file = "integrate.cpp"
    dll_file = "integrate.dll"
    base_path = r"C:\Users\KAMIC\Documents\project\Openhaptics"
    include_path = base_path
    lib_path = os.path.join(base_path, "lib") 
    if not os.path.exists(cpp_file): return False
    compile_cmd = ["cl", "/LD", "/EHsc", cpp_file, f"/I{include_path}", "/link", f"/LIBPATH:{lib_path}", "hd.lib", f"/OUT:{dll_file}"]
    try:
        subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
        return True
    except: return False

class DeviceData(ctypes.Structure):
    _fields_ = [
        ("posX", ctypes.c_double), ("posY", ctypes.c_double), ("posZ", ctypes.c_double),
        ("rotYaw", ctypes.c_double), ("rotPitch", ctypes.c_double), ("rotRoll", ctypes.c_double),
        ("jointA1", ctypes.c_double), ("planeAngle", ctypes.c_double), ("buttons", ctypes.c_int)
    ]

class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)
        self.pos = [0, 0, 0]
        self.angles = [0, 0, 0] 
        self.btn1_pressed = False
        self.btn2_pressed = False
        self.sphere_visible = False
        self.stl_visible = True  
        self.tool_mode_active = False 
        self.quad = None 
        self.cam_pos = [0, 0, 0]
        self.stl_mesh = None
        self.stl_list_id = None
        
    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glClearColor(0.05, 0.05, 0.05, 1.0)
        glLightfv(GL_LIGHT0, GL_POSITION, [100.0, 200.0, 300.0, 1.0])
        self.quad = gluNewQuadric()
        gluQuadricNormals(self.quad, GLU_SMOOTH)

    def set_mesh(self, mesh_data):
        self.stl_mesh = mesh_data
        self.makeCurrent()
        if self.stl_list_id: glDeleteLists(self.stl_list_id, 1)
        self.stl_list_id = glGenLists(1)
        glNewList(self.stl_list_id, GL_COMPILE)
        glBegin(GL_TRIANGLES)
        glColor3f(0.5, 0.5, 0.5)
        for face in self.stl_mesh.vectors:
            for vertex in face: glVertex3fv(vertex)
        glEnd()
        glEndList()
        self.doneCurrent()
        self.update()

    def paintGL(self):
        if not self.quad: return 
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        glLoadIdentity()
        glRotatef(self.angles[0], 0, 1, 0)
        glRotatef(self.angles[1], 0, 0, 1)
        glRotatef(-90, 0, 1, 0)
        m = glGetDoublev(GL_MODELVIEW_MATRIX)
        vx, vy, vz = m[2][0], m[2][1], m[2][2]
        glPopMatrix()
        dist = 300.0
        self.cam_pos = [self.pos[0] + (vx * dist), self.pos[1] + (vy * dist), self.pos[2] + (vz * dist)]
        glLoadIdentity()
        if self.tool_mode_active:
            gluLookAt(self.cam_pos[0], self.cam_pos[1], self.cam_pos[2], self.pos[0], self.pos[1], self.pos[2], 0, 1, 0)
        else:
            gluLookAt(250, 180, 250, 0, 0, 0, 0, 1, 0)
            glRotatef(45, 0, 1, 0)
        self.draw_grid()
        self.draw_origin_axes()
        if not self.tool_mode_active: self.draw_camera_point()
        if self.stl_list_id and self.stl_visible: glCallList(self.stl_list_id)
        if self.sphere_visible:
            glPushMatrix()
            glColor4f(0.6, 0.6, 0.6, 0.3)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            gluSphere(self.quad, 40.0, 32, 32)
            glDisable(GL_BLEND)
            glPopMatrix()
        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])
        glRotatef(self.angles[0], 0, 1, 0)      
        glRotatef(self.angles[1], 0, 0, 1)  
        glRotatef(self.angles[2], 1, 0, 0)
        self.draw_stylus()
        glPopMatrix()

    def draw_camera_point(self):
        glPushMatrix()
        glTranslatef(self.cam_pos[0], self.cam_pos[1], self.cam_pos[2])
        glDisable(GL_LIGHTING)
        glColor3f(0.0, 1.0, 0.0)
        gluSphere(self.quad, 3.0, 16, 16)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(self.pos[0]-self.cam_pos[0], self.pos[1]-self.cam_pos[1], self.pos[2]-self.cam_pos[2])
        glEnd()
        glEnable(GL_LIGHTING)
        glPopMatrix()

    def draw_grid(self):
        glDisable(GL_LIGHTING)
        glColor3f(0.2, 0.2, 0.2)
        glBegin(GL_LINES)
        for i in range(-200, 201, 40):
            glVertex3f(i, 0, -200); glVertex3f(i, 0, 200)
            glVertex3f(-200, 0, i); glVertex3f(200, 0, i)
        glEnd()
        glEnable(GL_LIGHTING)

    def draw_origin_axes(self):
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 1.0)
        gluSphere(self.quad, 2.0, 16, 16)
        glBegin(GL_LINES)
        glColor3f(1, 0, 0); glVertex3f(0, 0, 0); glVertex3f(150, 0, 0)
        glColor3f(0, 1, 0); glVertex3f(0, 0, 0); glVertex3f(0, 150, 0)
        glColor3f(0, 0.5, 1); glVertex3f(0, 0, 0); glVertex3f(0, 0, 150)
        glEnd()
        glEnable(GL_LIGHTING)

    def draw_stylus(self):
        if self.btn1_pressed: glColor3f(1.0, 0.1, 0.1)
        elif self.btn2_pressed: glColor3f(0.1, 1.0, 0.1)
        else: glColor3f(1.0, 0.9, 0.0)
        gluSphere(self.quad, 4.0, 16, 16) 
        glColor3f(0.0, 0.7, 1.0) 
        glPushMatrix()
        glRotatef(-90, 0, 1, 0) 
        gluCylinder(self.quad, 2.5, 1.5, 60, 20, 1) 
        glPopMatrix()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / h, 1.0, 2000.0)
        glMatrixMode(GL_MODELVIEW)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KAMIC 햅틱 시연 프로그램_260116 ver")
        self.resize(1200, 800)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        self.gl_widget = GLWidget()
        main_layout.addWidget(self.gl_widget, 8)
        side_layout = QVBoxLayout()
        self.info_panel = QtWidgets.QLabel("Initializing...")
        self.info_panel.setStyleSheet("background-color: #0F0F0F; color: #00FF00; font-family: 'Arial'; font-size: 11pt; padding: 15px;")
        self.info_panel.setAlignment(QtCore.Qt.AlignTop)
        side_layout.addWidget(self.info_panel)

        # --- [추가: 폰트 설정 공통 스타일] ---
        btn_font_style = "font-family: 'Malgun Gothic'; font-weight: bold;"

        self.calib_btn = QPushButton("CALIBRATE (3 SEC)")
        self.calib_btn.setMinimumHeight(45)
        self.calib_btn.setStyleSheet(f"{btn_font_style} font-size: 11pt; background-color: #AA5500; color: white;")
        self.calib_btn.clicked.connect(self.start_calibration)
        side_layout.addWidget(self.calib_btn)

        tool_layout = QHBoxLayout()
        self.tool_btn = QPushButton("TOOL MODE: OFF")
        self.tool_btn.setCheckable(True)
        self.tool_btn.setMinimumHeight(40)
        self.tool_btn.setStyleSheet(f"{btn_font_style} font-size: 10pt; background-color: #444; color: white;")
        self.tool_btn.clicked.connect(self.handle_tool_mode)
        tool_layout.addWidget(self.tool_btn)

        self.load_btn = QPushButton("LOAD STL")
        self.load_btn.setMinimumHeight(40)
        self.load_btn.setStyleSheet(f"{btn_font_style} font-size: 10pt; background-color: #555; color: white;")
        self.load_btn.clicked.connect(self.load_stl_file)
        tool_layout.addWidget(self.load_btn)

        tool_layout.addStretch(1)
        side_layout.addLayout(tool_layout)

        self.track_btn = QPushButton("TRACKING TEST: OFF")
        self.track_btn.setCheckable(True)
        self.track_btn.setMinimumHeight(50)
        self.track_btn.setStyleSheet(f"{btn_font_style} font-size: 11pt; background-color: #444; color: white;")
        self.track_btn.clicked.connect(self.handle_track_test)
        side_layout.addWidget(self.track_btn)

        self.force_btn = QPushButton("FORCE TEST: OFF")
        self.force_btn.setCheckable(True)
        self.force_btn.setMinimumHeight(50)
        self.force_btn.setStyleSheet(f"{btn_font_style} font-size: 11pt; background-color: #444; color: white;")
        self.force_btn.clicked.connect(self.handle_force_test)
        side_layout.addWidget(self.force_btn)

        main_layout.addLayout(side_layout, 2)
        self.filter_size = 5
        self.pitch_history = deque(maxlen=self.filter_size)
        self.yaw_sin_history = deque(maxlen=self.filter_size)
        self.yaw_cos_history = deque(maxlen=self.filter_size)
        self.last_btn2_state = False 
        self.is_calibrating = False
        self.calib_samples = []
        self.calib_timer = QtCore.QTimer()
        self.calib_timer.timeout.connect(self.collect_calib_data)
        self.data = DeviceData()
        self.haptic = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_all)
        self.monitor_timer = QtCore.QTimer()
        self.monitor_timer.timeout.connect(self.check_device_status)
        self.monitor_timer.start(500) 
        if build_dll(): self.init_haptics()

    def check_device_status(self):
        if not self.haptic: return
        try: connected = bool(self.haptic.isDeviceConnected())
        except: connected = False
        if connected:
            if not self.timer.isActive():
                if self.haptic.startHaptics() == 0: self.timer.start(16)
        else:
            if self.timer.isActive():
                self.timer.stop()
                self.haptic.stopHaptics()
            self.show_disconnected_error()

    def show_disconnected_error(self):
        self.info_panel.setText("[ DEVICE: DISCONNECTED ]\n\nSearching for device...\nPlease check power/cable.")
        self.info_panel.setStyleSheet("background-color: #550000; color: #FFFFFF; font-family: 'Arial'; font-size: 11pt; padding: 15px;")

    def start_calibration(self):
        if not self.timer.isActive(): return
        self.is_calibrating = True
        self.calib_samples = []
        self.calib_btn.setEnabled(False)
        self.calib_btn.setText("CALIBRATING...")
        self.calib_btn.setStyleSheet("font-family: 'Arial'; font-size: 11pt; font-weight: bold; background-color: #FF0000; color: white;")
        QtCore.QTimer.singleShot(3000, self.finish_calibration) #3초 동안 평균낸 좌표를 새로운 (0,0,0 으로 지정)
        self.calib_timer.start(20) 

    def collect_calib_data(self):
        if self.is_calibrating and self.haptic:
            self.calib_samples.append([self.data.posX, self.data.posY, self.data.posZ])

    def finish_calibration(self):
        self.calib_timer.stop()
        self.is_calibrating = False
        if self.calib_samples:
            avg_pos = np.mean(self.calib_samples, axis=0)
            if self.haptic:
                cx = getattr(self, 'total_offset_x', 0.0) + avg_pos[0]
                cy = getattr(self, 'total_offset_y', 0.0) + avg_pos[1]
                cz = getattr(self, 'total_offset_z', 0.0) + avg_pos[2]
                self.total_offset_x, self.total_offset_y, self.total_offset_z = cx, cy, cz
                self.haptic.setOffset(ctypes.c_double(cx), ctypes.c_double(cy), ctypes.c_double(cz))
        self.calib_btn.setEnabled(True)
        self.calib_btn.setText("CALIBRATE (3 SEC)")
        self.calib_btn.setStyleSheet("font-family: 'Arial'; font-size: 11pt; font-weight: bold; background-color: #AA5500; color: white;")

    def load_stl_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open STL file', '', "STL files (*.stl)")
        if fname:
            try:
                m = mesh.Mesh.from_file(fname)
                self.gl_widget.set_mesh(m)
                if self.haptic:
                    flat_data = m.vectors.flatten().astype(np.float64)
                    self.haptic.updateMesh(flat_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(m.vectors))
            except Exception as e:
                self.info_panel.setText(f"STL Load Error: {e}")

    def init_haptics(self):
        dll_path = os.path.join(os.getcwd(), "integrate.dll")
        try:
            self.haptic = ctypes.CDLL(dll_path)
            self.haptic.startHaptics.restype = ctypes.c_int
            self.haptic.getDeviceData.argtypes = [ctypes.POINTER(DeviceData)]
            self.haptic.setForceMode.argtypes = [ctypes.c_bool]
            self.haptic.setTrackMode.argtypes = [ctypes.c_bool]
            self.haptic.updateMesh.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
            self.haptic.setOffset.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
            self.haptic.isDeviceConnected.restype = ctypes.c_bool
            if self.haptic.startHaptics() == 0: self.timer.start(16)
            else: self.show_disconnected_error()
            self.total_offset_x = self.total_offset_y = self.total_offset_z = 0.0
        except: self.info_panel.setText("Load Error!")

    def handle_tool_mode(self, checked):
        if checked:
            self.tool_btn.setChecked(True)
            self.tool_btn.setText("TOOL MODE: ON")
            self.tool_btn.setStyleSheet("font-family: 'Arial'; font-weight: bold; font-size: 10pt; background-color: #0055AA; color: white;")
            self.gl_widget.tool_mode_active = True
        else:
            self.tool_btn.setChecked(False)
            self.tool_btn.setText("TOOL MODE: OFF")
            self.tool_btn.setStyleSheet("font-family: 'Arial'; font-weight: bold; font-size: 10pt; background-color: #444; color: white;")
            self.gl_widget.tool_mode_active = False

    def handle_track_test(self, checked):
        if checked:
            self.force_btn.setChecked(False)
            self.gl_widget.stl_visible = False
            self.track_btn.setText("TRACKING TEST: ON")
            self.track_btn.setStyleSheet("font-family: 'Arial'; font-weight: bold; font-size: 11pt; background-color: #00AA00; color: white;")
            self.gl_widget.sphere_visible = True
            if self.haptic: 
                self.haptic.setTrackMode(True)
                self.haptic.setForceMode(False)
        else:
            self.track_btn.setText("TRACKING TEST: OFF")
            self.track_btn.setStyleSheet("font-family: 'Arial'; font-weight: bold; font-size: 11pt; background-color: #444; color: white;")
            self.gl_widget.sphere_visible = False
            self.gl_widget.stl_visible = True
            if self.haptic: self.haptic.setTrackMode(False)

    def handle_force_test(self, checked):
        if checked:
            self.track_btn.setChecked(False)
            self.gl_widget.stl_visible = False
            self.force_btn.setText("FORCE TEST: ON")
            self.force_btn.setStyleSheet("font-family: 'Arial'; font-weight: bold; font-size: 11pt; background-color: #00AA00; color: white;")
            self.gl_widget.sphere_visible = True
            if self.haptic: 
                self.haptic.setForceMode(True)
                self.haptic.setTrackMode(False)
        else:
            self.force_btn.setText("FORCE TEST: OFF")
            self.force_btn.setStyleSheet("font-family: 'Arial'; font-weight: bold; font-size: 11pt; background-color: #444; color: white;")
            self.gl_widget.sphere_visible = False
            self.gl_widget.stl_visible = True
            if self.haptic: self.haptic.setForceMode(False)

    def update_all(self):
        if not self.haptic or not self.timer.isActive(): return
        self.haptic.getDeviceData(ctypes.byref(self.data))
        dx, dy, dz = self.data.posX, self.data.posY, self.data.posZ
        self.yaw_sin_history.append(math.sin(self.data.planeAngle))
        self.yaw_cos_history.append(math.cos(self.data.planeAngle))
        avg_yaw = math.degrees(math.atan2(sum(self.yaw_sin_history), sum(self.yaw_cos_history)))
        if avg_yaw < 0: avg_yaw += 360.0
        self.pitch_history.append(math.degrees(self.data.rotPitch))
        avg_pitch = sum(self.pitch_history) / len(self.pitch_history)
        btn1, btn2 = bool(self.data.buttons & 1), bool(self.data.buttons & 2)
        if btn2 and not self.last_btn2_state:
            self.handle_tool_mode(not self.gl_widget.tool_mode_active)
        self.last_btn2_state = btn2
        self.gl_widget.pos, self.gl_widget.angles = [dx, dy, dz], [avg_yaw, -avg_pitch, math.degrees(self.data.rotRoll)]
        self.gl_widget.btn1_pressed, self.gl_widget.btn2_pressed = btn1, btn2
        self.gl_widget.update()
        mode_str = "None"
        if self.track_btn.isChecked(): mode_str = "Surface Tracing"
        elif self.force_btn.isChecked(): mode_str = "Volume Force"
        self.info_panel.setStyleSheet("background-color: #0F0F0F; color: #00FF00; font-family: 'Arial'; font-size: 11pt; padding: 15px;")
        status_text = (f"[ DEVICE: CONNECTED ]\n----------------------\nPOS X: {dx:7.1f}\nPOS Y: {dy:7.1f}\nPOS Z: {dz:7.1f}\n\nYaw: {avg_yaw:7.1f}\nPitch: {avg_pitch:7.1f}\n\nMode: {mode_str}\nTool Mode: {'[ ACTIVE ]' if self.gl_widget.tool_mode_active else '[ OFF ]'}\nButton 1: {'[ ON ]' if btn1 else '[ OFF ]'}\nButton 2: {'[ ON ]' if btn2 else '[ OFF ]'}")
        self.info_panel.setText(status_text)

    def closeEvent(self, event):
        if self.haptic: self.haptic.stopHaptics()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())