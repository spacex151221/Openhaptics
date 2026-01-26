import sys
import ctypes
import os
import math
import time
import subprocess
from collections import deque
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QOpenGLWidget, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMenuBar, QMenu, QAction
from OpenGL.GL import *
from OpenGL.GLU import *
import stl
from stl import mesh

stl.stl.MAX_COUNT = 10**10 

# [NEW] 고속 STL 바이너리 파서 (Fast Binary I/O)
def fast_load_stl(filename):
    try:
        # 파일 크기 검증 (Binary STL = 80 header + 4 count + N * 50 triangles)
        file_size = os.path.getsize(filename)
        if file_size < 84: return None
        
        with open(filename, 'rb') as f:
            f.seek(80) # Skip header
            count_data = f.read(4)
            count = np.frombuffer(count_data, dtype=np.uint32)[0]
            
            # 크기가 일치하지 않으면 바이너리가 아니거나 손상된 파일
            if file_size != 84 + count * 50:
                return None
            
            # 구조 정의 및 메모리 매핑
            dtype = np.dtype([
                ('normal', ('<f4', 3)),
                ('vertices', ('<f4', (3, 3))),
                ('attr', '<u2')
            ])
            data = np.fromfile(f, dtype=dtype, count=count)
            
            # 햅틱용 float64 변환
            return data['vertices'].astype(np.float64)
    except Exception:
        return None

# DLL 생성 함수
def build_dll():
    cpp_file = "integrate.cpp"
    dll_file = "integrate.dll"
    base_path = r"C:\Users\KAMIC\Documents\project\Openhaptics"
    include_path = base_path
    lib_path = os.path.join(base_path, "lib") 
    
    if not os.path.exists(cpp_file): 
        print(f"Error: {cpp_file} not found.")
        return False
        
    compile_cmd = ["cl", "/LD", "/EHsc", "/openmp", "/arch:AVX2", "/O2", cpp_file, f"/I{include_path}", "/link", f"/LIBPATH:{lib_path}", "hd.lib", f"/OUT:{dll_file}"]
    try:
        result = subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
        print("DLL Build Success.")
        return True
    except subprocess.CalledProcessError as e:
        print("DLL Build Failed:")
        print(e.stdout)
        print(e.stderr)
        return False

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
        
        self.selected_points = []   
        self.path_points = []       
        self.generated_faces = [] 
        
        self.cam_distance = 300.0
        self.mesh_unique_verts = None
        self.mesh_vertices_flat = None # 원본 데이터 저장용 (Lazy Calc)
        
        # [VBO] 버퍼 ID 초기화
        self.vbo_id = 0
        self.vbo_normal_id = 0
        self.vertex_count = 0

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glClearColor(0.05, 0.05, 0.05, 1.0)
        
        glLightfv(GL_LIGHT0, GL_POSITION, [100.0, 200.0, 300.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])

        glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glMaterialfv(GL_FRONT, GL_SHININESS, [50.0])

        self.quad = gluNewQuadric()
        gluQuadricNormals(self.quad, GLU_SMOOTH)
        
        # [VBO] 버퍼 생성
        self.vbo_id = glGenBuffers(1)
        self.vbo_normal_id = glGenBuffers(1)

    def set_mesh_data(self, vertices_f64):
        # [Lazy Loading] 1. 원본 데이터 저장 (계산 안 함!)
        self.mesh_vertices_flat = vertices_f64.flatten().reshape(-1, 3) 
        self.mesh_unique_verts = None # 나중에 계산
        
        self.selected_points = []
        self.path_points = []
        self.generated_faces = []

        # [VBO Upload] 2. 렌더링 데이터 준비 (Float32)
        verts_f32 = vertices_f64.astype(np.float32) # (N, 3, 3)
        
        # 벡터화된 법선 계산 (Vectorized Normal Calculation)
        v0 = verts_f32[:, 0, :]
        v1 = verts_f32[:, 1, :]
        v2 = verts_f32[:, 2, :]
        
        # Cross product (Face Normals)
        edge1 = v1 - v0
        edge2 = v2 - v0
        normals = np.cross(edge1, edge2)
        
        # Normalize
        norm_lens = np.linalg.norm(normals, axis=1, keepdims=True)
        norm_lens[norm_lens == 0] = 1.0 
        normals /= norm_lens
        
        # Flat Shading용 정점 법선 복제
        vertex_normals = np.repeat(normals, 3, axis=0)
        
        # 정점 데이터 평탄화
        vertex_data = verts_f32.flatten()
        normal_data = vertex_normals.flatten()
        
        self.vertex_count = len(vertex_data) // 3
        
        # [VBO] 3. GPU로 데이터 전송
        self.makeCurrent()
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normal_id)
        glBufferData(GL_ARRAY_BUFFER, normal_data.nbytes, normal_data, GL_STATIC_DRAW)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0) # Unbind
        
        self.doneCurrent()
        self.update()

    def set_mesh(self, mesh_data):
        self.stl_mesh = mesh_data
        self.set_mesh_data(self.stl_mesh.vectors.astype(np.float64))

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
        
        dist = self.cam_distance
        self.cam_pos = [self.pos[0] + (vx * dist), self.pos[1] + (vy * dist), self.pos[2] + (vz * dist)]
        
        glLoadIdentity()
        if self.tool_mode_active:
            gluLookAt(self.cam_pos[0], self.cam_pos[1], self.cam_pos[2], self.pos[0], self.pos[1], self.pos[2], 0, 1, 0)
        else:
            gluLookAt(dist * 0.8, dist * 0.6, dist * 0.8, 0, 0, 0, 0, 1, 0)
            glRotatef(45, 0, 1, 0)
            
        self.draw_grid()
        self.draw_origin_axes()
        if not self.tool_mode_active: self.draw_camera_point()
        
        # [VBO Rendering with Dual Pass (Wireframe)]
        if self.stl_visible and self.vertex_count > 0:
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)
            
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
            glVertexPointer(3, GL_FLOAT, 0, None)
            
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normal_id)
            glNormalPointer(GL_FLOAT, 0, None)
            
            # Pass 1: Filled (면 그리기)
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(1.0, 1.0)
            glColor3f(0.7, 0.8, 0.9)
            glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
            glDisable(GL_POLYGON_OFFSET_FILL)
            
            # Pass 2: Wireframe (선 그리기) [NEW]
            glDisable(GL_LIGHTING)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE) # 라인 모드 전환
            glColor3f(0.05, 0.05, 0.05) # 검은색
            glLineWidth(1.0)
            glDrawArrays(GL_TRIANGLES, 0, self.vertex_count) # 한번 더 그리기
            
            # 상태 복구
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL) # 필 모드 복구
            glEnable(GL_LIGHTING)

            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_NORMAL_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

        if self.generated_faces:
            glDisable(GL_LIGHTING)
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(-5.0, -5.0) 
            
            for face_verts in self.generated_faces:
                glColor4f(0.0, 1.0, 0.0, 0.6)
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(3, GL_DOUBLE, 0, face_verts)
                glDrawArrays(GL_TRIANGLES, 0, len(face_verts))
                glDisableClientState(GL_VERTEX_ARRAY)
                
                glColor3f(0.3, 1.0, 0.3)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glLineWidth(1.5)
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(3, GL_DOUBLE, 0, face_verts)
                glDrawArrays(GL_TRIANGLES, 0, len(face_verts))
                glDisableClientState(GL_VERTEX_ARRAY)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                glLineWidth(1.0)

            glDisable(GL_POLYGON_OFFSET_FILL)
            glEnable(GL_LIGHTING)

        if self.selected_points:
            glColor3f(1.0, 0.0, 0.0) 
            for pt in self.selected_points:
                glPushMatrix()
                glTranslatef(pt[0], pt[1], pt[2])
                gluSphere(self.quad, 2.0, 16, 16)
                glPopMatrix()
        
        if len(self.path_points) > 1:
            glDisable(GL_LIGHTING)
            glDisable(GL_DEPTH_TEST) 
            glLineWidth(3.0)
            glColor3f(1.0, 1.0, 0.0) 
            
            glBegin(GL_LINE_STRIP)
            for pt in self.path_points:
                glVertex3fv(pt)
            glEnd()

            glLineWidth(1.0)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)

        if self.sphere_visible:
            glPushMatrix()
            glColor4f(0.6, 0.6, 0.6, 0.3)
            gluSphere(self.quad, 40.0, 32, 32)
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
        self.setWindowTitle("KAMIC 햅틱 시연 프로그램_20260126")
        self.resize(1200, 800)
        
        self.create_menubar()

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

        self.save_btn = QPushButton("SAVE STL")
        self.save_btn.setMinimumHeight(40)
        self.save_btn.setStyleSheet(f"{btn_font_style} font-size: 10pt; background-color: #0055AA; color: white;")
        self.save_btn.clicked.connect(self.save_stl_file)
        tool_layout.addWidget(self.save_btn)

        tool_layout.addStretch(1)
        side_layout.addLayout(tool_layout)

        # [NEW] ZOOM MODE 버튼
        self.zoom_mode_btn = QPushButton("ZOOM MODE: OFF")
        self.zoom_mode_btn.setCheckable(True)
        self.zoom_mode_btn.setMinimumHeight(50)
        self.zoom_mode_btn.setStyleSheet(f"{btn_font_style} font-size: 11pt; background-color: #444; color: white;")
        self.zoom_mode_btn.clicked.connect(self.handle_zoom_mode)
        side_layout.addWidget(self.zoom_mode_btn)

        self.point_btn = QPushButton("POINT CLICK: OFF")
        self.point_btn.setCheckable(True)
        self.point_btn.setMinimumHeight(50) 
        self.point_btn.setStyleSheet(f"{btn_font_style} font-size: 11pt; background-color: #444; color: white;")
        self.point_btn.clicked.connect(self.handle_point_click)
        side_layout.addWidget(self.point_btn)

        self.surface_btn = QPushButton("SURFACE VIEW: OFF")
        self.surface_btn.setCheckable(True)
        self.surface_btn.setMinimumHeight(50)
        self.surface_btn.setStyleSheet(f"{btn_font_style} font-size: 11pt; background-color: #444; color: white;")
        self.surface_btn.clicked.connect(self.handle_surface_view)
        side_layout.addWidget(self.surface_btn)

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
        self.last_btn1_state = False 
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
        
        self.mesh_unique_verts = None

        if build_dll(): self.init_haptics()

    def create_menubar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        
        load_action = QtWidgets.QAction('Load STL', self)
        load_action.setShortcut('Ctrl+L')
        load_action.triggered.connect(self.load_stl_file)
        file_menu.addAction(load_action)

        calib_menu = menubar.addMenu('Calibrate')
        start_calib_action = QtWidgets.QAction('Start Calibration (3s)', self)
        start_calib_action.setShortcut('Ctrl+K')
        start_calib_action.triggered.connect(self.start_calibration)
        calib_menu.addAction(start_calib_action)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.RightButton:
            self.handle_undo_point()
        super().mousePressEvent(event)

    def handle_undo_point(self):
        if self.point_btn.isChecked():
            if self.gl_widget.selected_points:
                self.gl_widget.selected_points.pop()
                self.gl_widget.path_points = [] 
                
                if len(self.gl_widget.selected_points) >= 2:
                    for i in range(len(self.gl_widget.selected_points) - 1):
                        p1 = self.gl_widget.selected_points[i]
                        p2 = self.gl_widget.selected_points[i+1]
                        self.gl_widget.path_points.extend(self.calculate_projected_path(p1, p2))
                
                self.gl_widget.update()
                self.info_panel.setText("[ UNDO ]\nLast point removed.")
            else:
                if self.gl_widget.generated_faces:
                    self.gl_widget.generated_faces.pop()
                    self.gl_widget.update()
                    self.info_panel.setText("[ UNDO ]\nLast face removed.")
    
    def save_stl_file(self):
        if not self.gl_widget.generated_faces:
            self.info_panel.setText("[ SAVE ERROR ]\nNo generated faces\nto save!")
            return

        fname, _ = QFileDialog.getSaveFileName(self, 'Save STL', '', "STL files (*.stl)")
        if fname:
            try:
                num_rings = 40 
                filtered_list = []

                for face_verts in self.gl_widget.generated_faces:
                    total_verts = len(face_verts)
                    count = total_verts // (num_rings * 6)
                    reshaped = face_verts.reshape(num_rings, count, 6, 3)
                    clean_patch = reshaped[:, :-1, :, :]
                    filtered_list.append(clean_patch.reshape(-1, 3))

                all_verts = np.concatenate(filtered_list, axis=0)
                num_tris = len(all_verts) // 3
                data = np.zeros(num_tris, dtype=mesh.Mesh.dtype)
                data['vectors'] = all_verts.reshape(num_tris, 3, 3)
                
                m = mesh.Mesh(data)
                m.save(fname)
                self.info_panel.setText(f"[ SAVED ]\nClean Slicing Applied!\n{os.path.basename(fname)}")
            except Exception as e:
                self.info_panel.setText(f"[ ERROR ]\nSlicing failed:\n{e}")

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
        QtCore.QTimer.singleShot(3000, self.finish_calibration) 
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
                self.info_panel.setText("Loading STL...")
                QtWidgets.QApplication.processEvents() # UI 갱신 유도
                
                start_time = time.time()
                
                # 1. 고속 로더 시도 (Binary)
                mesh_vertices = fast_load_stl(fname)
                
                if mesh_vertices is None:
                    # 2. 실패 시(ASCII or Corrupt) 기존 로더 사용 (Fallback)
                    print("Fast load failed/skipped. Using standard loader (slower)...")
                    m = mesh.Mesh.from_file(fname)
                    mesh_vertices = m.vectors.astype(np.float64)
                
                # GL 위젯 및 햅틱 업데이트
                self.gl_widget.set_mesh_data(mesh_vertices)
                
                if self.haptic:
                    flat_data = mesh_vertices.flatten()
                    self.haptic.updateMesh(
                        flat_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                        len(mesh_vertices)
                    )
                
                end_time = time.time()
                self.info_panel.setText(f"STL Loaded.\nVertices: {len(mesh_vertices)}\nTime: {end_time-start_time:.2f}s")

            except Exception as e:
                self.info_panel.setText(f"STL Load Error: {e}")

    def calculate_projected_path(self, start_pt, end_pt):
        if not self.haptic: return []
        vec = end_pt - start_pt
        dist = np.linalg.norm(vec)
        if dist < 0.1: return [start_pt]
        step_size = 0.5 
        num_steps = int(dist / step_size)
        if num_steps < 2: num_steps = 2
        t_values = np.linspace(0, 1, num_steps)
        line_points = start_pt + np.outer(t_values, vec)
        path = []
        out_pos = (ctypes.c_double * 3)() 
        for pt in line_points:
            self.haptic.getClosestPointOnMesh(
                ctypes.c_double(pt[0]), ctypes.c_double(pt[1]), ctypes.c_double(pt[2]), 
                out_pos
            )
            path.append(np.array([out_pos[0], out_pos[1], out_pos[2]]))
        return path
    
    def create_surface_patch(self, boundary_loop):
        if not self.haptic or not boundary_loop: return None
        
        boundary_np = np.array(boundary_loop, dtype=np.float64)
        count = len(boundary_np)
        
        max_verts = (60) * count * 6 * 3 
        out_verts = (ctypes.c_double * max_verts)()
        out_count = ctypes.c_int(0)
        
        self.haptic.computePatch(
            boundary_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(count),
            out_verts,
            ctypes.byref(out_count)
        )
        
        valid_cnt = out_count.value
        if valid_cnt == 0: return None
        
        buffer_array = np.ctypeslib.as_array(out_verts, shape=(valid_cnt,))
        return buffer_array.reshape(-1, 3)

    def init_haptics(self):
        dll_path = os.path.join(os.getcwd(), "integrate.dll")
        if not os.path.exists(dll_path):
             self.info_panel.setText("DLL Not Found!\nCheck Compiler.")
             return

        try:
            self.haptic = ctypes.CDLL(dll_path)
            self.haptic.startHaptics.restype = ctypes.c_int
            self.haptic.getDeviceData.argtypes = [ctypes.POINTER(DeviceData)]
            self.haptic.setForceMode.argtypes = [ctypes.c_bool]
            self.haptic.setTrackMode.argtypes = [ctypes.c_bool]
            self.haptic.updateMesh.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
            self.haptic.setOffset.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
            self.haptic.getClosestPointOnMesh.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
            self.haptic.computePatch.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)]
            
            self.haptic.printOpenMPStatus.argtypes = []
            self.haptic.printOpenMPStatus.restype = None
            self.haptic.printOpenMPStatus()

            self.haptic.isDeviceConnected.restype = ctypes.c_bool
            if self.haptic.startHaptics() == 0: self.timer.start(16)
            else: self.show_disconnected_error()
            self.total_offset_x = self.total_offset_y = self.total_offset_z = 0.0
        except Exception as e: 
            self.info_panel.setText(f"Load Error!\n{e}")

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

    def handle_zoom_mode(self, checked):
        if checked:
            self.zoom_mode_btn.setText("ZOOM MODE: ON")
            self.zoom_mode_btn.setStyleSheet("font-family: 'Arial'; font-weight: bold; font-size: 11pt; background-color: #0055AA; color: white;")
            self.point_btn.setChecked(False)
            self.handle_point_click(False)
        else:
            self.zoom_mode_btn.setText("ZOOM MODE: OFF")
            self.zoom_mode_btn.setStyleSheet("font-family: 'Arial'; font-weight: bold; font-size: 11pt; background-color: #444; color: white;")

    def handle_point_click(self, checked):
        if checked:
            self.point_btn.setText("POINT CLICK: ON")
            self.point_btn.setStyleSheet("font-family: 'Arial'; font-weight: bold; font-size: 11pt; background-color: #0055AA; color: white;")
            self.zoom_mode_btn.setChecked(False)
            self.handle_zoom_mode(False)
            
            # [Lazy Calc] 포인트 모드를 켤 때만 무거운 연산 수행
            if self.gl_widget.mesh_unique_verts is None and self.gl_widget.mesh_vertices_flat is not None:
                self.info_panel.setText("Processing Points...")
                QtWidgets.QApplication.processEvents()
                # 여기서 멈춤 발생 (로딩 땐 안 멈춤)
                self.gl_widget.mesh_unique_verts = np.unique(self.gl_widget.mesh_vertices_flat, axis=0)
                self.info_panel.setText("Points Ready.")
        else:
            self.point_btn.setText("POINT CLICK: OFF")
            self.point_btn.setStyleSheet("font-family: 'Arial'; font-weight: bold; font-size: 11pt; background-color: #444; color: white;")
            self.gl_widget.selected_points = [] 
            self.gl_widget.path_points = [] 
            self.gl_widget.update()
    
    def handle_surface_view(self, checked):
        if checked:
            self.surface_btn.setText("SURFACE VIEW: ON")
            self.surface_btn.setStyleSheet("font-family: 'Arial'; font-weight: bold; font-size: 11pt; background-color: #00AA00; color: white;")
            self.gl_widget.stl_visible = False 
        else:
            self.surface_btn.setText("SURFACE VIEW: OFF")
            self.surface_btn.setStyleSheet("font-family: 'Arial'; font-weight: bold; font-size: 11pt; background-color: #444; color: white;")
            self.gl_widget.stl_visible = True 
        self.gl_widget.update()

    def handle_track_test(self, checked):
        if checked:
            self.force_btn.setChecked(False)
            self.point_btn.setChecked(False)
            self.handle_point_click(False)
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
            if not self.surface_btn.isChecked():
                self.gl_widget.stl_visible = True
            if self.haptic: self.haptic.setTrackMode(False)

    def handle_force_test(self, checked):
        if checked:
            self.track_btn.setChecked(False)
            self.point_btn.setChecked(False)
            self.handle_point_click(False)
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
            if not self.surface_btn.isChecked():
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
        
        # [ZOOM MODE Interaction]
        if self.zoom_mode_btn.isChecked():
            zoom_speed = 3.0 # 줌 속도
            if btn1: # 줌 인 (거리 감소)
                self.gl_widget.cam_distance -= zoom_speed
                if self.gl_widget.cam_distance < 10.0: self.gl_widget.cam_distance = 10.0
            elif btn2: # 줌 아웃 (거리 증가)
                self.gl_widget.cam_distance += zoom_speed
                if self.gl_widget.cam_distance > 1000.0: self.gl_widget.cam_distance = 1000.0
        else:
            if self.point_btn.isChecked() and btn1 and not self.last_btn1_state:
                if self.gl_widget.mesh_unique_verts is not None:
                    curr_pos = np.array([dx, dy, dz])
                    dists_sq = np.sum((self.gl_widget.mesh_unique_verts - curr_pos)**2, axis=1)
                    min_idx = np.argmin(dists_sq)
                    selected_pt = self.gl_widget.mesh_unique_verts[min_idx]
                    self.gl_widget.selected_points.append(selected_pt)
                    if len(self.gl_widget.selected_points) >= 2:
                        prev_pt = self.gl_widget.selected_points[-2]
                        curr_pt = self.gl_widget.selected_points[-1]
                        new_path = self.calculate_projected_path(prev_pt, curr_pt)
                        self.gl_widget.path_points.extend(new_path)
            
            if btn2 and not self.last_btn2_state:
                if self.point_btn.isChecked():
                    if len(self.gl_widget.selected_points) >= 3:
                        start_pt = self.gl_widget.selected_points[-1]
                        end_pt = self.gl_widget.selected_points[0]
                        closing_path = self.calculate_projected_path(start_pt, end_pt)
                        full_loop = self.gl_widget.path_points + closing_path
                        
                        patch_verts = self.create_surface_patch(full_loop)
                        if patch_verts is not None:
                            self.gl_widget.generated_faces.append(patch_verts)
                        
                        self.gl_widget.selected_points = []
                        self.gl_widget.path_points = []
                        self.info_panel.setText("[ FACE CREATED ]\nHigh-Density Patch Generated.")
                    else:
                        self.info_panel.setText("[ ERROR ]\nNeed at least 3 points\nto create a face.")
                else:
                    self.handle_tool_mode(not self.gl_widget.tool_mode_active)
            
        self.last_btn1_state = btn1
        self.last_btn2_state = btn2
        self.gl_widget.pos, self.gl_widget.angles = [dx, dy, dz], [avg_yaw, -avg_pitch, math.degrees(self.data.rotRoll)]
        self.gl_widget.btn1_pressed, self.gl_widget.btn2_pressed = btn1, btn2
        self.gl_widget.update()
        
        mode_str = "None"
        if self.zoom_mode_btn.isChecked(): mode_str = "Camera Zoom"
        elif self.track_btn.isChecked(): mode_str = "Surface Tracing"
        elif self.force_btn.isChecked(): mode_str = "Volume Force"
        elif self.point_btn.isChecked(): mode_str = "Point Click" 
        
        if "FACE" not in self.info_panel.text() and "ERROR" not in self.info_panel.text() and "UNDO" not in self.info_panel.text() and "SAVED" not in self.info_panel.text():
            status_text = (f"[ DEVICE: CONNECTED ]\n----------------------\nPOS X: {dx:7.1f}\nPOS Y: {dy:7.1f}\nPOS Z: {dz:7.1f}\n\nYaw: {avg_yaw:7.1f}\nPitch: {avg_pitch:7.1f}\n\nMode: {mode_str}\nZoom Dist: {self.gl_widget.cam_distance:.1f}\nTool Mode: {'[ ACTIVE ]' if self.gl_widget.tool_mode_active else '[ OFF ]'}\nButton 1: {'[ ON ]' if btn1 else '[ OFF ]'}\nButton 2: {'[ ON ]' if btn2 else '[ OFF ]'}")
            self.info_panel.setText(status_text)

    def closeEvent(self, event):
        if self.haptic: self.haptic.stopHaptics()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())