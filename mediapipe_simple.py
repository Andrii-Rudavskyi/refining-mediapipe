import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
from datetime import datetime
import os
import glob
import csv
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

# Model paths
model_path = 'blaze_face_short_range.tflite'
landmarks_model_path = 'model/face_landmarker/face_landmarks_detector.tflite'
#landmarks_model_path = 'trained_models/landmarks_20260106_105539/landmarks_model.tflite'

# Initialize MediaPipe Face Detection
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
detector = vision.FaceDetector.create_from_options(options)

# Load landmarks detector with TensorFlow Lite
landmarks_interpreter = tf.lite.Interpreter(model_path=landmarks_model_path)
landmarks_interpreter.allocate_tensors()

# Get landmarks model details
landmarks_input_details = landmarks_interpreter.get_input_details()
landmarks_output_details = landmarks_interpreter.get_output_details()
_, landmarks_h, landmarks_w, _ = landmarks_input_details[0]['shape']
print(f"Landmarks Model Input Size: {landmarks_w}x{landmarks_h}")
print(f"Landmarks Model Outputs: {len(landmarks_output_details)}")
for i, output in enumerate(landmarks_output_details):
    print(f"  Output {i}: shape={output['shape']}")

# Global variables for GUI
cap = None
running = False
recording_video = False
recording_crop = False
recording_images = False
video_writer = None
crop_writer = None
tracked_bbox = None
tracking_active = False
frame_count = 0
detect_interval = 1
image_save_counter = 0
image_save_folder = None
landmarks_csv_file = None
landmarks_csv_writer = None
video_landmarks_csv_file = None
video_landmarks_csv_writer = None
video_frame_counter = 0

class FaceDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection & Landmark Tracking")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create video display frames
        video_frame = ttk.LabelFrame(main_frame, text="Original Video with Boxes", padding="5")
        video_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack()
        
        # Load video button
        self.load_video_button = ttk.Button(video_frame, text="Load Video", command=self.load_video)
        self.load_video_button.pack(pady=5)
        
        # Toggle for running on loaded video
        self.run_on_video = tk.BooleanVar(value=False)
        run_video_check = ttk.Checkbutton(video_frame, text="Run on loaded video", variable=self.run_on_video)
        run_video_check.pack(pady=5)
        
        # Label to show loaded video path
        self.loaded_video_label = ttk.Label(video_frame, text="No video loaded", foreground="gray")
        self.loaded_video_label.pack(pady=2)
        
        # Checkbox for landmarks on original video
        self.show_landmarks_on_video = tk.BooleanVar(value=False)
        landmarks_video_check = ttk.Checkbutton(video_frame, text="Landmarks on video", variable=self.show_landmarks_on_video)
        landmarks_video_check.pack(pady=5)
        
        crop_frame = ttk.LabelFrame(main_frame, text="Cropped Face with Landmarks", padding="5")
        crop_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.crop_label = ttk.Label(crop_frame)
        self.crop_label.pack()
        
        # Checkbox for landmarks overlay
        self.show_landmarks = tk.BooleanVar(value=True)
        landmarks_check = ttk.Checkbutton(crop_frame, text="Landmarks", variable=self.show_landmarks)
        landmarks_check.pack(pady=5)
        
        # Record images button
        self.record_images_button = ttk.Button(crop_frame, text="Record Images", command=self.toggle_record_images)
        self.record_images_button.pack(pady=5)
        
        # Load and process crop images controls
        crop_controls = ttk.Frame(crop_frame)
        crop_controls.pack(pady=5)
        
        self.load_crops_button = ttk.Button(crop_controls, text="Load Images", command=self.load_crop_images)
        self.load_crops_button.grid(row=0, column=0, padx=3)
        
        self.prev_crop_button = ttk.Button(crop_controls, text="Previous", command=self.previous_crop, state=tk.DISABLED)
        self.prev_crop_button.grid(row=0, column=1, padx=3)
        
        self.next_crop_button = ttk.Button(crop_controls, text="Next", command=self.next_crop, state=tk.DISABLED)
        self.next_crop_button.grid(row=0, column=2, padx=3)
        
        self.run_landmarks_button = ttk.Button(crop_frame, text="Run Landmarks", command=self.run_landmarks_on_crop, state=tk.DISABLED)
        self.run_landmarks_button.pack(pady=5)
        
        # Label for crop image info
        self.crop_info_label = ttk.Label(crop_frame, text="No images loaded", foreground="gray")
        self.crop_info_label.pack(pady=2)
        
        # 3D Visualization frame
        viz_3d_frame = ttk.LabelFrame(main_frame, text="3D Landmarks Visualization", padding="5")
        viz_3d_frame.grid(row=0, column=2, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create matplotlib figure for 3D plot
        self.fig_3d = Figure(figsize=(5, 5))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=viz_3d_frame)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Load CSV button
        self.load_csv_button = ttk.Button(viz_3d_frame, text="Load CSV", command=self.load_landmarks_csv)
        self.load_csv_button.pack(pady=5)
        
        # Analyze button
        self.analyze_button = ttk.Button(viz_3d_frame, text="Analyze Proportions", command=self.analyze_face_proportions)
        self.analyze_button.pack(pady=5)
        
        # Align pupils button
        self.align_button = ttk.Button(viz_3d_frame, text="Align Pupils", command=self.align_pupils)
        self.align_button.pack(pady=5)
        
        # Align sockets button
        self.align_sockets_button = ttk.Button(viz_3d_frame, text="Align Sockets", command=self.align_sockets)
        self.align_sockets_button.pack(pady=5)
        
        # Text widget for displaying aligned coordinates
        self.coords_text = tk.Text(viz_3d_frame, height=8, width=55, font=("Courier", 9))
        self.coords_text.pack(pady=5, padx=5)
        
        # Frame slider
        self.frame_slider_frame = ttk.Frame(viz_3d_frame)
        self.frame_slider_frame.pack(pady=5, fill=tk.X)
        
        ttk.Label(self.frame_slider_frame, text="Frame:").pack(side=tk.LEFT, padx=5)
        self.frame_slider_var = tk.IntVar(value=0)
        self.frame_slider = ttk.Scale(self.frame_slider_frame, from_=0, to=0, 
                                     variable=self.frame_slider_var, orient=tk.HORIZONTAL,
                                     command=self.update_3d_plot)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.frame_label = ttk.Label(self.frame_slider_frame, text="0/0")
        self.frame_label.pack(side=tk.LEFT, padx=5)
        
        # Loaded landmarks data
        self.landmarks_3d_data = None
        self.current_3d_frame = 0
        self.aligned_landmarks = None
        
        # Create parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="5")
        params_frame.grid(row=1, column=0, columnspan=3, pady=5)
        
        # Scale factor parameter
        scale_factor_frame = ttk.Frame(params_frame)
        scale_factor_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(scale_factor_frame, text="Scale Factor:").pack(side=tk.LEFT, padx=5)
        self.scale_factor_var = tk.DoubleVar(value=1.35)
        scale_factor_spinbox = ttk.Spinbox(scale_factor_frame, from_=1.0, to=2.0, increment=0.05, 
                                           textvariable=self.scale_factor_var, width=10)
        scale_factor_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Create button frame
        button_frame = ttk.Frame(main_frame, padding="5")
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        # Create buttons
        self.start_button = ttk.Button(button_frame, text="Start Program", command=self.start_program)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Program", command=self.stop_program, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)
        
        self.record_video_button = ttk.Button(button_frame, text="Record Video", command=self.toggle_record_video)
        self.record_video_button.grid(row=0, column=2, padx=5)
        
        self.record_crop_button = ttk.Button(button_frame, text="Record Crop", command=self.toggle_record_crop)
        self.record_crop_button.grid(row=0, column=3, padx=5)
        
        # Batch process button
        self.batch_process_button = ttk.Button(button_frame, text="Batch Process Videos", command=self.batch_process_videos)
        self.batch_process_button.grid(row=0, column=4, padx=5)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Status: Stopped", foreground="red")
        self.status_label.grid(row=3, column=0, columnspan=3, pady=5)
        
        self.running = False
        self.thread = None
        
        # Frame buffers for thread-safe GUI updates
        self.current_frame = None
        self.current_crop = None
        self.current_crop_no_landmarks = None
        self.current_landmarks = None
        self.frame_lock = threading.Lock()
        
        # Loaded video path
        self.loaded_video_path = None
        
        # Loaded crop images
        self.loaded_crops = []
        self.current_crop_index = 0
    
    def load_landmarks_csv(self):
        """Load landmarks CSV file for 3D visualization"""
        filename = filedialog.askopenfilename(
            title="Select Landmarks CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # Read CSV file
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                
                # Read all frames
                frames_data = []
                for row in reader:
                    frame_num = int(row[0])
                    landmarks_flat = [float(x) for x in row[1:]]
                    # Reshape to (478, 3)
                    landmarks = np.array(landmarks_flat).reshape(478, 3)
                    frames_data.append((frame_num, landmarks))
            
            self.landmarks_3d_data = frames_data
            
            if len(frames_data) > 0:
                # Update slider
                self.frame_slider.config(to=len(frames_data)-1)
                self.frame_slider_var.set(0)
                self.frame_label.config(text=f"0/{len(frames_data)-1}")
                
                # Display first frame
                self.update_3d_plot(0)
                
                print(f"Loaded {len(frames_data)} frames from CSV")
            else:
                messagebox.showwarning("Warning", "No data found in CSV file")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n{str(e)}")
    
    def update_3d_plot(self, val=None):
        """Update 3D plot with current frame landmarks"""
        if self.landmarks_3d_data is None or len(self.landmarks_3d_data) == 0:
            return
        
        frame_idx = int(self.frame_slider_var.get())
        frame_num, landmarks = self.landmarks_3d_data[frame_idx]
        
        # Update label
        self.frame_label.config(text=f"{frame_idx}/{len(self.landmarks_3d_data)-1}")
        
        # Clear and redraw
        self.ax_3d.clear()
        
        # Plot landmarks
        xs = landmarks[:, 0]
        ys = landmarks[:, 1]
        zs = landmarks[:, 2]
        
        self.ax_3d.scatter(xs, ys, zs, c='cyan', marker='o', s=1)
        
        # Highlight key landmarks
        left_pupil_idx = 468
        right_pupil_idx = 473
        nose_tip_idx = 1
        chin_idx = 152
        
        if len(landmarks) > max(left_pupil_idx, right_pupil_idx, nose_tip_idx, chin_idx):
            self.ax_3d.scatter([landmarks[left_pupil_idx, 0]], 
                             [landmarks[left_pupil_idx, 1]], 
                             [landmarks[left_pupil_idx, 2]], 
                             c='green', marker='o', s=20, label='Left Pupil')
            self.ax_3d.scatter([landmarks[right_pupil_idx, 0]], 
                             [landmarks[right_pupil_idx, 1]], 
                             [landmarks[right_pupil_idx, 2]], 
                             c='red', marker='o', s=20, label='Right Pupil')
            self.ax_3d.scatter([landmarks[nose_tip_idx, 0]], 
                             [landmarks[nose_tip_idx, 1]], 
                             [landmarks[nose_tip_idx, 2]], 
                             c='yellow', marker='o', s=20, label='Nose Tip')
            self.ax_3d.scatter([landmarks[chin_idx, 0]], 
                             [landmarks[chin_idx, 1]], 
                             [landmarks[chin_idx, 2]], 
                             c='magenta', marker='o', s=20, label='Chin')
        
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_title(f'Frame {frame_num}')
        self.ax_3d.legend()
        
        # Set equal aspect ratio
        max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
        mid_x = (xs.max()+xs.min()) * 0.5
        mid_y = (ys.max()+ys.min()) * 0.5
        mid_z = (zs.max()+zs.min()) * 0.5
        self.ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
        
        self.canvas_3d.draw()
    
    def align_pupils(self):
        """Align landmarks so pupils are at canonical positions (-31.5, 0, 0) and (31.5, 0, 0)"""
        if self.landmarks_3d_data is None or len(self.landmarks_3d_data) == 0:
            messagebox.showwarning("Warning", "Please load a landmarks CSV file first")
            return
        
        # Get current frame landmarks
        frame_idx = int(self.frame_slider.get())
        if frame_idx >= len(self.landmarks_3d_data):
            frame_idx = len(self.landmarks_3d_data) - 1
        
        frame_num, landmarks = self.landmarks_3d_data[frame_idx]
        
        # Landmark indices
        LEFT_PUPIL_IDX = 468
        RIGHT_PUPIL_IDX = 473
        NOSE_TIP_IDX = 1
        CHIN_IDX = 152
        LEFT_MOUTH_IDX = 61  # Left mouth corner
        RIGHT_MOUTH_IDX = 291  # Right mouth corner
        
        # Extract pupil positions
        left_pupil = landmarks[LEFT_PUPIL_IDX].copy()
        right_pupil = landmarks[RIGHT_PUPIL_IDX].copy()
        
        # Step 1: Translate so midpoint of pupils is at origin
        eye_midpoint = (left_pupil + right_pupil) / 2.0
        landmarks_centered = landmarks - eye_midpoint
        
        # Step 2: Calculate current pupil direction and rotate to align with X-axis in 3D
        pupil_vector = landmarks_centered[RIGHT_PUPIL_IDX] - landmarks_centered[LEFT_PUPIL_IDX]
        
        # Normalize pupil vector
        pupil_vector_norm = pupil_vector / np.linalg.norm(pupil_vector)
        
        # Target vector is X-axis
        target_vector = np.array([1.0, 0.0, 0.0])
        
        # Calculate rotation axis (cross product) and angle
        rotation_axis = np.cross(pupil_vector_norm, target_vector)
        rotation_axis_length = np.linalg.norm(rotation_axis)
        
        if rotation_axis_length > 1e-6:  # Not already aligned
            rotation_axis = rotation_axis / rotation_axis_length
            rotation_angle = np.arccos(np.clip(np.dot(pupil_vector_norm, target_vector), -1.0, 1.0))
            
            # Rodrigues' rotation formula
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            
            rotation_matrix = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * (K @ K)
        else:
            # Already aligned or opposite direction
            if np.dot(pupil_vector_norm, target_vector) < 0:
                # Opposite direction, rotate 180 degrees around Y-axis
                rotation_matrix = np.array([
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, -1]
                ])
            else:
                rotation_matrix = np.eye(3)
        
        landmarks_rotated = landmarks_centered @ rotation_matrix.T
        
        # Step 3: Rotate around X-axis to bring nose tip Y to 0
        nose_tip_rotated = landmarks_rotated[NOSE_TIP_IDX]
        nose_y = nose_tip_rotated[1]
        nose_z = nose_tip_rotated[2]
        
        # Calculate angle to rotate nose to XZ plane (Y=0)
        nose_angle = np.arctan2(nose_y, nose_z)
        
        # Rotation matrix around X-axis
        cos_nose = np.cos(nose_angle)
        sin_nose = np.sin(nose_angle)
        rotation_matrix_x = np.array([
            [1, 0, 0],
            [0, cos_nose, -sin_nose],
            [0, sin_nose, cos_nose]
        ])
        
        landmarks_rotated = landmarks_rotated @ rotation_matrix_x.T
        
        # Step 4: Scale so IPD = 63mm (pupils at ±31.5mm)
        current_ipd = np.linalg.norm(landmarks_rotated[RIGHT_PUPIL_IDX] - landmarks_rotated[LEFT_PUPIL_IDX])
        scale_factor = 63.0 / current_ipd
        
        self.aligned_landmarks = landmarks_rotated * scale_factor
        
        # Extract key landmark positions
        left_pupil_aligned = self.aligned_landmarks[LEFT_PUPIL_IDX]
        right_pupil_aligned = self.aligned_landmarks[RIGHT_PUPIL_IDX]
        nose_tip_aligned = self.aligned_landmarks[NOSE_TIP_IDX]
        chin_aligned = self.aligned_landmarks[CHIN_IDX]
        left_mouth_aligned = self.aligned_landmarks[LEFT_MOUTH_IDX]
        right_mouth_aligned = self.aligned_landmarks[RIGHT_MOUTH_IDX]
        
        # Format text for display
        coords_text = f"""Left Pupil:  ({left_pupil_aligned[0]:7.2f}, {left_pupil_aligned[1]:7.2f}, {left_pupil_aligned[2]:7.2f})
Right Pupil: ({right_pupil_aligned[0]:7.2f}, {right_pupil_aligned[1]:7.2f}, {right_pupil_aligned[2]:7.2f})
Nose Tip:    ({nose_tip_aligned[0]:7.2f}, {nose_tip_aligned[1]:7.2f}, {nose_tip_aligned[2]:7.2f})
Chin:        ({chin_aligned[0]:7.2f}, {chin_aligned[1]:7.2f}, {chin_aligned[2]:7.2f})
L Mouth:     ({left_mouth_aligned[0]:7.2f}, {left_mouth_aligned[1]:7.2f}, {left_mouth_aligned[2]:7.2f})
R Mouth:     ({right_mouth_aligned[0]:7.2f}, {right_mouth_aligned[1]:7.2f}, {right_mouth_aligned[2]:7.2f})"""
        
        # Update text widget
        self.coords_text.delete('1.0', tk.END)
        self.coords_text.insert('1.0', coords_text)
        
        # Update 3D plot with aligned coordinates
        self.update_3d_plot_aligned()
    
    def align_sockets(self):
        """Align landmarks using eye sockets (inner eye corners) at canonical positions (-31.5, 0, 0) and (31.5, 0, 0)"""
        if self.landmarks_3d_data is None or len(self.landmarks_3d_data) == 0:
            messagebox.showwarning("Warning", "Please load a landmarks CSV file first")
            return
        
        # Get current frame landmarks
        frame_idx = int(self.frame_slider.get())
        if frame_idx >= len(self.landmarks_3d_data):
            frame_idx = len(self.landmarks_3d_data) - 1
        
        frame_num, landmarks = self.landmarks_3d_data[frame_idx]
        
        # Landmark indices
        LEFT_EYE_INNER_IDX = 133  # Left eye inner corner
        RIGHT_EYE_INNER_IDX = 362  # Right eye inner corner
        NOSE_TIP_IDX = 1
        CHIN_IDX = 152
        LEFT_MOUTH_IDX = 61  # Left mouth corner
        RIGHT_MOUTH_IDX = 291  # Right mouth corner
        
        # Extract eye corner positions
        left_eye_corner = landmarks[LEFT_EYE_INNER_IDX].copy()
        right_eye_corner = landmarks[RIGHT_EYE_INNER_IDX].copy()
        
        # Step 1: Translate so midpoint of eye corners is at origin
        eye_midpoint = (left_eye_corner + right_eye_corner) / 2.0
        landmarks_centered = landmarks - eye_midpoint
        
        # Step 2: Calculate current eye corner direction and rotate to align with X-axis in 3D
        corner_vector = landmarks_centered[RIGHT_EYE_INNER_IDX] - landmarks_centered[LEFT_EYE_INNER_IDX]
        
        # Normalize corner vector
        corner_vector_norm = corner_vector / np.linalg.norm(corner_vector)
        
        # Target vector is X-axis
        target_vector = np.array([1.0, 0.0, 0.0])
        
        # Calculate rotation axis (cross product) and angle
        rotation_axis = np.cross(corner_vector_norm, target_vector)
        rotation_axis_length = np.linalg.norm(rotation_axis)
        
        if rotation_axis_length > 1e-6:  # Not already aligned
            rotation_axis = rotation_axis / rotation_axis_length
            rotation_angle = np.arccos(np.clip(np.dot(corner_vector_norm, target_vector), -1.0, 1.0))
            
            # Rodrigues' rotation formula
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            
            rotation_matrix = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * (K @ K)
        else:
            # Already aligned or opposite direction
            if np.dot(corner_vector_norm, target_vector) < 0:
                # Opposite direction, rotate 180 degrees around Y-axis
                rotation_matrix = np.array([
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, -1]
                ])
            else:
                rotation_matrix = np.eye(3)
        
        landmarks_rotated = landmarks_centered @ rotation_matrix.T
        
        # Step 3: Rotate around X-axis to bring nose tip Y to 0
        nose_tip_rotated = landmarks_rotated[NOSE_TIP_IDX]
        nose_y = nose_tip_rotated[1]
        nose_z = nose_tip_rotated[2]
        
        # Calculate angle to rotate nose to XZ plane (Y=0)
        nose_angle = np.arctan2(nose_y, nose_z)
        
        # Rotation matrix around X-axis
        cos_nose = np.cos(nose_angle)
        sin_nose = np.sin(nose_angle)
        rotation_matrix_x = np.array([
            [1, 0, 0],
            [0, cos_nose, -sin_nose],
            [0, sin_nose, cos_nose]
        ])
        
        landmarks_rotated = landmarks_rotated @ rotation_matrix_x.T
        
        # Step 4: Scale so distance between eye corners = 63mm (corners at ±31.5mm)
        current_distance = np.linalg.norm(landmarks_rotated[RIGHT_EYE_INNER_IDX] - landmarks_rotated[LEFT_EYE_INNER_IDX])
        scale_factor = 63.0 / current_distance
        
        self.aligned_landmarks = landmarks_rotated * scale_factor
        
        # Extract key landmark positions
        left_corner_aligned = self.aligned_landmarks[LEFT_EYE_INNER_IDX]
        right_corner_aligned = self.aligned_landmarks[RIGHT_EYE_INNER_IDX]
        nose_tip_aligned = self.aligned_landmarks[NOSE_TIP_IDX]
        chin_aligned = self.aligned_landmarks[CHIN_IDX]
        left_mouth_aligned = self.aligned_landmarks[LEFT_MOUTH_IDX]
        right_mouth_aligned = self.aligned_landmarks[RIGHT_MOUTH_IDX]
        
        # Format text for display
        coords_text = f"""Left Socket:  ({left_corner_aligned[0]:7.2f}, {left_corner_aligned[1]:7.2f}, {left_corner_aligned[2]:7.2f})
Right Socket: ({right_corner_aligned[0]:7.2f}, {right_corner_aligned[1]:7.2f}, {right_corner_aligned[2]:7.2f})
Nose Tip:     ({nose_tip_aligned[0]:7.2f}, {nose_tip_aligned[1]:7.2f}, {nose_tip_aligned[2]:7.2f})
Chin:         ({chin_aligned[0]:7.2f}, {chin_aligned[1]:7.2f}, {chin_aligned[2]:7.2f})
L Mouth:      ({left_mouth_aligned[0]:7.2f}, {left_mouth_aligned[1]:7.2f}, {left_mouth_aligned[2]:7.2f})
R Mouth:      ({right_mouth_aligned[0]:7.2f}, {right_mouth_aligned[1]:7.2f}, {right_mouth_aligned[2]:7.2f})"""
        
        # Update text widget
        self.coords_text.delete('1.0', tk.END)
        self.coords_text.insert('1.0', coords_text)
        
        # Update 3D plot with aligned coordinates
        self.update_3d_plot_aligned()
    
    def update_3d_plot_aligned(self):
        """Update 3D plot with aligned landmarks"""
        if self.aligned_landmarks is None:
            return
        
        self.ax_3d.clear()
        
        # Plot all landmarks
        xs = self.aligned_landmarks[:, 0]
        ys = self.aligned_landmarks[:, 1]
        zs = self.aligned_landmarks[:, 2]
        
        self.ax_3d.scatter(xs, ys, zs, c='cyan', marker='.', s=1)
        
        # Highlight key landmarks
        left_pupil_idx = 468
        right_pupil_idx = 473
        nose_tip_idx = 1
        chin_idx = 152
        left_mouth_idx = 61
        right_mouth_idx = 291
        
        self.ax_3d.scatter([self.aligned_landmarks[left_pupil_idx, 0]], 
                         [self.aligned_landmarks[left_pupil_idx, 1]], 
                         [self.aligned_landmarks[left_pupil_idx, 2]], 
                         c='green', marker='o', s=20, label='Left Pupil')
        self.ax_3d.scatter([self.aligned_landmarks[right_pupil_idx, 0]], 
                         [self.aligned_landmarks[right_pupil_idx, 1]], 
                         [self.aligned_landmarks[right_pupil_idx, 2]], 
                         c='red', marker='o', s=20, label='Right Pupil')
        self.ax_3d.scatter([self.aligned_landmarks[nose_tip_idx, 0]], 
                         [self.aligned_landmarks[nose_tip_idx, 1]], 
                         [self.aligned_landmarks[nose_tip_idx, 2]], 
                         c='yellow', marker='o', s=20, label='Nose Tip')
        self.ax_3d.scatter([self.aligned_landmarks[chin_idx, 0]], 
                         [self.aligned_landmarks[chin_idx, 1]], 
                         [self.aligned_landmarks[chin_idx, 2]], 
                         c='magenta', marker='o', s=20, label='Chin')
        self.ax_3d.scatter([self.aligned_landmarks[left_mouth_idx, 0]], 
                         [self.aligned_landmarks[left_mouth_idx, 1]], 
                         [self.aligned_landmarks[left_mouth_idx, 2]], 
                         c='blue', marker='o', s=20, label='L Mouth')
        self.ax_3d.scatter([self.aligned_landmarks[right_mouth_idx, 0]], 
                         [self.aligned_landmarks[right_mouth_idx, 1]], 
                         [self.aligned_landmarks[right_mouth_idx, 2]], 
                         c='orange', marker='o', s=20, label='R Mouth')
        
        self.ax_3d.set_xlabel('X (mm)')
        self.ax_3d.set_ylabel('Y (mm)')
        self.ax_3d.set_zlabel('Z (mm)')
        self.ax_3d.set_title('Aligned Face Landmarks (Canonical Space)')
        self.ax_3d.legend()
        
        # Equal aspect ratio
        max_range = 100  # Roughly face size in mm
        self.ax_3d.set_xlim(-max_range, max_range)
        self.ax_3d.set_ylim(-max_range, max_range)
        self.ax_3d.set_zlim(-max_range, max_range)
        
        self.canvas_3d.draw()
    
    def analyze_face_proportions(self):
        """Analyze face proportions from loaded CSV to calculate canonical model points"""
        if self.landmarks_3d_data is None or len(self.landmarks_3d_data) == 0:
            messagebox.showwarning("Warning", "Please load a landmarks CSV file first")
            return
        
        # Landmark indices
        LEFT_EYE_IDX = 468
        RIGHT_EYE_IDX = 473
        NOSE_TIP_IDX = 1
        CHIN_IDX = 152
        
        print("\n" + "="*60)
        print("FACE PROPORTION ANALYSIS")
        print("="*60)
        
        # Analyze each frame in face-local coordinates
        ipd_values = []
        nose_positions = []
        chin_positions = []
        
        for frame_num, landmarks in self.landmarks_3d_data:
            # Get key points
            left_eye = landmarks[LEFT_EYE_IDX]
            right_eye = landmarks[RIGHT_EYE_IDX]
            nose = landmarks[NOSE_TIP_IDX]
            chin = landmarks[CHIN_IDX]
            
            # Face-local coordinate system
            origin = (left_eye + right_eye) / 2.0
            
            # IPD (distance between eyes)
            ipd = np.linalg.norm(right_eye - left_eye)
            ipd_values.append(ipd)
            
            # Transform to face-local coordinates (normalized by IPD)
            nose_relative = (nose - origin) / ipd
            chin_relative = (chin - origin) / ipd
            
            nose_positions.append(nose_relative)
            chin_positions.append(chin_relative)
        
        # Calculate statistics
        ipd_values = np.array(ipd_values)
        nose_positions = np.array(nose_positions)
        chin_positions = np.array(chin_positions)
        
        # Average nose and chin positions (in units of IPD)
        nose_avg = nose_positions.mean(axis=0)
        chin_avg = chin_positions.mean(axis=0)
        
        nose_std = nose_positions.std(axis=0)
        chin_std = chin_positions.std(axis=0)
        
        print(f"\nAnalyzed {len(self.landmarks_3d_data)} frames")
        print(f"\n📊 IPD Statistics (model units):")
        print(f"   Mean: {ipd_values.mean():.2f}")
        print(f"   Std:  {ipd_values.std():.2f}")
        print(f"   CV:   {(ipd_values.std() / ipd_values.mean() * 100):.2f}%")
        
        print(f"\n📏 Nose Position (relative to eye midpoint, in IPD units):")
        print(f"   X: {nose_avg[0]:+.4f} ± {nose_std[0]:.4f} (left-right)")
        print(f"   Y: {nose_avg[1]:+.4f} ± {nose_std[1]:.4f} (up-down)")
        print(f"   Z: {nose_avg[2]:+.4f} ± {nose_std[2]:.4f} (depth)")
        
        print(f"\n📏 Chin Position (relative to eye midpoint, in IPD units):")
        print(f"   X: {chin_avg[0]:+.4f} ± {chin_std[0]:.4f} (left-right)")
        print(f"   Y: {chin_avg[1]:+.4f} ± {chin_std[1]:.4f} (up-down)")
        print(f"   Z: {chin_avg[2]:+.4f} ± {chin_std[2]:.4f} (depth)")
        
        # Calculate CV for consistency check
        nose_cv = (nose_std / np.abs(nose_avg)).mean() * 100
        chin_cv = (chin_std / np.abs(chin_avg)).mean() * 100
        
        print(f"\n✅ Consistency Check:")
        print(f"   Nose CV: {nose_cv:.2f}% {'(CONSISTENT)' if nose_cv < 5 else '(VARIABLE)'}")
        print(f"   Chin CV: {chin_cv:.2f}% {'(CONSISTENT)' if chin_cv < 5 else '(VARIABLE)'}")
        
        # Assuming average IPD = 63mm, calculate model points
        assumed_ipd_mm = 63.0
        
        print(f"\n🎯 CANONICAL MODEL POINTS (assuming IPD = {assumed_ipd_mm}mm):")
        print(f"\n   model_points = np.array([")
        print(f"       # Left eye center")
        print(f"       ({-assumed_ipd_mm/2:.1f}, 0.0, 0.0),")
        print(f"       # Right eye center")
        print(f"       ({assumed_ipd_mm/2:.1f}, 0.0, 0.0),")
        print(f"       # Nose tip")
        print(f"       ({nose_avg[0] * assumed_ipd_mm:.1f}, {nose_avg[1] * assumed_ipd_mm:.1f}, {nose_avg[2] * assumed_ipd_mm:.1f}),")
        print(f"       # Chin")
        print(f"       ({chin_avg[0] * assumed_ipd_mm:.1f}, {chin_avg[1] * assumed_ipd_mm:.1f}, {chin_avg[2] * assumed_ipd_mm:.1f}),")
        print(f"   ], dtype=np.float32)")
        
        print("\n" + "="*60)
        
        # Show message box with results
        result_text = f"Proportions Analysis:\n\n"
        result_text += f"Nose CV: {nose_cv:.2f}% {'✓ Consistent' if nose_cv < 5 else '✗ Variable'}\n"
        result_text += f"Chin CV: {chin_cv:.2f}% {'✓ Consistent' if chin_cv < 5 else '✗ Variable'}\n\n"
        result_text += f"See console for detailed model points"
        
        messagebox.showinfo("Face Proportion Analysis", result_text)
    
    def load_video(self):
        """Load a video file for processing"""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        
        if filename:
            self.loaded_video_path = filename
            # Show just the filename in the label
            self.loaded_video_label.config(text=f"Loaded: {os.path.basename(filename)}", foreground="green")
            self.run_on_video.set(True)  # Auto-enable the checkbox
    
    def load_crop_images(self):
        """Load crop images from a folder"""
        if self.running:
            self.crop_info_label.config(text="Stop the program first to load images", foreground="red")
            return
        
        folder = filedialog.askdirectory(title="Select Folder with Crop Images")
        
        if folder:
            # Load all image files from the folder
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(folder, ext)))
            
            image_files.sort()  # Sort files by name
            
            if image_files:
                self.loaded_crops = image_files
                self.current_crop_index = 0
                
                # Enable navigation and run buttons
                self.prev_crop_button.config(state=tk.NORMAL)
                self.next_crop_button.config(state=tk.NORMAL)
                self.run_landmarks_button.config(state=tk.NORMAL)
                
                self.crop_info_label.config(text=f"Loaded {len(image_files)} images", foreground="green")
                
                # Display first image without landmarks
                self.display_crop_image()
            else:
                self.crop_info_label.config(text="No images found in folder", foreground="red")
    
    def previous_crop(self):
        """Move to previous crop image"""
        if self.running:
            return
        
        if self.loaded_crops and self.current_crop_index > 0:
            self.current_crop_index -= 1
            self.display_crop_image()
    
    def next_crop(self):
        """Move to next crop image"""
        if self.running:
            return
        
        if self.loaded_crops and self.current_crop_index < len(self.loaded_crops) - 1:
            self.current_crop_index += 1
            self.display_crop_image()
    
    def display_crop_image(self):
        """Display current crop image without processing"""
        if not self.loaded_crops:
            return
        
        image_path = self.loaded_crops[self.current_crop_index]
        image = cv2.imread(image_path)
        
        if image is not None:
            # Update info label
            filename = os.path.basename(image_path)
            self.crop_info_label.config(
                text=f"Image {self.current_crop_index + 1}/{len(self.loaded_crops)}: {filename}",
                foreground="blue"
            )
            
            # Display image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (256, 256))
            img = Image.fromarray(image_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            self.crop_label.imgtk = imgtk
            self.crop_label.configure(image=imgtk)
    
    def run_landmarks_on_crop(self):
        """Run landmarks detection on current crop image"""
        global landmarks_interpreter, landmarks_input_details, landmarks_output_details
        global landmarks_w, landmarks_h
        
        if self.running:
            self.crop_info_label.config(text="Stop the program first to process images", foreground="red")
            return
        
        if not self.loaded_crops:
            return
        
        image_path = self.loaded_crops[self.current_crop_index]
        crop = cv2.imread(image_path)
        
        if crop is None:
            return
        
        # Run landmarks detection
        landmarks_input = cv2.resize(crop, (landmarks_w, landmarks_h))
        landmarks_input = landmarks_input.astype(np.float32) / 255.0
        landmarks_input = np.expand_dims(landmarks_input, axis=0)
        
        landmarks_interpreter.set_tensor(landmarks_input_details[0]['index'], landmarks_input)
        landmarks_interpreter.invoke()
        
        landmarks_raw = landmarks_interpreter.get_tensor(landmarks_output_details[0]['index'])[0]
        landmarks = landmarks_raw.reshape(-1, 3)
        
        # Get score if available
        landmarks_score = None
        if len(landmarks_output_details) > 1:
            landmarks_score_raw = landmarks_interpreter.get_tensor(landmarks_output_details[1]['index'])
            landmarks_score = float(landmarks_score_raw.flatten()[0])
        
        # Draw landmarks on crop
        display_crop = crop.copy()
        crop_h, crop_w = display_crop.shape[:2]
        
        print(f"Crop size: {crop_w}x{crop_h}, Landmarks shape: {landmarks.shape}")
        print(f"First 5 landmarks: {landmarks[:5]}")
        
        # Draw all landmarks - coordinates are already in pixels, not normalized!
        num_drawn = 0
        for i, (lx, ly, lz) in enumerate(landmarks):
            x = int(lx)  # Already in pixels
            y = int(ly)  # Already in pixels
            # Draw larger, more visible circles
            cv2.circle(display_crop, (x, y), 1, (0, 255, 0), -1)
            num_drawn += 1
        
        print(f"Drew {num_drawn} landmarks")
        
        # Display score if available
        if landmarks_score is not None:
            cv2.putText(display_crop, f"Score: {landmarks_score:.3f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the crop with landmarks
        display_crop_rgb = cv2.cvtColor(display_crop, cv2.COLOR_BGR2RGB)
        display_crop_resized = cv2.resize(display_crop_rgb, (256, 256))
        img = Image.fromarray(display_crop_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        self.crop_label.imgtk = imgtk
        self.crop_label.configure(image=imgtk)
        
        # Update info label with score
        filename = os.path.basename(image_path)
        score_text = f" (Score: {landmarks_score:.3f})" if landmarks_score is not None else ""
        self.crop_info_label.config(
            text=f"Image {self.current_crop_index + 1}/{len(self.loaded_crops)}: {filename}{score_text}",
            foreground="green"
        )
    
    def start_program(self):
        global cap, running
        if not self.running:
            # Check if we should use loaded video or camera
            if self.run_on_video.get() and self.loaded_video_path:
                cap = cv2.VideoCapture(self.loaded_video_path)
                if not cap.isOpened():
                    self.status_label.config(text="Error: Cannot open video file", foreground="red")
                    return
            else:
                cap = cv2.VideoCapture(1)
                if not cap.isOpened():
                    self.status_label.config(text="Error: Cannot open camera", foreground="red")
                    return
                # Set camera resolution to 1280x480 for side-by-side
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            running = True
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Running", foreground="green")
            
            # Start processing thread
            self.thread = threading.Thread(target=self.process_video, daemon=True)
            self.thread.start()
            
            # Start GUI update loop
            self.update_gui()
    
    def stop_program(self):
        global cap, running, video_writer, crop_writer, recording_video, recording_crop
        global recording_images, image_save_counter, image_save_folder
        global landmarks_csv_file, landmarks_csv_writer
        global video_landmarks_csv_file, video_landmarks_csv_writer, video_frame_counter
        running = False
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped", foreground="red")
        
        # Stop recording if active
        if recording_video and video_writer is not None:
            video_writer.release()
            video_writer = None
            recording_video = False
            self.record_video_button.config(text="Record Video")
            
            # Close video CSV file
            if video_landmarks_csv_file is not None:
                video_landmarks_csv_file.close()
                video_landmarks_csv_file = None
                video_landmarks_csv_writer = None
                video_frame_counter = 0
        
        if recording_crop and crop_writer is not None:
            crop_writer.release()
            crop_writer = None
            recording_crop = False
            self.record_crop_button.config(text="Record Crop")
        
        if recording_images:
            recording_images = False
            
            # Close CSV file
            if landmarks_csv_file is not None:
                landmarks_csv_file.close()
                landmarks_csv_file = None
                landmarks_csv_writer = None
            
            self.record_images_button.config(text="Record Images")
            print(f"Stopped recording images. Saved {image_save_counter} images")
            image_save_folder = None
            image_save_counter = 0
        
        if cap is not None:
            cap.release()
    
    def toggle_record_video(self):
        global recording_video, video_writer
        global video_landmarks_csv_file, video_landmarks_csv_writer, video_frame_counter
        if not self.running:
            return
        
        if not recording_video:
            # Start recording
            import csv
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))
            recording_video = True
            video_frame_counter = 0
            
            # Create CSV file for landmarks
            csv_filename = f"video_{timestamp}_landmarks.csv"
            video_landmarks_csv_file = open(csv_filename, 'w', newline='')
            video_landmarks_csv_writer = csv.writer(video_landmarks_csv_file)
            
            # Write header
            header = ['frame_number']
            for i in range(478):  # 478 landmarks
                header.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
            video_landmarks_csv_writer.writerow(header)
            
            self.record_video_button.config(text="Stop Recording Video")
            print(f"Started recording video: {filename} and landmarks: {csv_filename}")
        else:
            # Stop recording
            recording_video = False
            if video_writer is not None:
                video_writer.release()
                video_writer = None
            
            # Close CSV file
            if video_landmarks_csv_file is not None:
                video_landmarks_csv_file.close()
                video_landmarks_csv_file = None
                video_landmarks_csv_writer = None
                print(f"Stopped recording video. Recorded {video_frame_counter} frames")
                video_frame_counter = 0
            
            self.record_video_button.config(text="Record Video")
    
    def toggle_record_crop(self):
        global recording_crop, crop_writer
        if not self.running:
            return
        
        if not recording_crop:
            # Start recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crop_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            crop_writer = cv2.VideoWriter(filename, fourcc, 30.0, (256, 256))
            recording_crop = True
            self.record_crop_button.config(text="Stop Recording Crop")
            print(f"Started recording crop: {filename}")
        else:
            # Stop recording
            recording_crop = False
            if crop_writer is not None:
                crop_writer.release()
                crop_writer = None
            self.record_crop_button.config(text="Record Crop")
            print("Stopped recording crop")
    
    def toggle_record_images(self):
        global recording_images, image_save_counter, image_save_folder
        global landmarks_csv_file, landmarks_csv_writer
        if not self.running:
            return
        
        if not recording_images:
            # Start recording images
            import os
            import csv
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_save_folder = f"dataset/{timestamp}"
            os.makedirs(image_save_folder, exist_ok=True)
            image_save_counter = 0
            
            # Create CSV file for landmarks
            csv_path = f"{image_save_folder}/landmarks.csv"
            landmarks_csv_file = open(csv_path, 'w', newline='')
            landmarks_csv_writer = csv.writer(landmarks_csv_file)
            
            # Write header
            header = ['image_filename']
            for i in range(478):  # 478 landmarks
                header.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
            landmarks_csv_writer.writerow(header)
            
            recording_images = True
            self.record_images_button.config(text="Stop Recording Images")
            print(f"Started recording images to: {image_save_folder}")
        else:
            # Stop recording images
            recording_images = False
            
            # Close CSV file
            if landmarks_csv_file is not None:
                landmarks_csv_file.close()
                landmarks_csv_file = None
                landmarks_csv_writer = None
            
            self.record_images_button.config(text="Record Images")
            print(f"Stopped recording images. Saved {image_save_counter} images to {image_save_folder}")
            image_save_folder = None
            image_save_counter = 0
    
    def process_video(self):
        global cap, running, tracked_bbox, tracking_active, frame_count
        global recording_video, recording_crop, video_writer, crop_writer
        
        tracked_bbox = None
        tracking_active = False
        frame_count = 0
        
        while running and cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame_count += 1
            h, w, _ = frame.shape
            # If camera is set to 1280x480, extract left image (640x480)
            if w == 1280 and h == 480:
                left_img = frame[:, :640]
                display_frame = left_img.copy()
            else:
                display_frame = frame.copy()
            
            # Decide whether to run face detection
            run_detection = False
            if not tracking_active:
                if frame_count == 1 or frame_count % detect_interval == 0:
                    run_detection = True
            landmarks_display = None
            landmarks_display_clean = None
            # Run face detection if needed
            # Use left_img if available, else frame
            process_img = display_frame
            if run_detection:
                frame_rgb = cv2.cvtColor(process_img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                results = detector.detect(mp_image)
                if results.detections:
                    detection = results.detections[0]
                    bbox = detection.bounding_box
                    x = bbox.origin_x
                    y = bbox.origin_y
                    width = bbox.width
                    height = bbox.height
                    
                    tracked_bbox = {
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height,
                        'confidence': detection.categories[0].score
                    }
                    tracking_active = True
                    cv2.putText(display_frame, "DETECTING", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    tracking_active = False
                    tracked_bbox = None
            else:
                if tracked_bbox is not None:
                    cv2.putText(display_frame, "TRACKING", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Process landmarks if we have a bounding box
            if tracked_bbox is not None:
                x = tracked_bbox['x']
                y = tracked_bbox['y']
                width = tracked_bbox['width']
                height = tracked_bbox['height']
                
                center_x = x + width / 2
                center_y = y + height / 2
                
                # First, get a quick crop to detect head orientation
                scale_factor = self.scale_factor_var.get()
                scaled_width = width * scale_factor
                scaled_height = height * scale_factor
                scaled_x = int(center_x - scaled_width / 2)
                scaled_y = int(center_y - scaled_height / 2)
                scaled_x2 = int(center_x + scaled_width / 2)
                scaled_y2 = int(center_y + scaled_height / 2)
                
                scaled_x = max(0, scaled_x)
                scaled_y = max(0, scaled_y)
                scaled_x2 = min(w, scaled_x2)
                scaled_y2 = min(h, scaled_y2)
                
                cropped_temp = frame[scaled_y:scaled_y2, scaled_x:scaled_x2]
                
                if cropped_temp.size > 0:
                    cropped_rgb_temp = cv2.cvtColor(cropped_temp, cv2.COLOR_BGR2RGB)
                    landmarks_input_temp = cv2.resize(cropped_rgb_temp, (landmarks_w, landmarks_h))
                    
                    # First pass: detect landmarks to get head orientation
                    landmarks_input_first = (landmarks_input_temp.astype(np.float32) / 127.5) - 1.0
                    landmarks_input_first = np.expand_dims(landmarks_input_first, axis=0)
                    
                    landmarks_interpreter.set_tensor(landmarks_input_details[0]['index'], landmarks_input_first)
                    landmarks_interpreter.invoke()
                    
                    landmarks_raw_first = landmarks_interpreter.get_tensor(landmarks_output_details[0]['index'])[0]
                    landmarks_first = landmarks_raw_first.reshape(-1, 3)
                    
                    # Calculate head orientation using eye landmarks
                    left_eye_idx = 33
                    right_eye_idx = 263
                    
                    angle_deg = 0
                    if len(landmarks_first) > max(left_eye_idx, right_eye_idx):
                        left_eye = landmarks_first[left_eye_idx]
                        right_eye = landmarks_first[right_eye_idx]
                        
                        dx = right_eye[0] - left_eye[0]
                        dy = right_eye[1] - left_eye[1]
                        angle_rad = np.arctan2(dy, dx)
                        angle_deg = np.degrees(angle_rad)
                    
                    # Now rotate the bounding box and crop from the rotated frame
                    # Get rotation matrix for the full frame around the face center
                    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle_deg, 1.0)
                    
                    # Rotate the entire frame
                    frame_rotated = cv2.warpAffine(frame, rotation_matrix, (w, h))
                    
                    # Crop from the rotated frame using the same bounding box coordinates
                    cropped = frame_rotated[scaled_y:scaled_y2, scaled_x:scaled_x2]
                    
                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    landmarks_input_resized = cv2.resize(cropped_rgb, (landmarks_w, landmarks_h))
                    
                    # Create display copies
                    landmarks_display = landmarks_input_resized.copy()
                    landmarks_display_clean = landmarks_input_resized.copy()
                    
                    # Second pass: detect landmarks on rotated crop
                    landmarks_input = (landmarks_input_resized.astype(np.float32) / 127.5) - 1.0
                    landmarks_input = np.expand_dims(landmarks_input, axis=0)
                    
                    landmarks_interpreter.set_tensor(landmarks_input_details[0]['index'], landmarks_input)
                    landmarks_interpreter.invoke()
                    
                    landmarks_raw = landmarks_interpreter.get_tensor(landmarks_output_details[0]['index'])[0]
                    
                    landmarks_score = None
                    if len(landmarks_output_details) > 1:
                        landmarks_score_raw = landmarks_interpreter.get_tensor(landmarks_output_details[1]['index'])
                        landmarks_score = float(landmarks_score_raw.flatten()[0])
                    
                    landmarks = landmarks_raw.reshape(-1, 3)
                    
                    if landmarks_score is not None and landmarks_score < 0:
                        tracking_active = False
                        tracked_bbox = None
                        print(f"Tracking lost: score={landmarks_score:.3f}")
                        landmarks_display = None
                        landmarks_display_clean = None
                    else:
                        landmarks_x = landmarks[:, 0]
                        landmarks_y = landmarks[:, 1]
                        
                        min_x = np.min(landmarks_x)
                        max_x = np.max(landmarks_x)
                        min_y = np.min(landmarks_y)
                        max_y = np.max(landmarks_y)
                        
                        scale_x = (scaled_x2 - scaled_x) / landmarks_w
                        scale_y = (scaled_y2 - scaled_y) / landmarks_h
                        
                        predicted_x = int(scaled_x + min_x * scale_x)
                        predicted_y = int(scaled_y + min_y * scale_y)
                        predicted_width = int((max_x - min_x) * scale_x)
                        predicted_height = int((max_y - min_y) * scale_y)
                        
                        predicted_size = max(predicted_width, predicted_height)
                        
                        predicted_center_x = predicted_x + predicted_width / 2
                        predicted_center_y = predicted_y + predicted_height / 2
                        predicted_x = int(predicted_center_x - predicted_size / 2)
                        predicted_y = int(predicted_center_y - predicted_size / 2)
                        predicted_width = predicted_size
                        predicted_height = predicted_size
                        
                        padding = 0.1
                        predicted_x = int(predicted_x - predicted_width * padding)
                        predicted_y = int(predicted_y - predicted_height * padding)
                        predicted_width = int(predicted_width * (1 + 2 * padding))
                        predicted_height = int(predicted_height * (1 + 2 * padding))
                        
                        predicted_x = max(0, predicted_x)
                        predicted_y = max(0, predicted_y)
                        predicted_width = min(w - predicted_x, predicted_width)
                        predicted_height = min(h - predicted_y, predicted_height)
                        
                        tracked_bbox = {
                            'x': predicted_x,
                            'y': predicted_y,
                            'width': predicted_width,
                            'height': predicted_height,
                            'confidence': landmarks_score if landmarks_score is not None else 0.5
                        }
                        
                        if landmarks_score is not None:
                            cv2.putText(display_frame, f"Score: {landmarks_score:.3f}", (10, 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        num_landmarks = landmarks.shape[0]
                        
                        for i in range(num_landmarks):
                            x_pixel = landmarks[i, 0]
                            y_pixel = landmarks[i, 1]
                            
                            x_pixel = max(0, min(landmarks_w - 1, x_pixel))
                            y_pixel = max(0, min(landmarks_h - 1, y_pixel))
                            
                            cv2.circle(landmarks_display, 
                                      (int(x_pixel * 16), int(y_pixel * 16)), 
                                      32,
                                      (0, 255, 255), 
                                      -1, 
                                      cv2.LINE_AA,
                                      shift=4)
                        
                        # Draw rotated bounding boxes on original frame
                        # Calculate the four corners of the scaled bounding box
                        box_corners = np.array([
                            [scaled_x, scaled_y],
                            [scaled_x2, scaled_y],
                            [scaled_x2, scaled_y2],
                            [scaled_x, scaled_y2]
                        ], dtype=np.float32)
                        
                        # Create rotation matrix around face center to visualize the rotated crop
                        # This shows where the crop would be if we rotate the box
                        vis_rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), -angle_deg, 1.0)
                        
                        # Calculate and display pupil positions on original frame
                        left_pupil_idx = 468  # Left iris center
                        right_pupil_idx = 473  # Right iris center
                        
                        if num_landmarks > max(left_pupil_idx, right_pupil_idx):
                            # Get pupil positions in crop space
                            left_pupil_crop = landmarks[left_pupil_idx]
                            right_pupil_crop = landmarks[right_pupil_idx]
                            
                            # Transform to original frame coordinates
                            crop_h, crop_w = cropped.shape[:2]
                            scale_x_pupil = crop_w / landmarks_w
                            scale_y_pupil = crop_h / landmarks_h
                            
                            # Left pupil
                            lx_crop = left_pupil_crop[0] * scale_x_pupil
                            ly_crop = left_pupil_crop[1] * scale_y_pupil
                            lz = left_pupil_crop[2]  # Z coordinate from model
                            lx_rotated = scaled_x + lx_crop
                            ly_rotated = scaled_y + ly_crop
                            point_rotated = np.array([[lx_rotated, ly_rotated, 1.0]])
                            point_original = vis_rotation_matrix.dot(point_rotated.T).T
                            left_pupil_x = int(point_original[0, 0])
                            left_pupil_y = int(point_original[0, 1])
                            
                            # Right pupil
                            rx_crop = right_pupil_crop[0] * scale_x_pupil
                            ry_crop = right_pupil_crop[1] * scale_y_pupil
                            rz = right_pupil_crop[2]  # Z coordinate from model
                            rx_rotated = scaled_x + rx_crop
                            ry_rotated = scaled_y + ry_crop
                            point_rotated = np.array([[rx_rotated, ry_rotated, 1.0]])
                            point_original = vis_rotation_matrix.dot(point_rotated.T).T
                            right_pupil_x = int(point_original[0, 0])
                            right_pupil_y = int(point_original[0, 1])
                            
                            # Display coordinates under score
                            cv2.putText(display_frame, f"L Pupil: ({left_pupil_x}, {left_pupil_y}, {lz:.2f})", (10, 90),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.putText(display_frame, f"R Pupil: ({right_pupil_x}, {right_pupil_y}, {rz:.2f})", (10, 120),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Transform corners using rotation matrix
                        ones = np.ones((4, 1))
                        box_corners_homogeneous = np.hstack([box_corners, ones])
                        box_corners_rotated = vis_rotation_matrix.dot(box_corners_homogeneous.T).T
                        box_corners_rotated = box_corners_rotated.astype(np.int32)
                        
                        # Draw rotated blue rectangle (scaled processing box)
                        cv2.polylines(display_frame, [box_corners_rotated], True, (255, 0, 0), 2)
                        
                        # Draw rotated green rectangle (original face box)
                        box_corners_orig = np.array([
                            [x, y],
                            [x + width, y],
                            [x + width, y + height],
                            [x, y + height]
                        ], dtype=np.float32)
                        
                        ones_orig = np.ones((4, 1))
                        box_corners_orig_homogeneous = np.hstack([box_corners_orig, ones_orig])
                        box_corners_orig_rotated = vis_rotation_matrix.dot(box_corners_orig_homogeneous.T).T
                        box_corners_orig_rotated = box_corners_orig_rotated.astype(np.int32)
                        
                        cv2.polylines(display_frame, [box_corners_orig_rotated], True, (0, 255, 0), 2)
                        
                        confidence = tracked_bbox.get('confidence', 0.0)
                        # Put text at the first corner of the rotated green box
                        cv2.putText(display_frame, f"{confidence:.2f}", 
                                   tuple(box_corners_orig_rotated[0]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Draw landmarks on original video frame if checkbox is enabled
                        if self.show_landmarks_on_video.get():
                            # Transform landmarks from crop space back to original frame
                            crop_h, crop_w = cropped.shape[:2]
                            scale_x = crop_w / landmarks_w
                            scale_y = crop_h / landmarks_h
                            
                            # Pupil landmark indices
                            left_pupil_idx = 468
                            right_pupil_idx = 473
                            
                            for i in range(num_landmarks):
                                # Scale from model output (256x256) to crop size
                                lx_crop = landmarks[i, 0] * scale_x
                                ly_crop = landmarks[i, 1] * scale_y
                                
                                # Translate to position in rotated frame
                                lx_rotated = scaled_x + lx_crop
                                ly_rotated = scaled_y + ly_crop
                                
                                # Apply inverse rotation to get original frame coordinates
                                point_rotated = np.array([[lx_rotated, ly_rotated, 1.0]])
                                point_original = vis_rotation_matrix.dot(point_rotated.T).T
                                
                                x_orig = int(point_original[0, 0])
                                y_orig = int(point_original[0, 1])
                                
                                # Draw on original frame
                                if 0 <= x_orig < w and 0 <= y_orig < h:
                                    # Draw larger circles and labels for pupils
                                    if i == left_pupil_idx:
                                        cv2.circle(display_frame, (x_orig, y_orig), 5, (0, 255, 0), -1)
                                        cv2.putText(display_frame, f"L", (x_orig + 10, y_orig - 10),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    elif i == right_pupil_idx:
                                        cv2.circle(display_frame, (x_orig, y_orig), 5, (0, 255, 0), -1)
                                        cv2.putText(display_frame, f"R", (x_orig + 10, y_orig - 10),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    else:
                                        cv2.circle(display_frame, (x_orig, y_orig), 2, (0, 255, 255), -1)
            
            # Record if needed
            if recording_video and video_writer is not None:
                video_writer.write(frame)  # Record raw video without overlays
                
                # Write landmarks to CSV
                global video_landmarks_csv_writer, video_frame_counter
                if video_landmarks_csv_writer is not None and 'landmarks' in locals():
                    video_frame_counter += 1
                    landmarks_flat = landmarks.flatten().tolist()
                    video_landmarks_csv_writer.writerow([video_frame_counter] + landmarks_flat)
            
            if recording_crop and crop_writer is not None:
                # Record based on landmarks checkbox state
                if self.show_landmarks.get() and landmarks_display is not None:
                    crop_writer.write(cv2.cvtColor(landmarks_display, cv2.COLOR_RGB2BGR))
                elif not self.show_landmarks.get() and landmarks_display_clean is not None:
                    crop_writer.write(cv2.cvtColor(landmarks_display_clean, cv2.COLOR_RGB2BGR))
            
            # Save individual images if recording
            if recording_images and image_save_folder is not None:
                global image_save_counter, landmarks_csv_writer
                # Choose which version to save based on landmarks checkbox
                if self.show_landmarks.get() and landmarks_display is not None:
                    save_image = cv2.cvtColor(landmarks_display, cv2.COLOR_RGB2BGR)
                elif not self.show_landmarks.get() and landmarks_display_clean is not None:
                    save_image = cv2.cvtColor(landmarks_display_clean, cv2.COLOR_RGB2BGR)
                else:
                    save_image = None
                
                if save_image is not None and 'landmarks' in locals():
                    image_save_counter += 1
                    image_filename = f"{image_save_counter:06d}.png"
                    full_path = f"{image_save_folder}/{image_filename}"
                    cv2.imwrite(full_path, save_image)
                    
                    # Save landmarks to CSV
                    if landmarks_csv_writer is not None:
                        # Flatten landmarks array to 1D list
                        landmarks_flat = landmarks.flatten().tolist()
                        # Write row: [filename, x0, y0, z0, x1, y1, z1, ...]
                        landmarks_csv_writer.writerow([image_filename] + landmarks_flat)
            
            # Store frames in buffer for GUI update
            with self.frame_lock:
                self.current_frame = display_frame.copy()
                self.current_crop = landmarks_display.copy() if landmarks_display is not None else None
                self.current_crop_no_landmarks = landmarks_display_clean.copy() if landmarks_display_clean is not None else None
    
    def update_gui(self):
        """Update GUI displays from buffer (runs on main thread)"""
        if self.running:
            with self.frame_lock:
                if self.current_frame is not None:
                    # Convert and display main frame
                    frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (640, 480))
                    img = Image.fromarray(frame_resized)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)
                
                # Convert and display crop
                # Choose crop version based on checkbox
                crop_to_display = None
                if self.show_landmarks.get():
                    crop_to_display = self.current_crop
                else:
                    crop_to_display = self.current_crop_no_landmarks
                
                if crop_to_display is not None:
                    crop_resized = cv2.resize(crop_to_display, (256, 256))
                    img_crop = Image.fromarray(crop_resized)
                    imgtk_crop = ImageTk.PhotoImage(image=img_crop)
                    self.crop_label.imgtk = imgtk_crop
                    self.crop_label.configure(image=imgtk_crop)
                else:
                    # Show blank if no crop available
                    blank = np.zeros((256, 256, 3), dtype=np.uint8)
                    img_crop = Image.fromarray(blank)
                    imgtk_crop = ImageTk.PhotoImage(image=img_crop)
                    self.crop_label.imgtk = imgtk_crop
                    self.crop_label.configure(image=imgtk_crop)
        
        # Schedule next update (30 FPS = ~33ms delay)
        self.root.after(33, self.update_gui)
    
    def batch_process_videos(self):
        """Process all videos in a selected folder"""
        import csv
        
        # Select folder with videos
        folder = filedialog.askdirectory(title="Select Folder with Videos")
        if not folder:
            return
        
        # Find all video files
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(folder, ext)))
        
        if not video_files:
            self.status_label.config(text="No video files found in folder", foreground="red")
            return
        
        self.status_label.config(text=f"Found {len(video_files)} videos. Processing...", foreground="blue")
        self.root.update()
        
        # Process each video
        for video_idx, video_path in enumerate(video_files):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # Create separate folders for left and right
            output_folder_left = os.path.join('dataset', f"{video_name}_left")
            output_folder_right = os.path.join('dataset', f"{video_name}_right")
            os.makedirs(output_folder_left, exist_ok=True)
            os.makedirs(output_folder_right, exist_ok=True)
            
            self.status_label.config(
                text=f"Processing video {video_idx + 1}/{len(video_files)}: {video_name}",
                foreground="blue"
            )
            self.root.update()
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open {video_path}")
                continue
            
            # Open CSV files for left and right
            csv_path_left = os.path.join(output_folder_left, 'landmarks.csv')
            csv_file_left = open(csv_path_left, 'w', newline='')
            fieldnames = ['filename'] + [f'landmark_{i}' for i in range(1434)]
            csv_writer_left = csv.DictWriter(csv_file_left, fieldnames=fieldnames)
            csv_writer_left.writeheader()
            
            csv_path_right = os.path.join(output_folder_right, 'landmarks.csv')
            csv_file_right = open(csv_path_right, 'w', newline='')
            csv_writer_right = csv.DictWriter(csv_file_right, fieldnames=fieldnames)
            csv_writer_right.writeheader()
            
            frame_idx = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Update status every 30 frames
                if frame_idx % 30 == 0:
                    self.status_label.config(
                        text=f"Video {video_idx + 1}/{len(video_files)}: {video_name} - Frame {frame_idx}/{total_frames}",
                        foreground="blue"
                    )
                    self.root.update()
                
                # Split frame into left and right halves
                frame_h, frame_w = frame.shape[:2]
                mid_x = frame_w // 2
                frame_left = frame[:, :mid_x]
                frame_right = frame[:, mid_x:]
                
                # Process left side
                rgb_frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
                mp_image_left = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame_left)
                detection_result_left = detector.detect(mp_image_left)
                
                if detection_result_left.detections:
                    self._process_face_detection(
                        detection_result_left.detections[0],
                        frame_left,
                        frame_idx,
                        output_folder_left,
                        csv_writer_left
                    )
                
                # Process right side
                rgb_frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
                mp_image_right = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame_right)
                detection_result_right = detector.detect(mp_image_right)
                
                if detection_result_right.detections:
                    self._process_face_detection(
                        detection_result_right.detections[0],
                        frame_right,
                        frame_idx,
                        output_folder_right,
                        csv_writer_right
                    )
                
                frame_idx += 1
            
            cap.release()
            csv_file_left.close()
            csv_file_right.close()
            
            print(f"Processed {video_name}: {frame_idx} frames, saved to {output_folder_left} and {output_folder_right}")
        
        self.status_label.config(
            text=f"Batch processing complete! Processed {len(video_files)} videos.",
            foreground="green"
        )
    
    def _process_face_detection(self, detection, frame, frame_idx, output_folder, csv_writer):
        """Helper method to process a single face detection"""
        bbox = detection.bounding_box
        
        x = int(bbox.origin_x)
        y = int(bbox.origin_y)
        w = int(bbox.width)
        h = int(bbox.height)
        
        # Extract face crop with margin
        scale_factor = self.scale_factor_var.get()
        center_x = x + w // 2
        center_y = y + h // 2
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        x1 = max(0, center_x - new_w // 2)
        y1 = max(0, center_y - new_h // 2)
        x2 = min(frame.shape[1], center_x + new_w // 2)
        y2 = min(frame.shape[0], center_y + new_h // 2)
        
        face_crop = frame[y1:y2, x1:x2]
        
        if face_crop.size > 0:
            # Run landmarks detection
            landmarks_input = cv2.resize(face_crop, (landmarks_w, landmarks_h))
            landmarks_input_normalized = landmarks_input.astype(np.float32) / 255.0
            landmarks_input_batch = np.expand_dims(landmarks_input_normalized, axis=0)
            
            landmarks_interpreter.set_tensor(landmarks_input_details[0]['index'], landmarks_input_batch)
            landmarks_interpreter.invoke()
            
            landmarks_raw = landmarks_interpreter.get_tensor(landmarks_output_details[0]['index'])[0]
            landmarks_flat = landmarks_raw.flatten()  # Ensure 1D array
            
            # Save image (256x256)
            image_filename = f"{frame_idx:06d}.png"
            image_path = os.path.join(output_folder, image_filename)
            cv2.imwrite(image_path, landmarks_input)
            
            # Save landmarks to CSV
            row_data = {'filename': image_filename}
            for i, val in enumerate(landmarks_flat):
                row_data[f'landmark_{i}'] = float(val)
            csv_writer.writerow(row_data)

# Create and run GUI
root = tk.Tk()
app = FaceDetectionGUI(root)
root.mainloop()