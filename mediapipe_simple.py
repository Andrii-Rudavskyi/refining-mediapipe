import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
from datetime import datetime

# Model paths
model_path = 'blaze_face_short_range.tflite'
landmarks_model_path = 'model/face_landmarker/face_landmarks_detector.tflite'

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
video_writer = None
crop_writer = None
tracked_bbox = None
tracking_active = False
frame_count = 0
detect_interval = 1

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
        
        crop_frame = ttk.LabelFrame(main_frame, text="Cropped Face with Landmarks", padding="5")
        crop_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.crop_label = ttk.Label(crop_frame)
        self.crop_label.pack()
        
        # Checkbox for landmarks overlay
        self.show_landmarks = tk.BooleanVar(value=True)
        landmarks_check = ttk.Checkbutton(crop_frame, text="Landmarks", variable=self.show_landmarks)
        landmarks_check.pack(pady=5)
        
        # Create parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="5")
        params_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
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
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Create buttons
        self.start_button = ttk.Button(button_frame, text="Start Program", command=self.start_program)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Program", command=self.stop_program, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)
        
        self.record_video_button = ttk.Button(button_frame, text="Record Video", command=self.toggle_record_video)
        self.record_video_button.grid(row=0, column=2, padx=5)
        
        self.record_crop_button = ttk.Button(button_frame, text="Record Crop", command=self.toggle_record_crop)
        self.record_crop_button.grid(row=0, column=3, padx=5)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Status: Stopped", foreground="red")
        self.status_label.grid(row=3, column=0, columnspan=2, pady=5)
        
        self.running = False
        self.thread = None
        
        # Frame buffers for thread-safe GUI updates
        self.current_frame = None
        self.current_crop = None
        self.current_crop_no_landmarks = None
        self.current_landmarks = None
        self.frame_lock = threading.Lock()
        
    def start_program(self):
        global cap, running
        if not self.running:
            cap = cv2.VideoCapture(1)
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
        
        if recording_crop and crop_writer is not None:
            crop_writer.release()
            crop_writer = None
            recording_crop = False
            self.record_crop_button.config(text="Record Crop")
        
        if cap is not None:
            cap.release()
    
    def toggle_record_video(self):
        global recording_video, video_writer
        if not self.running:
            return
        
        if not recording_video:
            # Start recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))
            recording_video = True
            self.record_video_button.config(text="Stop Recording Video")
            print(f"Started recording video: {filename}")
        else:
            # Stop recording
            recording_video = False
            if video_writer is not None:
                video_writer.release()
                video_writer = None
            self.record_video_button.config(text="Record Video")
            print("Stopped recording video")
    
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
            
            # Resize frame for display if needed
            display_frame = frame.copy()
            
            # Decide whether to run face detection
            run_detection = False
            if not tracking_active:
                if frame_count == 1 or frame_count % detect_interval == 0:
                    run_detection = True
            
            landmarks_display = None
            
            # Run face detection if needed
            if run_detection:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                
                cropped = frame[scaled_y:scaled_y2, scaled_x:scaled_x2]
                
                if cropped.size > 0:
                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    landmarks_input_resized = cv2.resize(cropped_rgb, (landmarks_w, landmarks_h))
                    landmarks_display = landmarks_input_resized.copy()
                    landmarks_display_clean = landmarks_input_resized.copy()  # Copy without landmarks
                    
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
                        
                        cv2.rectangle(display_frame, (scaled_x, scaled_y), (scaled_x2, scaled_y2), (255, 0, 0), 2)
                        cv2.rectangle(display_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                        
                        confidence = tracked_bbox.get('confidence', 0.0)
                        cv2.putText(display_frame, f"{confidence:.2f}", (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Record if needed
            if recording_video and video_writer is not None:
                video_writer.write(display_frame)
            
            if recording_crop and crop_writer is not None:
                # Record based on landmarks checkbox state
                if self.show_landmarks.get() and landmarks_display is not None:
                    crop_writer.write(cv2.cvtColor(landmarks_display, cv2.COLOR_RGB2BGR))
                elif not self.show_landmarks.get() and landmarks_display_clean is not None:
                    crop_writer.write(cv2.cvtColor(landmarks_display_clean, cv2.COLOR_RGB2BGR))
            
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

# Create and run GUI
root = tk.Tk()
app = FaceDetectionGUI(root)
root.mainloop()