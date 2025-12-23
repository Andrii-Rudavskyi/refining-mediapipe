import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import tensorflow as tf
import numpy as np

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

# Start camera
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    # Process the frame
    results = detector.detect(mp_image)
    
    # Draw bounding boxes
    if results.detections:
        for detection in results.detections:
            # Get bounding box
            bbox = detection.bounding_box
            h, w, _ = frame.shape
            
            # Convert to pixel coordinates
            x = bbox.origin_x
            y = bbox.origin_y
            width = bbox.width
            height = bbox.height
            
            # Calculate center for scaling
            center_x = x + width / 2
            center_y = y + height / 2
            
            # Calculate scaled box (1.35x)
            scale_factor = 1.35
            scaled_width = width * scale_factor
            scaled_height = height * scale_factor
            scaled_x = int(center_x - scaled_width / 2)
            scaled_y = int(center_y - scaled_height / 2)
            scaled_x2 = int(center_x + scaled_width / 2)
            scaled_y2 = int(center_y + scaled_height / 2)
            
            # Clamp coordinates to frame boundaries
            scaled_x = max(0, scaled_x)
            scaled_y = max(0, scaled_y)
            scaled_x2 = min(w, scaled_x2)
            scaled_y2 = min(h, scaled_y2)
            
            # Crop the blue box region
            cropped = frame[scaled_y:scaled_y2, scaled_x:scaled_x2]
            
            # Process cropped face with landmarks detector
            if cropped.size > 0:
                # Prepare input for landmarks model
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                landmarks_input_resized = cv2.resize(cropped_rgb, (landmarks_w, landmarks_h))
                
                # Create display copy before normalization
                landmarks_display = landmarks_input_resized.copy()
                
                # Normalize for model input
                landmarks_input = (landmarks_input_resized.astype(np.float32) / 127.5) - 1.0
                landmarks_input = np.expand_dims(landmarks_input, axis=0)
                
                # Run landmarks inference
                landmarks_interpreter.set_tensor(landmarks_input_details[0]['index'], landmarks_input)
                landmarks_interpreter.invoke()
                
                # Get landmarks output
                landmarks_raw = landmarks_interpreter.get_tensor(landmarks_output_details[0]['index'])[0]
                
                # Reshape from [1, 1, 1434] to [478, 3] (478 landmarks with x, y, z)
                landmarks = landmarks_raw.reshape(-1, 3)
                
                # Draw landmarks on the 256x256 resized image
                # Landmarks are already in pixel coordinates for 256x256 image
                num_landmarks = landmarks.shape[0]
                for i in range(num_landmarks):
                    x_pixel = landmarks[i, 0]
                    y_pixel = landmarks[i, 1]
                    
                    # Clamp to image bounds
                    x_pixel = max(0, min(landmarks_w - 1, x_pixel))
                    y_pixel = max(0, min(landmarks_h - 1, y_pixel))
                    
                    # Draw landmark point with subpixel accuracy (shift by 4 bits for fixed-point)
                    cv2.circle(landmarks_display, 
                              (int(x_pixel * 16), int(y_pixel * 16)), 
                              32,  # radius * 16 for subpixel
                              (0, 255, 255), 
                              -1, 
                              cv2.LINE_AA,  # Anti-aliased line
                              shift=4)  # Shift for fixed-point coordinates
                
                cv2.imshow('Landmarks on 256x256 Input', landmarks_display)
            
            # Draw scaled blue rectangle
            cv2.rectangle(frame, (scaled_x, scaled_y), (scaled_x2, scaled_y2), (255, 0, 0), 2)
            
            # Draw original green rectangle
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            # Draw confidence score
            confidence = detection.categories[0].score
            cv2.putText(frame, f"{confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow('MediaPipe Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()