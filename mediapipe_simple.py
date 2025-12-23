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

# Tracking state variables
tracked_bbox = None  # Stores the predicted bounding box
tracking_active = False
frame_count = 0
detect_interval = 1  # Detect face every N frames when tracking is lost

# Start camera
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame_count += 1
    h, w, _ = frame.shape
    
    # Decide whether to run face detection
    run_detection = False
    if not tracking_active:
        # No tracking active - run detection on first frame or at intervals
        if frame_count == 1 or frame_count % detect_interval == 0:
            run_detection = True
    
    detected_bbox = None
    
    # Run face detection if needed
    if run_detection:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Process the frame
        results = detector.detect(mp_image)
        
        # Get bounding box from detection
        if results.detections:
            detection = results.detections[0]  # Use first detection
            bbox = detection.bounding_box
            
            # Convert to pixel coordinates
            x = bbox.origin_x
            y = bbox.origin_y
            width = bbox.width
            height = bbox.height
            
            detected_bbox = {
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'confidence': detection.categories[0].score
            }
            tracked_bbox = detected_bbox
            tracking_active = True
            cv2.putText(frame, "DETECTING", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # No face detected
            tracking_active = False
            tracked_bbox = None
    else:
        # Use tracked bounding box (predicted from previous frame)
        if tracked_bbox is not None:
            cv2.putText(frame, "TRACKING", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Process landmarks if we have a bounding box (detected or tracked)
    if tracked_bbox is not None:
        x = tracked_bbox['x']
        y = tracked_bbox['y']
        width = tracked_bbox['width']
        height = tracked_bbox['height']
        
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
            
            # Get landmarks confidence score (usually output index 1)
            landmarks_score = None
            if len(landmarks_output_details) > 1:
                landmarks_score_raw = landmarks_interpreter.get_tensor(landmarks_output_details[1]['index'])
                landmarks_score = float(landmarks_score_raw.flatten()[0])
            
            # Reshape from [1, 1, 1434] to [478, 3] (478 landmarks with x, y, z)
            landmarks = landmarks_raw.reshape(-1, 3)
            
            # Check if tracking is valid based on landmarks score
            if landmarks_score is not None and landmarks_score < 0:
                # Tracking lost - score below threshold
                tracking_active = False
                tracked_bbox = None
                print(f"Tracking lost: score={landmarks_score:.3f}")
            else:
                # Predict next bounding box based on landmarks
                # Calculate bbox from landmarks for tracking
                landmarks_x = landmarks[:, 0]
                landmarks_y = landmarks[:, 1]
                
                # Find bounding box of landmarks (in 256x256 space)
                min_x = np.min(landmarks_x)
                max_x = np.max(landmarks_x)
                min_y = np.min(landmarks_y)
                max_y = np.max(landmarks_y)
                
                # Transform back to original frame coordinates
                scale_x = (scaled_x2 - scaled_x) / landmarks_w
                scale_y = (scaled_y2 - scaled_y) / landmarks_h
                
                predicted_x = int(scaled_x + min_x * scale_x)
                predicted_y = int(scaled_y + min_y * scale_y)
                predicted_width = int((max_x - min_x) * scale_x)
                predicted_height = int((max_y - min_y) * scale_y)
                
                # Make bounding box square (use max dimension)
                predicted_size = max(predicted_width, predicted_height)
                
                # Center the square box
                predicted_center_x = predicted_x + predicted_width / 2
                predicted_center_y = predicted_y + predicted_height / 2
                predicted_x = int(predicted_center_x - predicted_size / 2)
                predicted_y = int(predicted_center_y - predicted_size / 2)
                predicted_width = predicted_size
                predicted_height = predicted_size
                
                # Add padding to predicted bbox (10%)
                padding = 0.1
                predicted_x = int(predicted_x - predicted_width * padding)
                predicted_y = int(predicted_y - predicted_height * padding)
                predicted_width = int(predicted_width * (1 + 2 * padding))
                predicted_height = int(predicted_height * (1 + 2 * padding))
                
                # Clamp to frame boundaries
                predicted_x = max(0, predicted_x)
                predicted_y = max(0, predicted_y)
                predicted_width = min(w - predicted_x, predicted_width)
                predicted_height = min(h - predicted_y, predicted_height)
                
                # Update tracked bbox with predicted bbox for next frame
                tracked_bbox = {
                    'x': predicted_x,
                    'y': predicted_y,
                    'width': predicted_width,
                    'height': predicted_height,
                    'confidence': landmarks_score if landmarks_score is not None else 0.5
                }
                
                if landmarks_score is not None:
                    cv2.putText(frame, f"Score: {landmarks_score:.3f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
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
                
                # Draw scaled blue rectangle (actual processing area)
                cv2.rectangle(frame, (scaled_x, scaled_y), (scaled_x2, scaled_y2), (255, 0, 0), 2)
                
                # Draw original green rectangle (bounding box)
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                
                # Draw confidence score
                confidence = tracked_bbox.get('confidence', 0.0)
                cv2.putText(frame, f"{confidence:.2f}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow('MediaPipe Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()