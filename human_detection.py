#!/usr/bin/env python3
"""
Simple camera-based object detection and tracking using YOLOv8n + BotSort
Uses camera 0 (default webcam)
"""

import cv2
from ultralytics import YOLO

def main():
    # Initialize YOLOv8n model
    print("Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    
    # Open camera 0
    print("Opening camera 0...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera 0")
        return
    
    # Set camera properties (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Starting detection and tracking...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera")
            break
        
        # Run YOLO detection with BotSort tracking
        results = model.track(
            frame,
            tracker="botsort.yaml",  # Use BotSort tracker
            persist=True,  # Persist tracks across frames
            verbose=False  # Don't print to console
        )
        
        # Draw results on frame
        annotated_frame = results[0].plot()
        
        # Display FPS (approximate)
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, "YOLOv8n + BotSort", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('YOLOv8n + BotSort Tracking', annotated_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    main()

