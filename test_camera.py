#!/usr/bin/env python3
"""
RealSense camera test — color and depth. Press 'q' to quit.
"""

import pyrealsense2 as rs
import numpy as np
import cv2

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipe.start(cfg)

try:
    while True:
        frame = pipe.wait_for_frames()
        depth_frame = frame.get_depth_frame()
        color_frame = frame.get_color_frame()
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # --- FIX STARTS HERE ---
        # Normalize depth data to 8-bit for visualization
        # Alpha is a scaling factor; adjust based on your camera's distance
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # Show the images
        cv2.imshow('rgb', color_image)
        cv2.imshow('depth', depth_colormap)
        # --- FIX ENDS HERE ---

        if cv2.waitKey(1) == ord('q'):
            break
finally:
    pipe.stop()
    cv2.destroyAllWindows()
    
