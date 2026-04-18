#!/usr/bin/env python3

# Live bottle detector using OWLv2
# Adapted from Boxer project for real-time webcam detection

import cv2
import torch
import numpy as np
import time
from owl.owl_wrapper import OwlWrapper

def main():
    # Initialize OWL detector for bottles
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    owl = OwlWrapper(
        device=device,
        text_prompts=["bottle"],  # Detect bottles
        min_confidence=0.1,  # Lower threshold for demo
    )

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set camera resolution (lower = faster)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting live bottle detection. Press 'q' to quit.")
    print("Note: First run will be slow while models warm up.\n")

    frame_count = 0
    fps_times = []
    frame_skip = 2
    detections = None

    while True:
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detection every N frames
        if frame_count % frame_skip == 0:
            t0 = time.time()
            # OWL expects [0, 255] float range, NOT [0, 1]
            img_tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float()[None]
            # Returns: boxes [N,4], scores [N], labels [N], None
            boxes, scores, labels, _ = owl.forward(img_tensor)
            detections = (boxes, scores, labels)
            elapsed = time.time() - t0
            fps_times.append(elapsed)
            if len(fps_times) > 10:
                fps_times.pop(0)
        
        # Draw detections
        if detections is not None:
            boxes, scores, labels = detections
            for i in range(len(boxes)):
                if scores[i] > 0.1:
                    x1, x2, y1, y2 = boxes[i].numpy()  # [x1, x2, y1, y2]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"bottle: {scores[i]:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1)-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show FPS
        if fps_times:
            avg_time = np.mean(fps_times)
            fps = 1.0 / avg_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Live Bottle Detection', frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()