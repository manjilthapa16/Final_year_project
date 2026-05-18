import cv2
import mediapipe as mp
import numpy as np
import math
import json
import os
import argparse
import sys

# ── POSE-SPECIFIC KEYPOINTS (Plank) ───────────────────────────────────────
# Plank is a full-body alignment pose. We care about the straight line
# from shoulders to hips to ankles.
PLANK_LANDMARKS = {
    "alignment": [11, 12, 23, 24, 27, 28], # Shoulders, Hips, Ankles
    "support": [13, 14, 15, 16]            # Elbows/Wrists
}

class PlankAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return math.degrees(math.acos(np.clip(cosine_angle, -1.0, 1.0)))

    def get_pose_features(self, landmarks):
        # We look at the side-profile view (avg of left and right)
        avg_sh_y = (landmarks[11].y + landmarks[12].y) / 2.0
        avg_hp_y = (landmarks[23].y + landmarks[24].y) / 2.0
        avg_an_y = (landmarks[27].y + landmarks[28].y) / 2.0
        
        # Hip relative to the shoulder-ankle line
        # 0.0 means hips are perfectly in line. 
        # Positive = hips lower than line (sagging). Negative = hips higher (piking).
        line_mid_y = (avg_sh_y + avg_an_y) / 2.0
        hip_deviation = avg_hp_y - line_mid_y

        return {
            "hip_deviation": hip_deviation,
            "sh_y": avg_sh_y,
            "hp_y": avg_hp_y,
            "an_y": avg_an_y
        }

    def evaluate_hybrid(self, features):
        issues = []
        
        # --- RULE 1: Detect Sagging Hips ---
        if features["hip_deviation"] > 0.06:
            issues.append("Hips are sagging. Squeeze your glutes and core.")
            
        # --- RULE 2: Detect Piked Hips ---
        if features["hip_deviation"] < -0.06:
            issues.append("Hips are too high. Lower them to form a straight line.")

        # --- ANOMALY DETECTION (Placeholder) ---
        similarity = 0.95 
        
        status = "good" if not issues and similarity > 0.80 else "needs_adjustment"
        
        return {
            "status": status,
            "feedback": issues[0] if issues else "Perfect plank alignment. Keep it up!",
            "metrics": features
        }

def run_camera():
    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose
    analyzer = PlankAnalyzer()
    
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                features = analyzer.get_pose_features(results.pose_landmarks.landmark)
                analysis = analyzer.evaluate_hybrid(features)
                
                color = (0, 255, 0) if analysis["status"] == "good" else (0, 165, 255)
                cv2.rectangle(frame, (0,0), (640, 60), (40,40,40), -1)
                cv2.putText(frame, f"PLANK: {analysis['status'].upper()}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, analysis["feedback"], (20, 450), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("Plank Coach", frame)
            if cv2.waitKey(5) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

class MockLandmark:
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

def analyze_landmarks(landmarks_file, visibility_threshold=0.3):
    try:
        with open(landmarks_file, 'r') as f:
            landmarks_data = json.load(f)
    except Exception:
        return {"success": False, "error": "Unable to read landmarks JSON"}

    if not landmarks_data:
        return {"success": False, "error": "No person detected"}

    landmarks = [MockLandmark(lm.get('x', 0.0), lm.get('y', 0.0), lm.get('z', 0.0), lm.get('visibility', 1.0)) for lm in landmarks_data]
    
    analyzer = PlankAnalyzer()
    features = analyzer.get_pose_features(landmarks)
    analysis = analyzer.evaluate_hybrid(features)

    return {
        "success": True,
        "pose": "plank",
        "status": analysis["status"],
        "feedback": analysis["feedback"],
        "metrics": {
            "hip_deviation": round(features["hip_deviation"], 3),
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks", required=False)
    parser.add_argument("--camera", action="store_true")
    args = parser.parse_args()

    if args.camera or not args.landmarks:
        run_camera()
    else:
        result = analyze_landmarks(args.landmarks)
        sys.stdout.write(json.dumps(result))

if __name__ == "__main__":
    main()

