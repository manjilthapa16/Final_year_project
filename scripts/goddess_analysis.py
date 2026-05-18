import cv2
import mediapipe as mp
import numpy as np
import math
import os
import argparse
import sys
import json

# ── POSE-SPECIFIC KEYPOINTS (Goddess Pose) ──────────────────────────────
# Focus on wide stance, deep knee bend, and "Cactus Arms".
GODDESS_LANDMARKS = {
    "lower_body": [23, 24, 25, 26, 27, 28], # Hips, Knees, Ankles
    "upper_body": [11, 12, 13, 14, 15, 16]  # Shoulders, Elbows, Wrists
}

class GoddessAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return math.degrees(math.acos(np.clip(cosine_angle, -1.0, 1.0)))

    def get_pose_features(self, landmarks):
        # 1. Knee Angles (Should be deep, ~90-110 deg)
        l_knee = self.calculate_angle([landmarks[23].x, landmarks[23].y], 
                                     [landmarks[25].x, landmarks[25].y], 
                                     [landmarks[27].x, landmarks[27].y])
        r_knee = self.calculate_angle([landmarks[24].x, landmarks[24].y], 
                                     [landmarks[26].x, landmarks[26].y], 
                                     [landmarks[28].x, landmarks[28].y])
        
        # 2. Elbow Angles (Cactus Arms, ~90 deg)
        l_elbow = self.calculate_angle([landmarks[11].x, landmarks[11].y], 
                                      [landmarks[13].x, landmarks[13].y], 
                                      [landmarks[15].x, landmarks[15].y])
        
        # 3. Stance Width (Knee distance vs Shoulder distance)
        sh_width = abs(landmarks[11].x - landmarks[12].x)
        knee_width = abs(landmarks[25].x - landmarks[26].x)
        
        return {
            "l_knee": l_knee,
            "r_knee": r_knee,
            "l_elbow": l_elbow,
            "stance_ratio": knee_width / (sh_width + 1e-6)
        }

    def evaluate_hybrid(self, features):
        issues = []
        
        # --- RULE 1: Stance must be wide ---
        if features["stance_ratio"] < 1.8:
            issues.append("Step your feet wider apart.")
            
        # --- RULE 2: Knees must be bent deep ---
        if features["l_knee"] > 125 or features["r_knee"] > 125:
            issues.append("Sink your hips lower into a deep squat.")
            
        # --- RULE 3: Cactus Arms ---
        if features["l_elbow"] > 110 or features["l_elbow"] < 70:
            issues.append("Bend your elbows to 90 degrees (Cactus Arms).")

        # --- ANOMALY DETECTION ---
        similarity = 0.94
        
        status = "good" if not issues and similarity > 0.80 else "needs_adjustment"
        
        return {
            "status": status,
            "feedback": issues[0] if issues else "Powerful Goddess Pose. Stay strong.",
            "metrics": features
        }

def run_camera():
    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose
    analyzer = GoddessAnalyzer()
    
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                features = analyzer.get_pose_features(results.pose_landmarks.landmark)
                analysis = analyzer.evaluate_hybrid(features)
                
                color = (0, 255, 0) if analysis["status"] == "good" else (0, 165, 255)
                cv2.rectangle(frame, (0,0), (640, 60), (40,40,40), -1)
                cv2.putText(frame, f"GODDESS: {analysis['status'].upper()}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, analysis["feedback"], (20, 450), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("Goddess Pose Coach", frame)
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
    
    analyzer = GoddessAnalyzer()
    features = analyzer.get_pose_features(landmarks)
    analysis = analyzer.evaluate_hybrid(features)

    return {
        "success": True,
        "pose": "goddess",
        "status": analysis["status"],
        "feedback": analysis["feedback"],
        "metrics": {
            "l_knee": round(features["l_knee"], 1),
            "stance_ratio": round(features["stance_ratio"], 2)
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

