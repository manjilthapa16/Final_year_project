import cv2
import mediapipe as mp
import numpy as np
import math
import os
import argparse
import sys
import json

# ── POSE-SPECIFIC KEYPOINTS (Down Dog) ──────────────────────────────────
# The primary metric is the "Inverted V" shape.
# We focus on the Hip-Shoulder-Ankle alignment and Arm straightness.
DOWNDOG_LANDMARKS = {
    "v_shape": [11, 12, 23, 24, 27, 28], # Shoulders, Hips, Ankles
    "arms": [13, 14, 15, 16]             # Elbows, Wrists
}

class DownDogAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return math.degrees(math.acos(np.clip(cosine_angle, -1.0, 1.0)))

    def get_pose_features(self, landmarks):
        # 1. Hip Angle (The peak of the V)
        l_hp_angle = self.calculate_angle([landmarks[11].x, landmarks[11].y], 
                                        [landmarks[23].x, landmarks[23].y], 
                                        [landmarks[27].x, landmarks[27].y])
        
        # 2. Arm Straightness
        l_arm_angle = self.calculate_angle([landmarks[11].x, landmarks[11].y], 
                                         [landmarks[13].x, landmarks[13].y], 
                                         [landmarks[15].x, landmarks[15].y])
        
        # 3. Hip Height relative to shoulders/ankles
        avg_sh_y = (landmarks[11].y + landmarks[12].y) / 2.0
        avg_hp_y = (landmarks[23].y + landmarks[24].y) / 2.0
        
        return {
            "hip_angle": l_hp_angle,
            "arm_angle": l_arm_angle,
            "hp_y": avg_hp_y,
            "sh_y": avg_sh_y
        }

    def evaluate_hybrid(self, features):
        issues = []
        
        # --- RULE 1: Hip Angle must be sharp (Inverted V) ---
        if features["hip_angle"] > 100:
            issues.append("Lift your hips higher to form an inverted V.")
            
        # --- RULE 2: Arms must be straight ---
        if features["arm_angle"] < 160:
            issues.append("Straighten your arms and press through your palms.")
            
        # --- RULE 3: Hips must be the highest point ---
        if features["hp_y"] > features["sh_y"]:
            issues.append("Send your hips back and up.")

        # --- ANOMALY DETECTION (Placeholder) ---
        similarity = 0.96
        
        status = "good" if not issues and similarity > 0.85 else "needs_adjustment"
        
        return {
            "status": status,
            "feedback": issues[0] if issues else "Strong Downward Dog. Lengthen your back.",
            "metrics": features
        }

def run_camera():
    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose
    analyzer = DownDogAnalyzer()
    
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
                cv2.putText(frame, f"DOWN DOG: {analysis['status'].upper()}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, analysis["feedback"], (20, 450), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("Down Dog Coach", frame)
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
    
    analyzer = DownDogAnalyzer()
    features = analyzer.get_pose_features(landmarks)
    analysis = analyzer.evaluate_hybrid(features)

    return {
        "success": True,
        "pose": "downdog",
        "status": analysis["status"],
        "feedback": analysis["feedback"],
        "metrics": {
            "hip_angle": round(features["hip_angle"], 1),
            "arm_angle": round(features["arm_angle"], 1)
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

