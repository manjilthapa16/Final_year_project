import cv2
import mediapipe as mp
import numpy as np
import math
import json
import os
import argparse
import sys

# ── POSE-SPECIFIC KEYPOINTS (Tree Pose) ───────────────────────────────────
# We focus on the standing leg, the lifted leg, and the torso verticality.
# Hand positions are calculated but ignored for the 'Good/Bad' status check
# to accommodate variations (chest vs overhead).
TREE_LANDMARKS = {
    "standing_leg": [24, 26, 28], # Hip, Knee, Ankle (Right) or [23, 25, 27] (Left)
    "lifted_leg": [23, 25, 27],   # The opposite leg
    "torso": [11, 12, 23, 24]
}

class TreeAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return math.degrees(math.acos(np.clip(cosine_angle, -1.0, 1.0)))

    def get_pose_features(self, landmarks):
        # 1. Identify which leg is standing (the straighter one)
        l_knee_angle = self.calculate_angle([landmarks[23].x, landmarks[23].y], 
                                          [landmarks[25].x, landmarks[25].y], 
                                          [landmarks[27].x, landmarks[27].y])
        r_knee_angle = self.calculate_angle([landmarks[24].x, landmarks[24].y], 
                                          [landmarks[26].x, landmarks[26].y], 
                                          [landmarks[28].x, landmarks[28].y])
        
        standing_side = "right" if r_knee_angle > l_knee_angle else "left"
        
        if standing_side == "right":
            standing_knee = r_knee_angle
            lifted_knee = l_knee_angle
            # Foot placement: lifted ankle relative to standing knee
            foot_pos_y = landmarks[27].y 
            knee_pos_y = landmarks[26].y
        else:
            standing_knee = l_knee_angle
            lifted_knee = r_knee_angle
            foot_pos_y = landmarks[28].y
            knee_pos_y = landmarks[25].y

        # Torso verticality
        l_sh, l_hp = landmarks[11], landmarks[23]
        torso_lean = abs(l_sh.x - l_hp.x)

        return {
            "standing_knee": standing_knee,
            "lifted_knee": lifted_knee,
            "torso_lean": torso_lean,
            "foot_pos_y": foot_pos_y,
            "knee_pos_y": knee_pos_y,
            "side": standing_side
        }

    def evaluate_hybrid(self, features):
        issues = []
        
        # --- RULE 1: Standing leg must be straight ---
        if features["standing_knee"] < 165:
            issues.append("Straighten your standing leg.")
            
        # --- RULE 2: Lifted leg must be bent ---
        if features["lifted_knee"] > 140:
            issues.append("Lift your foot and place it on your inner thigh.")
            
        # --- RULE 3: Foot placement safety (Don't press on the knee) ---
        # If ankle is too close to knee Y-coordinate
        if abs(features["foot_pos_y"] - features["knee_pos_y"]) < 0.05:
            issues.append("Avoid placing your foot directly on the knee joint.")
            
        # --- RULE 4: Torso verticality ---
        if features["torso_lean"] > 0.10:
            issues.append("Center your weight. Avoid leaning to the side.")

        # --- ANOMALY DETECTION (Placeholder) ---
        similarity = 0.98 # High similarity to 'Good Pose' average
        
        status = "good" if not issues and similarity > 0.85 else "needs_adjustment"
        
        return {
            "status": status,
            "feedback": issues[0] if issues else "Stable Tree Pose. Breathe and focus.",
            "metrics": features
        }

def run_camera():
    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose
    analyzer = TreeAnalyzer()
    
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                features = analyzer.get_pose_features(results.pose_landmarks.landmark)
                analysis = analyzer.evaluate_hybrid(features)
                
                # UI
                color = (0, 255, 0) if analysis["status"] == "good" else (0, 165, 255)
                cv2.rectangle(frame, (0,0), (640, 60), (40,40,40), -1)
                cv2.putText(frame, f"TREE POSE: {analysis['status'].upper()}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, analysis["feedback"], (20, 450), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("Tree Pose Coach", frame)
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

    # Convert dictionary list to mock objects for analyzer
    landmarks = [MockLandmark(lm.get('x', 0.0), lm.get('y', 0.0), lm.get('z', 0.0), lm.get('visibility', 1.0)) for lm in landmarks_data]
    
    analyzer = TreeAnalyzer()
    features = analyzer.get_pose_features(landmarks)
    analysis = analyzer.evaluate_hybrid(features)

    return {
        "success": True,
        "pose": "tree",
        "status": analysis["status"],
        "feedback": analysis["feedback"],
        "metrics": {
            "standing_knee": round(features["standing_knee"], 1),
            "lifted_knee": round(features["lifted_knee"], 1),
            "torso_lean": round(features["torso_lean"], 3),
            "side": features["side"]
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks", required=False, help="Path to landmarks JSON")
    parser.add_argument("--camera", action="store_true", help="Run in camera mode")
    args = parser.parse_args()

    if args.camera or not args.landmarks:
        run_camera()
    else:
        result = analyze_landmarks(args.landmarks)
        sys.stdout.write(json.dumps(result))

if __name__ == "__main__":
    main()

