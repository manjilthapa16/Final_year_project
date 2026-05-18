import cv2
import mediapipe as mp
import numpy as np
import math
import joblib
import os
import argparse
import sys
import json

try:
    from scripts.pose_features import (
        LANDMARKS,
        FEATURE_NAMES,
        compute_engineered_features,
        has_required_visibility,
        feature_vector,
    )
except ModuleNotFoundError:
    from pose_features import (  # type: ignore
        LANDMARKS,
        FEATURE_NAMES,
        compute_engineered_features,
        has_required_visibility,
        feature_vector,
    )

# ── POSE-SPECIFIC KEYPOINTS (Squat) ──────────────────────────────────────
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

class SquatAnalyzer:
    def __init__(self, model_path=None, le_path=None):
        self.mp_pose = mp.solutions.pose
        self.clf = None
        self.le = None
        
        if model_path and os.path.exists(model_path):
            try:
                self.clf = joblib.load(model_path)
                if isinstance(self.clf, dict): self.clf = self.clf.get("model")
            except: pass
            
        if le_path and os.path.exists(le_path):
            try:
                self.le = joblib.load(le_path)
            except: pass

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return math.degrees(math.acos(np.clip(cosine_angle, -1.0, 1.0)))

    def get_pose_features(self, landmarks):
        # 1. Knee Angles
        l_knee = self.calculate_angle([landmarks[23].x, landmarks[23].y], 
                                     [landmarks[25].x, landmarks[25].y], 
                                     [landmarks[27].x, landmarks[27].y])
        r_knee = self.calculate_angle([landmarks[24].x, landmarks[24].y], 
                                     [landmarks[26].x, landmarks[26].y], 
                                     [landmarks[28].x, landmarks[28].y])
        
        # 2. Hip Angles
        l_hip = self.calculate_angle([landmarks[11].x, landmarks[11].y],
                                    [landmarks[23].x, landmarks[23].y],
                                    [landmarks[25].x, landmarks[25].y])
        r_hip = self.calculate_angle([landmarks[12].x, landmarks[12].y],
                                    [landmarks[24].x, landmarks[24].y],
                                    [landmarks[26].x, landmarks[26].y])

        # 3. Torso Lean
        l_sh, l_hp = landmarks[11], landmarks[23]
        torso_lean = abs(l_sh.x - l_hp.x)
        
        # 4. Knee forward
        knee_forward = abs(landmarks[25].x - landmarks[27].x)

        return {
            "knee_angle": (l_knee + r_knee) / 2.0,
            "hip_angle": (l_hip + r_hip) / 2.0,
            "torso_lean": torso_lean,
            "knee_forward": knee_forward
        }

    def evaluate_hybrid(self, metrics, landmarks_raw):
        issues = []
        
        knee_angle = metrics["knee_angle"]
        torso_lean = metrics["torso_lean"]
        hip_angle = metrics["hip_angle"]
        knee_forward = metrics["knee_forward"]
        
        phase = "standing"
        if knee_angle < 150: phase = "descending"
        if knee_angle < 110: phase = "bottom"

        # --- RULE 1: Torso Safety ---
        is_folding = (hip_angle < 62) or (torso_lean > 0.20)
        if is_folding:
            issues.append("Keep your chest up. Avoid folding forward.")
            
        # --- RULE 2: Depth check ---
        if phase == "bottom" and knee_angle > 115:
            issues.append("Sink slightly deeper into the squat.")

        # --- RULE 3: Heels Rising ---
        if phase in ["descending", "bottom"] and knee_forward > 0.35:
            issues.append("Keep your heels grounded. Don't let your knees shift too far forward.")

        # --- ML MODEL OVERRIDE ---
        ml_source = "None"
        if self.clf and not issues:
            try:
                feats_dict, _ = compute_engineered_features(landmarks_raw)
                X = np.array([feature_vector(feats_dict)], dtype=np.float32)
                pred_idx = self.clf.predict(X)[0]
                proba = float(np.max(self.clf.predict_proba(X)[0]))
                
                pred_label = self.le.inverse_transform([pred_idx])[0] if self.le else str(pred_idx)
                
                if pred_label == "squat_bad_back" and proba > 0.45:
                    issues.append("Your back looks rounded. Lengthen your spine.")
                    ml_source = f"ML ({pred_label})"
                elif pred_label == "squat_bad_heel" and proba > 0.45:
                    issues.append("Check your feet. Ensure your heels aren't lifting.")
                    ml_source = f"ML ({pred_label})"
            except: pass
        
        status = "good" if not issues else "needs_adjustment"
        return {
            "status": status,
            "feedback": issues[0] if issues else "Solid squat form. Keep it controlled.",
            "phase": phase,
            "source": ml_source if ml_source != "None" else "Rule Engine"
        }

class MockLandmark:
    def __init__(self, x, y, z, visibility):
        self.x = x; self.y = y; self.z = z; self.visibility = visibility

def analyze_landmarks(landmarks_file):
    try:
        with open(landmarks_file, 'r') as f: landmarks_data = json.load(f)
    except: return {"success": False, "error": "Unable to read landmarks JSON"}
    if not landmarks_data: return {"success": False, "error": "No person detected"}
    landmarks = [MockLandmark(lm.get('x', 0.0), lm.get('y', 0.0), lm.get('z', 0.0), lm.get('visibility', 1.0)) for lm in landmarks_data]
    
    project_root = "/home/anurag/Desktop/your_gym_copy"
    model_path = os.path.join(project_root, "models", "pose_classifier.pkl")
    le_path = os.path.join(project_root, "models", "label_encoder.pkl")
    analyzer = SquatAnalyzer(model_path, le_path)
    features = analyzer.get_pose_features(landmarks)
    analysis = analyzer.evaluate_hybrid(features, landmarks)
    return {
        "success": True, "pose": "squat", "status": analysis["status"],
        "feedback": analysis["feedback"], "phase": analysis["phase"],
        "source": analysis["source"],
        "metrics": {"knee_angle": round(features["knee_angle"], 1), "torso_lean": round(features["torso_lean"], 3)}
    }

def analyze_image(image_file):
    import cv2; import mediapipe as mp
    image = cv2.imread(image_file)
    if image is None: return {"success": False, "error": "Unable to read image file"}
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)
    if not results.pose_landmarks: return {"success": False, "error": "No person detected"}
    landmarks = [MockLandmark(lm.x, lm.y, lm.z, lm.visibility) for lm in results.pose_landmarks.landmark]
    
    project_root = "/home/anurag/Desktop/your_gym_copy"
    model_path = os.path.join(project_root, "models", "pose_classifier.pkl")
    le_path = os.path.join(project_root, "models", "label_encoder.pkl")
    analyzer = SquatAnalyzer(model_path, le_path)
    features = analyzer.get_pose_features(landmarks)
    analysis = analyzer.evaluate_hybrid(features, landmarks)
    return {
        "success": True, "pose": "squat", "status": analysis["status"],
        "feedback": analysis["feedback"], "phase": analysis["phase"],
        "source": analysis["source"],
        "metrics": {"knee_angle": round(features["knee_angle"], 1), "hip_angle": round(features["hip_angle"], 1), "torso_lean": round(features["torso_lean"], 3)}
    }

def run_camera():
    import cv2; import mediapipe as mp
    project_root = "/home/anurag/Desktop/your_gym_copy"
    model_path = os.path.join(project_root, "models", "pose_classifier.pkl")
    le_path = os.path.join(project_root, "models", "label_encoder.pkl")
    cap = cv2.VideoCapture(0); mp_pose = mp.solutions.pose; analyzer = SquatAnalyzer(model_path, le_path)
    with mp_pose.Pose(min_detection_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1); results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                features = analyzer.get_pose_features(results.pose_landmarks.landmark)
                analysis = analyzer.evaluate_hybrid(features, results.pose_landmarks.landmark)
                color = (0, 255, 0) if analysis["status"] == "good" else (0, 165, 255)
                cv2.rectangle(frame, (0,0), (640, 60), (40,40,40), -1)
                cv2.putText(frame, f"SQUAT: {analysis['status'].upper()} ({analysis['phase']})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, analysis["feedback"], (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow("Squat Coach", frame)
            if cv2.waitKey(5) & 0xFF == ord('q'): break
    cap.release(); cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks"); parser.add_argument("--image"); parser.add_argument("--camera", action="store_true")
    # Compat args
    parser.add_argument("--pose"); parser.add_argument("--previous_feedback"); parser.add_argument("--previous_metrics")
    args = parser.parse_args()
    if args.image: result = analyze_image(args.image)
    elif args.camera or not args.landmarks: run_camera(); return
    else: result = analyze_landmarks(args.landmarks)
    sys.stdout.write(json.dumps(result))

if __name__ == "__main__":
    main()
