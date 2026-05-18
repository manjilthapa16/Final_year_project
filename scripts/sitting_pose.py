import argparse
import json
import os
import random
import sys
import math

import numpy as np

try:
    from scripts.pose_features import (
        LANDMARKS,
        compute_engineered_features,
        has_required_visibility,
        landmarks_to_array,
    )
except ModuleNotFoundError:
    from pose_features import (  # type: ignore
        LANDMARKS,
        compute_engineered_features,
        has_required_visibility,
        landmarks_to_array,
    )

# Upper body landmarks
NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

SITTING_CORE = (LEFT_SHOULDER, RIGHT_SHOULDER) # Minimum for any analysis

# ── Model path ───────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "sitting_posture_clf.pkl")

# ── Per-class feedback ───────────────────────────────────────────────────────
SITTING_FEEDBACK = {
    "goodpose": {
        "status": "good",
        "issue": "sitting_stable",
        "severity": 0.0,
        "feedback": "Excellent sitting form. Spine neutral and shoulders relaxed.",
    },
    "titled_head": {
        "status": "needs_adjustment",
        "issue": "sitting_tilted_head",
        "severity": 0.7,
        "feedback": "Your head is tilted to one side. Straighten your neck and look straight ahead.",
    },
    "round_head": {
        "status": "needs_adjustment",
        "issue": "sitting_round_head",
        "severity": 0.75,
        "feedback": "Your head is drooping forward. Lift your chin and bring your ears directly over your shoulders.",
    },
    "round_back": {
        "status": "needs_adjustment",
        "issue": "sitting_round_back",
        "severity": 0.8,
        "feedback": "Your back is rounded. Sit tall — pull your shoulder blades back and straighten your spine.",
    },
}

def _distance(p1, p2):
    return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

def _names_from_indices(indices):
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    names = [mp_pose.PoseLandmark(i).name for i in indices]
    return [name.replace("LEFT_", "").replace("RIGHT_", "") for name in names]

def _compute_sitting_posture_metrics(landmarks, engineered_metrics=None):
    l_sh = landmarks[LEFT_SHOULDER]
    r_sh = landmarks[RIGHT_SHOULDER]
    l_ear = landmarks[LEFT_EAR]
    r_ear = landmarks[RIGHT_EAR]

    shoulder_width = _distance(l_sh, r_sh)
    ear_width = _distance(l_ear, r_ear)
    avg_ear_y = (l_ear['y'] + r_ear['y']) / 2.0
    avg_sh_y = (l_sh['y'] + r_sh['y']) / 2.0

    metrics = {
        "neck_tilt": round(abs(l_ear['y'] - r_ear['y']) / (ear_width + 1e-6), 3),
        "shoulder_tilt": round(abs(l_sh['y'] - r_sh['y']) / (shoulder_width + 1e-6), 3),
        "spine_extension": round((avg_sh_y - avg_ear_y) / (shoulder_width + 1e-6), 3),
        "side": engineered_metrics["side"] if engineered_metrics else "front",
    }

    if engineered_metrics:
        metrics["torso_lean"] = round(engineered_metrics["torso_lean"], 3)
        metrics["hip_angle"] = round(engineered_metrics["hip_angle"], 1)
    else:
        metrics["torso_lean"] = None

    return metrics

def analyze_sitting_posture(landmarks, metrics=None):
    """
    Analyzes sitting posture using shoulders, ears, and hips if available.
    """
    l_sh = landmarks[LEFT_SHOULDER]
    r_sh = landmarks[RIGHT_SHOULDER]
    l_ear = landmarks[LEFT_EAR]
    r_ear = landmarks[RIGHT_EAR]

    # Calculate scale-independent distances for normalization
    shoulder_width = _distance(l_sh, r_sh)
    ear_width = _distance(l_ear, r_ear)

    # 1. Check Head Tilt (Ear Levelness)
    head_tilt = abs(l_ear['y'] - r_ear['y'])
    head_tilt_ratio = head_tilt / (ear_width + 1e-6)
    
    if head_tilt_ratio > 0.18:
        return {
            "status": "needs_adjustment",
            "issue": "sitting_tilted_head",
            "severity": min(1.0, (head_tilt_ratio - 0.18) / 0.22),
            "feedback": "Your head is tilted. Straighten your neck and look straight ahead."
        }

    # 2. Check for Slouching / Forward Lean
    if metrics:
        torso_lean = metrics["torso_lean"]
        if torso_lean > 0.28:
            return {
                "status": "needs_adjustment",
                "issue": "sitting_slouch",
                "severity": min(1.0, (torso_lean - 0.28) / 0.3),
                "feedback": "You are slouching or leaning forward. Sit up straight."
            }

    # 3. Check Shoulder Levelness
    shoulder_tilt = abs(l_sh['y'] - r_sh['y'])
    sh_tilt_ratio = shoulder_tilt / (shoulder_width + 1e-6)
    if sh_tilt_ratio > 0.12:
        return {
            "status": "needs_adjustment",
            "issue": "sitting_side_lean",
            "severity": min(1.0, (sh_tilt_ratio - 0.12) / 0.15),
            "feedback": "You are leaning to one side. Level your shoulders."
        }

    # 4. Vertical compression (Hunching/Slumped)
    avg_ear_y = (l_ear['y'] + r_ear['y']) / 2.0
    avg_sh_y = (l_sh['y'] + r_sh['y']) / 2.0
    neck_height = (avg_sh_y - avg_ear_y) / (shoulder_width + 1e-6)
    
    if neck_height < 0.28:
        return {
            "status": "needs_adjustment",
            "issue": "sitting_hunch",
            "severity": min(1.0, (0.28 - neck_height) / 0.15),
            "feedback": "Don't hunch your shoulders. Lengthen your neck and sit tall."
        }

    return {
        "status": "good",
        "issue": "sitting_stable",
        "severity": 0.0,
        "feedback": "Excellent sitting form. Spine neutral and shoulders relaxed."
    }

class MockLandmark:
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
    def __getitem__(self, key):
        return getattr(self, key)

def get_hybrid_hint(landmarks_list, lm_array, metrics, clf=None):
    """
    Core Hybrid Logic: Rules FIRST, ML SECOND.
    Returns: (hint_dict, source_name)
    """
    # 1. Check Rules FIRST (Safety/Obvious errors)
    rule_hint = analyze_sitting_posture(landmarks_list, metrics)
    
    if rule_hint['status'] != "good":
        return rule_hint, "Rule Override"
    
    if clf:
        try:
            feats = np.array([_extract_sitting_features(lm_array)], dtype=np.float32)
            pred = clf.predict(feats)[0]
            proba = float(np.max(clf.predict_proba(feats)[0]))
            ml_hint = SITTING_FEEDBACK.get(pred, SITTING_FEEDBACK["goodpose"])
            
            res = ml_hint.copy()
            res["severity"] = round(res.get("severity", 0.0) * proba, 3)
            return res, f"ML Model ({pred})"
        except:
            return rule_hint, "Rules Only (ML Error)"
            
    return rule_hint, "Rules Only"

def _analyze_internal(landmarks, visibility_threshold=0.3):
    required = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_EAR, RIGHT_EAR)
    ok, missing, conf = has_required_visibility(landmarks, required, visibility_threshold)

    if not ok:
        return {
            "success": False,
            "error": "Head or shoulders not clear",
            "feedback": "Please ensure your head and shoulders are fully visible in the frame.",
            "visibility": {"score": round(conf, 3), "missing": _names_from_indices(missing)}
        }

    lm_array = np.array([[p.x, p.y, p.z, p.visibility] for p in landmarks], dtype=np.float32)
    has_hips, _, _ = has_required_visibility(landmarks, (LEFT_HIP, RIGHT_HIP), visibility_threshold)
    metrics = None
    if has_hips:
        _, metrics = compute_engineered_features(landmarks)
    
    import joblib
    clf = None
    if os.path.isfile(_MODEL_PATH):
        try:
            clf = joblib.load(_MODEL_PATH)
        except:
            pass

    hint, source = get_hybrid_hint(landmarks, lm_array, metrics, clf)

    sitting_metrics = _compute_sitting_posture_metrics(landmarks, metrics)

    res = {
        "success": True,
        "pose": "sitting",
        "status": hint["status"],
        "issue": hint["issue"],
        "severity": round(hint["severity"], 3),
        "feedback": hint["feedback"],
        "source": source,
        "metrics": sitting_metrics,
        "visibility": {"score": round(conf, 3), "missing": []}
    }
        
    return res

def analyze_landmarks(landmarks_file, visibility_threshold=0.3):
    try:
        with open(landmarks_file, 'r') as f:
            landmarks_data = json.load(f)
    except Exception:
        return {"success": False, "error": "Unable to read landmarks JSON"}

    if not landmarks_data:
        return {"success": False, "error": "No person detected"}

    landmarks = [MockLandmark(lm.get('x', 0.0), lm.get('y', 0.0), lm.get('z', 0.0), lm.get('visibility', 1.0)) for lm in landmarks_data]
    return _analyze_internal(landmarks, visibility_threshold)

def _extract_sitting_features(lm_array):
    l_ear = lm_array[LEFT_EAR, :2]
    r_ear = lm_array[RIGHT_EAR, :2]
    l_sh  = lm_array[LEFT_SHOULDER, :2]
    r_sh  = lm_array[RIGHT_SHOULDER, :2]
    l_hip = lm_array[LEFT_HIP, :2]
    r_hip = lm_array[RIGHT_HIP, :2]
    nose  = lm_array[NOSE, :2]

    avg_ear = (l_ear + r_ear) / 2.0
    avg_sh  = (l_sh + r_sh) / 2.0
    avg_hip = (l_hip + r_hip) / 2.0

    def dist(a, b): return float(np.linalg.norm(a - b))
    sh_width  = max(dist(l_sh, r_sh), 1e-6)

    return [
        (l_ear[1] - r_ear[1]) / sh_width,
        (l_sh[1]  - r_sh[1])  / sh_width,
        (avg_sh[1] - avg_ear[1]) / sh_width,
        (avg_sh[0] - avg_hip[0]) / sh_width,
        dist(l_ear, r_ear) / sh_width,
        (nose[0] - avg_sh[0]) / sh_width,
        (avg_sh[1] - nose[1]) / sh_width,
        (l_sh[1] - l_ear[1]) / sh_width,
        (r_sh[1] - r_ear[1]) / sh_width,
        abs(l_hip[1] - r_hip[1]) / sh_width,
    ]

def analyze_image(image_file, visibility_threshold=0.3):
    import cv2
    import mediapipe as mp
    image = cv2.imread(image_file)
    if image is None:
        return {"success": False, "error": "Unable to read image file"}
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return {"success": False, "error": "No person detected"}
    landmarks = [MockLandmark(lm.x, lm.y, lm.z, lm.visibility) for lm in results.pose_landmarks.landmark]
    return _analyze_internal(landmarks, visibility_threshold)

def run_camera(visibility_threshold=0.3):
    import cv2
    import joblib
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    clf = None
    if os.path.isfile(_MODEL_PATH):
        try: clf = joblib.load(_MODEL_PATH)
        except: pass

    cap = cv2.VideoCapture(0)
    print("Starting Sitting Posture Coach... Press 'q' to quit.")
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame = cv2.flip(frame, 1)
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            h, w, _ = frame.shape
            status_text, feedback_text, color, pred_source = "Waiting...", "", (255, 255, 255), "None"

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                landmarks_list = [MockLandmark(p.x, p.y, p.z, p.visibility) for p in results.pose_landmarks.landmark]
                lm_array = np.array([[p.x, p.y, p.z, p.visibility] for p in results.pose_landmarks.landmark], dtype=np.float32)
                required = [NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER]
                if any(lm_array[i, 3] < visibility_threshold for i in required):
                    status_text, color = "Adjust Camera", (0, 0, 255)
                else:
                    has_hips, _, _ = has_required_visibility(landmarks_list, (LEFT_HIP, RIGHT_HIP), visibility_threshold)
                    metrics = compute_engineered_features(landmarks_list)[1] if has_hips else None
                    hint, pred_source = get_hybrid_hint(landmarks_list, lm_array, metrics, clf)
                    status_text, feedback_text = f"{pred_source}: {hint['status'].upper()}", hint['feedback']
                    color = (0, 255, 0) if hint['status'] == "good" else (0, 165, 255)

            cv2.rectangle(frame, (0, 0), (w, 60), (40, 40, 40), -1)
            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if feedback_text: cv2.putText(frame, feedback_text, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow('Sitting Posture Coach', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks", help="Path to landmarks JSON")
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--camera", action="store_true")
    # Add optional arguments for backend compatibility
    parser.add_argument("--pose", help="Target pose name (optional)")
    parser.add_argument("--previous_feedback", help="Previous feedback text (optional)")
    parser.add_argument("--previous_metrics", help="Previous metrics JSON (optional)")
    
    args = parser.parse_args()
    if (not args.image and not args.landmarks) or args.camera: run_camera()
    elif args.image: result = analyze_image(args.image)
    elif args.landmarks: result = analyze_landmarks(args.landmarks)
    sys.stdout.write(json.dumps(result))

if __name__ == "__main__":
    main()
