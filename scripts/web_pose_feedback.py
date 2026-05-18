import argparse
import json
import os
import random
import sys

try:
    from groq import Groq
except ImportError:
    Groq = None

import joblib
import mediapipe as mp
import numpy as np

try:
    from scripts.pose_features import (
        COMMON_REQUIRED,
        compute_engineered_features,
        feature_vector,
        has_required_visibility,
        required_indices_for_pose,
    )
except ModuleNotFoundError:
    from pose_features import (  # type: ignore
        COMMON_REQUIRED,
        compute_engineered_features,
        feature_vector,
        has_required_visibility,
        required_indices_for_pose,
    )

# Load ML Model
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
_model_path = os.path.join(_project_root, "models", "pose_classifier.pkl")
_label_path = os.path.join(_project_root, "models", "label_encoder.pkl")

try:
    _blob = joblib.load(_model_path)
    if isinstance(_blob, dict) and "model" in _blob:
        MODEL = _blob["model"]
    else:
        MODEL = _blob
    LABEL_ENCODER = joblib.load(_label_path)
except Exception:
    MODEL = None
    LABEL_ENCODER = None

mp_pose = mp.solutions.pose


def _names_from_indices(indices):
    names = [mp_pose.PoseLandmark(i).name for i in indices]
    return [name.replace("LEFT_", "").replace("RIGHT_", "") for name in names]


def _smooth_metrics(current, previous, alpha=0.4):
    if not previous:
        return current
    smoothed = {}
    for k, v in current.items():
        if isinstance(v, (int, float)) and k in previous and k != "side":
            smoothed[k] = alpha * v + (1 - alpha) * previous[k]
        else:
            smoothed[k] = v
    return smoothed


def _squat_phase(knee_angle, prev_knee_angle=None):
    if knee_angle < 105:
        return "bottom"
    
    # Directional logic using velocity
    if prev_knee_angle is not None:
        diff = knee_angle - prev_knee_angle
        if diff < -1.2:
            return "descending"
        if diff > 1.2:
            return "ascending"

    if knee_angle < 155:
        return "active"
    return "standing"


def build_pose_hint(selected_pose, metrics, prev_metrics=None, ml_results=None):
    pose = (selected_pose or "").lower()
    knee_angle = metrics["knee_angle"]
    hip_angle = metrics["hip_angle"]
    torso_lean = metrics["torso_lean"]
    hip_height_delta = metrics["hip_height_delta"]

    prev_knee = prev_metrics.get("knee_angle") if prev_metrics else None

    # Neural Gating: Check if ML model is confident that form is good
    is_ml_safe = False
    if ml_results and ml_results["top_class"].endswith("_good"):
        if ml_results["top_prob"] > 0.88:
            is_ml_safe = True

    if pose == "squat":
        phase = _squat_phase(knee_angle, prev_knee)
        
        # Prioritize Back Safety (Torso Fold)
        if phase in {"descending", "bottom"} and torso_lean > 0.23:
            # If ML says it's good, maybe it's just a long torso? 
            # But safety first if confidence is low.
            if not (is_ml_safe and ml_results["top_prob"] > 0.95):
                severity = min(1.0, (torso_lean - 0.23) / 0.18)
                return {
                    "status": "needs_adjustment",
                    "issue": "squat_torso_fold",
                    "phase": phase,
                    "severity": severity,
                    "feedback": random.choice([
                        "Keep chest up and brace your core to avoid folding forward.",
                        "Chest up! Don't let your torso collapse.",
                        "Look ahead and keep your heart lifted.",
                        "Brace your core to stay more upright."
                    ]),
                }

        # Check Depth
        if phase == "bottom" and knee_angle > 112:
            severity = min(1.0, (knee_angle - 112) / 38)
            return {
                "status": "needs_adjustment",
                "issue": "squat_shallow_depth",
                "phase": phase,
                "severity": severity,
                "feedback": random.choice([
                    "Go a little deeper while keeping heels grounded.",
                    "Try to get your hips lower for full depth.",
                    "A bit deeper! Aim for thighs parallel to the floor.",
                    "Sink those hips back and down a touch more."
                ]),
            }

        # Standing / Ready Phase
        if phase == "standing":
            return {
                "status": "good",
                "issue": "squat_ready",
                "phase": phase,
                "severity": 0.0,
                "feedback": random.choice([
                    "Great setup. Start when ready.",
                    "Ready to go. Keep that core tight.",
                    "Perfect stance. Send hips back to begin.",
                    "Solid start position."
                ]),
            }

        # Default Good Phase
        return {
            "status": "good",
            "issue": "squat_stable",
            "phase": phase,
            "severity": 0.0,
            "feedback": random.choice([
                "Solid squat pattern. Keep it up!",
                "Great rhythm. Knees tracking perfectly.",
                "Nice control on the movement.",
                "Strong reps, keep pushing through mid-foot."
            ]),
        }

    if pose == "plank":
        if hip_height_delta > 0.07:
            severity = min(1.0, (hip_height_delta - 0.07) / 0.16)
            return {
                "status": "needs_adjustment",
                "issue": "plank_hips_low",
                "phase": "hold",
                "severity": severity,
                "feedback": random.choice([
                    "Lift your hips slightly by squeezing glutes and core.",
                    "Don't let your hips sag! Pull them up.",
                    "Engage your core to lift your midsection.",
                    "Hips are a bit low, bring them up to neutral."
                ]),
            }
        if hip_height_delta < -0.07:
            severity = min(1.0, (abs(hip_height_delta) - 0.07) / 0.16)
            return {
                "status": "needs_adjustment",
                "issue": "plank_hips_high",
                "phase": "hold",
                "severity": severity,
                "feedback": random.choice([
                    "Lower hips a bit to align shoulders, hips, and ankles.",
                    "Hips are too high! Flatten your back.",
                    "Bring your hips down into a straight line.",
                    "Avoid the pike position, lower those hips."
                ]),
            }
        return {
            "status": "good",
            "issue": "plank_stable",
            "phase": "hold",
            "severity": 0.0,
            "feedback": random.choice([
                "Good plank line. Keep neck neutral.",
                "Rock solid plank. Breathe through it.",
                "Perfect alignment from head to heels.",
                "Strong core engagement!"
            ]),
        }

    if pose == "downdog":
        if hip_height_delta > -0.05:
            severity = min(1.0, (hip_height_delta + 0.05) / 0.18)
            return {
                "status": "needs_adjustment",
                "issue": "downdog_hips_low",
                "phase": "hold",
                "severity": severity,
                "feedback": random.choice([
                    "Send hips up and back to lengthen your spine.",
                    "Push through your hands to lift your hips higher.",
                    "Hips up! Aim for an inverted V-shape.",
                    "Lengthen your back by pushing hips toward the ceiling."
                ]),
            }
        return {
            "status": "good",
            "issue": "downdog_stable",
            "phase": "hold",
            "severity": 0.0,
            "feedback": random.choice([
                "Nice down dog shape. Press through palms.",
                "Great inversion! Lengthen that back.",
                "Good stretch! Keep sending hips high.",
                "Strong down dog position."
            ]),
        }

    if pose == "tree":
        # Increased threshold from 0.10 to 0.50 because shoulder-width 
        # based normalization makes 0.10 too sensitive.
        if torso_lean > 0.50:
            severity = min(1.0, (torso_lean - 0.50) / 0.25)
            return {
                "status": "needs_adjustment",
                "issue": "tree_torso_lean",
                "phase": "hold",
                "severity": severity,
                "feedback": random.choice([
                    "Stack ribs over hips and fix your gaze for balance.",
                    "Stay upright! Don't lean into the supporting leg.",
                    "Center your weight over your standing foot.",
                    "Find your vertical axis, stay tall."
                ]),
            }
        return {
            "status": "good",
            "issue": "tree_stable",
            "phase": "hold",
            "severity": 0.0,
            "feedback": random.choice([
                "Good tree pose balance. Breathe steadily.",
                "Stable and focused. Nice job.",
                "Great centering. Keep hips level.",
                "Rock solid tree pose!"
            ]),
        }

    if pose in {"warrior2", "goddess"}:
        if knee_angle > 125:
            severity = min(1.0, (knee_angle - 125) / 45)
            return {
                "status": "needs_adjustment",
                "issue": f"{pose}_knee_bend",
                "phase": "hold",
                "severity": severity,
                "feedback": random.choice([
                    "Bend your knee more and keep it tracking over toes.",
                    "Sink deeper into that front knee.",
                    "Try to get your thigh closer to parallel with the floor.",
                    "More bend! Build that leg strength."
                ]),
            }
        if hip_angle < 145:
            severity = min(1.0, (145 - hip_angle) / 40)
            return {
                "status": "needs_adjustment",
                "issue": f"{pose}_torso_collapsed",
                "phase": "hold",
                "severity": severity,
                "feedback": random.choice([
                    "Lift your torso taller and keep your chest open.",
                    "Don't lean forward, keep your spine vertical.",
                    "Open your heart and stay tall through the crown of your head.",
                    "Brace your core to keep your torso upright."
                ]),
            }
        return {
            "status": "good",
            "issue": f"{pose}_stable",
            "phase": "hold",
            "severity": 0.0,
            "feedback": random.choice([
                f"Strong {pose} stance. Stay grounded.",
                f"Powerful {pose}! Keep that chest open.",
                "Nice alignment. Breath through the hold.",
                "Solid foundation."
            ]),
        }

    return {
        "status": "good",
        "issue": "posture_stable",
        "phase": "hold",
        "severity": 0.0,
        "feedback": random.choice([
            "Good posture.",
            "Form looks solid.",
            "Keep it up!",
            "Great hold."
        ]),
    }


def generate_groq_feedback(pose, status, issue, phase, severity, fallback_feedback, previous_feedback=None):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or Groq is None:
        return fallback_feedback

    try:
        client = Groq(api_key=api_key, max_retries=1)
        
        avoid_instruction = f" Do NOT repeat or use similar phrasing to: '{previous_feedback}'." if previous_feedback else ""

        if status == "good":
            prompt = (
                f"You are an encouraging AI gym coach evaluating a user's {pose} form. "
                f"The user is currently doing a great job (phase: {phase}). "
                f"Give a short, dynamic, 1-sentence compliment or encouragement (under 10 words).{avoid_instruction} "
                f"Do not use quotes. Do not say 'Here is a ...'"
            )
        else:
            prompt = (
                f"You are an encouraging AI gym coach evaluating a user's {pose} form. "
                f"Phase: {phase}. Issue detected: {issue}. Severity: {severity:.2f} (0 is minor, 1 is major). "
                f"The standard advice is: '{fallback_feedback}'. "
                f"Provide a short, punchy 1-sentence verbal cue to correct this.{avoid_instruction} "
                f"Keep it under 15 words. Be dynamic and varied. Do not use quotes."
            )

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.9, # Increased for more variety
            max_tokens=30,
            timeout=2.0
        )
        content = chat_completion.choices[0].message.content.strip()
        return content.strip('"\'')
    except Exception:
        return fallback_feedback


class MockLandmark:
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def analyze_landmarks(landmarks_file, selected_pose, visibility_threshold=0.3, previous_feedback=None, previous_metrics=None):
    try:
        with open(landmarks_file, 'r') as f:
            landmarks_data = json.load(f)
    except Exception as e:
        return {
            "success": False,
            "error": "Unable to read landmarks JSON"
        }

    if not landmarks_data:
        return {
            "success": False,
            "error": "No person detected",
            "feedback": "Move your whole body into frame and try again."
        }

    landmarks = []
    for lm in landmarks_data:
        # Default visibility to 1.0 if not provided by the JS client
        landmarks.append(MockLandmark(
            lm.get('x', 0.0),
            lm.get('y', 0.0),
            lm.get('z', 0.0),
            lm.get('visibility', lm.get('visibility', 1.0))
        ))

    common_ok, common_missing, common_conf = has_required_visibility(
        landmarks,
        COMMON_REQUIRED,
        visibility_threshold,
    )

    if not common_ok:
        return {
            "success": False,
            "error": "Low landmark confidence",
            "feedback": "I need a clearer full-body view before giving feedback.",
            "visibility": {
                "score": round(common_conf, 3),
                "missing": _names_from_indices(common_missing)[:6],
            },
        }

    pose_required = required_indices_for_pose(selected_pose)
    pose_ok, pose_missing, pose_conf = has_required_visibility(
        landmarks,
        pose_required,
        visibility_threshold,
    )

    if not pose_ok:
        return {
            "success": False,
            "error": "Selected pose landmarks not clear",
            "feedback": "Adjust camera angle so key joints for this pose are visible.",
            "visibility": {
                "score": round(pose_conf, 3),
                "missing": _names_from_indices(pose_missing)[:6],
            },
        }

    # 1. Compute Raw Features
    raw_features, raw_metrics = compute_engineered_features(landmarks)

    # 2. Apply Temporal Smoothing (EMA)
    smoothed_metrics = _smooth_metrics(raw_metrics, previous_metrics, alpha=0.45)

    ml_results = None
    if MODEL:
        try:
            feats = feature_vector(raw_features)
            X = np.array([feats])
            probs = MODEL.predict_proba(X)[0]
            top_idx = np.argmax(probs)
            ml_results = {
                "top_class": str(LABEL_ENCODER.inverse_transform([top_idx])[0]),
                "top_prob": float(probs[top_idx])
            }
        except Exception:
            pass

    # 4. Generate Feedback Hint
    hint = build_pose_hint(selected_pose, smoothed_metrics, prev_metrics=previous_metrics, ml_results=ml_results)
    
    # 5. Dynamic LLM Feedback
    dynamic_feedback = generate_groq_feedback(
        selected_pose,
        hint["status"],
        hint["issue"],
        hint["phase"],
        hint["severity"],
        hint["feedback"],
        previous_feedback=previous_feedback
    )

    return {
        "success": True,
        "pose": selected_pose,
        "status": hint["status"],
        "issue": hint["issue"],
        "phase": hint["phase"],
        "severity": round(hint["severity"], 3),
        "feedback": dynamic_feedback,
        "metrics": {
            "knee_angle": round(smoothed_metrics["knee_angle"], 1),
            "hip_angle": round(smoothed_metrics["hip_angle"], 1),
            "torso_lean": round(smoothed_metrics["torso_lean"], 3),
            "hip_height_delta": round(smoothed_metrics["hip_height_delta"], 3),
            "side": smoothed_metrics["side"],
        },
        "ml_ai": ml_results,
        "visibility": {
            "score": round(pose_conf, 3),
            "missing": [],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks", required=True)
    parser.add_argument("--pose", required=True)
    parser.add_argument("--previous_feedback", required=False)
    parser.add_argument("--previous_metrics", required=False)
    args = parser.parse_args()

    prev_metrics = None
    if args.previous_metrics:
        try:
            prev_metrics = json.loads(args.previous_metrics)
        except Exception:
            pass

    result = analyze_landmarks(
        args.landmarks, 
        args.pose, 
        previous_feedback=args.previous_feedback,
        previous_metrics=prev_metrics
    )
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
