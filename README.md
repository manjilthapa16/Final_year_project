# Your Gym Buddy

Your Gym Buddy is a real-time posture coach using MediaPipe + a classifier, with deterministic coaching cues and optional LLM tone polishing. It supports multiple exercises including squats, planks, tree pose, and sitting posture.

## Features
- **Rotation-invariant feature engineering**: normalized joint angles + relative distances (not raw landmarks only).
- **Hybrid Coaching**: Rule-based safety/form cues are primary, supplemented by ML classification.
- **Optional LLM polish**: Groq rewrites deterministic cues for tone and personalization.
- **Squat phase state machine**: `descent -> bottom -> ascent` plus rep counting.
- **Pose-specific Analysis**: Specialized logic for Squat, Plank, Downward Dog, Goddess, and Tree pose.
- **Sitting Posture Correction**: Real-time feedback for seated workers to prevent hunching.
- **Dataset quality tooling**: Class-balance report + training-time balancing.

## Project Structure
```text
Final_year_project/
├── data/
│   ├── processed/          # Preprocessed landmark CSVs
│   └── raw/                # Image dataset categorized by pose
├── models/                 # Saved .pkl models and encoders
├── scripts/
│   ├── pose_features.py    # Shared feature engineering logic
│   ├── extract_landmarks.py # Landmark extraction from dataset
│   ├── train_classifier.py # ML model training with rebalancing
│   ├── evaluate_model.py   # Model evaluation and confusion matrix
│   ├── data_quality_report.py # Dataset imbalance analysis
│   ├── feedback_agent.py   # LLM integration (Groq)
│   ├── sitting_pose.py     # Seated posture analysis script
│   └── squat_analysis.py   # Squat-specific hybrid logic
├── gym_buddy/              # Flutter-based mobile application
├── web/                    # Full-stack web application
│   ├── client/             # React (Vite) frontend
│   └── server/             # Node.js (Express) backend
├── posture_analyzer.py     # Main CLI/Real-time camera application
├── requirements.txt        # Python dependencies
└── README.md
```

## Setup
```bash
cd ~/Desktop/Final_year_project
# Create and activate venv
python3 -m venv venv
source venv/bin/activate
# Install dependencies
pip install -r requirements.txt
```

## Prepare Data + Train
```bash
# 1. Extract landmarks from raw images
python3 -m scripts.extract_landmarks
# 2. Check data balance
python3 -m scripts.data_quality_report
# 3. Train the classifier
python3 -m scripts.train_classifier
# 4. Evaluate performance (generates confusion matrix)
python3 -m scripts.evaluate_model
```

## Run
```bash
# Optional: Set up Groq API Key for LLM feedback
echo "GROQ_API_KEY=your_key_here" > .env

# Run the main analyzer (Camera mode)
python3 posture_analyzer.py

# Run pose-specific scripts
python3 -m scripts.sitting_pose --camera
python3 -m scripts.squat_analysis --camera
```

## Web & Mobile
- **Web App**: Navigate to `web/server` and `web/client` for setup. The web app uses a shared backend to analyze pose landmarks sent from the browser.
- **Mobile App**: The `gym_buddy/` directory contains a Flutter project for cross-platform mobile support.

## Runtime Notes
- **Confidence Gating**: Analysis only runs when key joints are visible to prevent erroneous feedback.
- **Hybrid Logic**: The system prioritizes deterministic rules (e.g., knee angles, hip height) to ensure safety before applying ML classification.
- **LLM Usage**: Groq rephrases the rule-based cues. If no API key is provided, the system defaults to clear, deterministic text.
