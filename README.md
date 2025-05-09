# AI Personal Trainer

## Overview
The AI Personal Trainer is a computer vision application that uses pose estimation to track and count various exercises in real-time. It provides visual feedback and counts repetitions for exercises like dumbbell curls, jumps, jumping jacks, twisters, pushups, and squats.

## Features
- Real-time pose estimation using MediaPipe
- Exercise repetition counting for 6 different exercises
- Visual progress bar and counters
- FPS display for performance monitoring
- User-friendly interface with exercise statistics

## Supported Exercises
1. **Dumbbell Curls**: Tracks arm angles to count curls.
2. **Jumps**: Detects jumps using eye and hip positions.
3. **Jumping Jacks**: Measures hand distance to count repetitions.
4. **Twisters**: Tracks torso twists using cross-body hand distance.
5. **Pushups**: Monitors arm angles and body position.
6. **Squats**: Measures hip-to-knee distance for counting.

## Requirements
- Python 3.7+
- OpenCV (`pip install opencv-python`)
- MediaPipe (`pip install mediapipe`)
- NumPy (`pip install numpy`)

## Installation
1. Clone the repository:
   
   git clone https://github.com/Reddybangaram123/Ai-Personal-Trainer.git
   
   cd Ai-Personal-Trainer
   
2. Install dependencies:
   
   pip install -r requirements.txt
   

## Usage
1. Run the application:
   
   python My_AI_Trainer.py

2. Position yourself in front of the camera to start exercise tracking.
3. Press `q` to quit the application.

## File Structure
- `trainer.py`: Main application script
- `PoseEstimationModule.py`: Custom pose estimation module

## How It Works
1. The application captures video from your webcam.
2. MediaPipe processes each frame to detect body landmarks.
3. The AI Trainer analyzes these landmarks to detect specific exercises.
4. Counters increment when full exercise movements are completed.
5. Visual feedback is displayed on-screen.

## Customization
You can adjust the following parameters in `trainer.py`:
- Exercise thresholds in the `states` dictionary
- Display positions and colors
- Camera resolution settings

## Troubleshooting
- Ensure proper lighting for better pose detection.
- Make sure your whole body is visible in the frame.
- If exercises aren't being counted, adjust the thresholds in the code.

## Future Improvements
- Add more exercise types
- Implement a workout timer
- Add voice feedback
- Save workout statistics
