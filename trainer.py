import cv2
import PoseEstimationModule as pem
import numpy as np
import time

class AITrainer:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.detector = pem.poseDetector()
        
        self.counters = {
            'Dumbbells': 0,
            'Jumps': 0,
            'Jumping Jacks': 0,
            'Twisters': 0,
            'Pushups': 0,
            'Squats': 0
        }
        
        # Enhanced exercise states with completion tracking
        self.states = {
            'dumbbells': {
                'phase': 'up', 
                'ready': False,
                'min_angle': 160,  # Arm extended
                'max_angle': 60    # Arm curled
            },
            'jumps': {
                'phase': 'up',
                'ready': False,
                'eye_threshold': 100,
                'hip_diff_threshold': 50
            },
            'jumping_jacks': {
                'phase': 'closed',
                'ready': False,
                'hand_threshold': 500,
                'foot_threshold': 300
            },
            'twisters': {
                'phase': 'center',
                'ready': False,
                'angle_threshold': 30
            },
            'pushups': {
                'phase': 'up',
                'ready': False,
                'min_angle': 140,
                'max_angle': 90
            },
            'squats': {
                'phase': 'up',
                'ready': False,
                'min_dist': 150,
                'max_dist': 200
            }
        }
        
        self.pTime = time.time()
        self.progress = 0

    def run(self):
        while True:
            success, img = self.cap.read()
            if not success:
                break

            img = cv2.flip(img, 1)
            img = self.detector.findPose(img, draw=True)
            lmList = self.detector.getPosition(img, draw=False)

            if lmList and len(lmList) >= 28:
                try:
                    self.process_exercises(img, lmList)
                except Exception as e:
                    print(f"Error: {e}")

            self.display_ui(img)
            
            cv2.imshow("AI Personal Trainer", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def process_exercises(self, img, lmList):
        # Get key points
        left_shoulder = lmList[12][1:]
        right_shoulder = lmList[11][1:]
        left_elbow = lmList[14][1:]
        right_elbow = lmList[13][1:]
        left_wrist = lmList[16][1:]
        right_wrist = lmList[15][1:]
        left_hip = lmList[24][1:]
        right_hip = lmList[23][1:]
        left_knee = lmList[26][1:]
        right_knee = lmList[25][1:]
        eye = lmList[1][1:]

        # Calculate metrics
        arm_angle = self.detector.findAngle(img, 12, 14, 16)
        hip_knee_dist, _, _ = self.detector.findDistance(left_hip, left_knee)
        hand_dist, _, _ = self.detector.findDistance(left_wrist, right_wrist)
        cross_dist, _, _ = self.detector.findDistance(left_wrist, right_hip)
        hip_diff = abs(left_hip[1] - right_hip[1])

        # Exercise detection - only counts when full movement is completed
        self.detect_dumbbells(arm_angle)
        self.detect_jumps(eye, hip_diff)
        self.detect_jumping_jacks(hand_dist)
        self.detect_twisters(cross_dist)
        self.detect_pushups(arm_angle, eye)
        self.detect_squats(hip_knee_dist, eye)

        # Progress bar decay
        self.progress = max(0, self.progress - 2)

    def detect_dumbbells(self, angle):
        state = self.states['dumbbells']
        
        # Count only when completing full curl up and down
        if angle > state['min_angle'] and state['phase'] == 'up':
            state['phase'] = 'down'
            state['ready'] = True  # Ready to count when coming back up
        elif angle < state['max_angle'] and state['phase'] == 'down' and state['ready']:
            self.counters['Dumbbells'] += 1
            state['phase'] = 'up'
            state['ready'] = False
            self.progress = 100
            print("Dumbbell curl counted!")

    def detect_jumps(self, eye, hip_diff):
        state = self.states['jumps']
        
        # Count when landing after jump
        if eye[1] < state['eye_threshold'] and hip_diff < state['hip_diff_threshold']:
            if state['phase'] == 'up':
                state['phase'] = 'down'
                state['ready'] = True  # Ready to count when standing up
        elif eye[1] > state['eye_threshold'] + 50 and state['phase'] == 'down' and state['ready']:
            self.counters['Jumps'] += 1
            state['phase'] = 'up'
            state['ready'] = False
            self.progress = 100
            print("Jump counted!")

    def detect_jumping_jacks(self, hand_dist):
        state = self.states['jumping_jacks']
        
        # Count when returning to closed position
        if hand_dist > state['hand_threshold']:
            if state['phase'] == 'closed':
                state['phase'] = 'open'
                state['ready'] = True  # Ready to count when closing
        elif hand_dist < state['hand_threshold'] - 100 and state['phase'] == 'open' and state['ready']:
            self.counters['Jumping Jacks'] += 1
            state['phase'] = 'closed'
            state['ready'] = False
            self.progress = 100
            print("Jumping jack counted!")

    def detect_twisters(self, cross_dist):
        state = self.states['twisters']
        
        # Count when returning to center
        if cross_dist < 100:
            if state['phase'] == 'center':
                state['phase'] = 'twisted'
                state['ready'] = True  # Ready to count when untwisting
        elif cross_dist > 150 and state['phase'] == 'twisted' and state['ready']:
            self.counters['Twisters'] += 1
            state['phase'] = 'center'
            state['ready'] = False
            self.progress = 100
            print("Twister counted!")

    def detect_pushups(self, angle, eye):
        state = self.states['pushups']
        
        # Count when pushing back up
        if angle > state['min_angle'] and eye[1] > 400 and state['phase'] == 'up':
            state['phase'] = 'down'
            state['ready'] = True  # Ready to count when pushing up
        elif angle < state['max_angle'] and state['phase'] == 'down' and state['ready']:
            self.counters['Pushups'] += 1
            state['phase'] = 'up'
            state['ready'] = False
            self.progress = 100
            print("Pushup counted!")

    def detect_squats(self, distance, eye):
        state = self.states['squats']
        
        # Count when standing back up
        if distance < state['min_dist'] and eye[1] < 400 and state['phase'] == 'up':
            state['phase'] = 'down'
            state['ready'] = True  # Ready to count when standing
        elif distance > state['max_dist'] and state['phase'] == 'down' and state['ready']:
            self.counters['Squats'] += 1
            state['phase'] = 'up'
            state['ready'] = False
            self.progress = 100
            print("Squat counted!")

    def display_ui(self, img):
        # Display counters (left side)
        y_pos = 150
        for exercise, count in self.counters.items():
            cv2.putText(img, exercise, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 0, 0), 2)
            cv2.putText(img, str(count), (50, y_pos + 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.5, (0, 0, 255), 3)
            y_pos += 100

        # Progress bar (right side)
        cv2.rectangle(img, (1100, 200), (1150, 700), (0, 255, 0), 3)
        progress_height = int(500 * (self.progress / 100))
        cv2.rectangle(img, (1100, 700 - progress_height), (1150, 700), 
                     (0, 255, 255), cv2.FILLED)
        cv2.putText(img, f"{int(self.progress)}%", (1100, 750), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        # Screen recorder text (top right)
        cv2.putText(img, "SCREEN RECORDER", (1000, 50), 
                   cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

        # Display FPS
        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (50, 50), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

if __name__ == "__main__":
    trainer = AITrainer()
    trainer.run()