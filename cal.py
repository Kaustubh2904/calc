import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
from collections import deque

class AirCanvas:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.temp_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.prev_point = None
        self.drawing_buffer = deque(maxlen=1000)
        self.expression = ""
        self.is_drawing = False
        self.current_stroke = []
        self.strokes = []

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize camera
        self.setup_camera()

        try:
            self.model = load_model('digit_recognition_model.h5')
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Running without digit recognition model")
            self.model = None

    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Failed to open webcam")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def get_finger_state(self, hand_landmarks):
        """Improved finger state detection"""
        index_tip = hand_landmarks.landmark[8].y
        index_pip = hand_landmarks.landmark[7].y
        middle_tip = hand_landmarks.landmark[12].y
        middle_pip = hand_landmarks.landmark[11].y

        # Calculate the height of a finger segment for threshold
        finger_height = abs(hand_landmarks.landmark[8].y - hand_landmarks.landmark[5].y)
        threshold = finger_height * 0.2  # 20% of finger height

        index_up = (index_pip - index_tip) > threshold
        middle_up = (middle_pip - middle_tip) > threshold

        return index_up and not middle_up

    def normalize_stroke(self, stroke):
        """Normalize stroke points to standard size"""
        if not stroke:
            return []

        points = np.array(stroke)
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)

        width = max_x - min_x
        height = max_y - min_y
        if width == 0: width = 1
        if height == 0: height = 1

        normalized = (points - [min_x, min_y]) / [width, height]

        return normalized

    def recognize_operator(self, stroke):
        """Improved operator recognition"""
        if len(stroke) < 4:
            return None

        normalized = self.normalize_stroke(stroke)
        if len(normalized) == 0:
            return None

        points = np.array(stroke)
        x_coords, y_coords = points[:, 0], points[:, 1]
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        aspect_ratio = width / height if height != 0 else float('inf')

        img_size = 28
        pattern = np.zeros((img_size, img_size), dtype=np.uint8)
        scaled_points = (normalized * (img_size-2) + 1).astype(int)

        for i in range(len(scaled_points) - 1):
            cv2.line(pattern, tuple(scaled_points[i]), tuple(scaled_points[i+1]), 255, 1)

        vertical_projection = np.sum(pattern, axis=0)
        horizontal_projection = np.sum(pattern, axis=1)

        deltas = np.diff(points, axis=0)
        angles = np.arctan2(deltas[:, 1], deltas[:, 0])
        direction_changes = np.sum(np.abs(np.diff(angles)) > np.pi/4)

        if 0.8 < aspect_ratio < 1.2:  # Square-like
            h_center = np.sum(horizontal_projection[img_size//3:2*img_size//3]) > 0
            v_center = np.sum(vertical_projection[img_size//3:2*img_size//3]) > 0
            if h_center and v_center and direction_changes <= 4:
                return '+'
            if direction_changes == 2 and np.std(horizontal_projection) < np.mean(horizontal_projection):
                return '*'
        elif aspect_ratio > 2:  # Horizontal shape
            if np.max(horizontal_projection) > 3 * np.mean(horizontal_projection) and direction_changes <= 2:
                return '-'
        elif aspect_ratio < 0.5:  # Vertical shape
            if np.max(vertical_projection) > 3 * np.mean(vertical_projection) and direction_changes <= 2:
                return '/'

        return None

    def preprocess_for_digit(self, roi):
        """Improved preprocessing for digit recognition"""
        try:
            min_size = 28
            if roi.shape[0] < min_size or roi.shape[1] < min_size:
                scale = min_size / min(roi.shape[0], roi.shape[1])
                roi = cv2.resize(roi, None, fx=scale, fy=scale)

            if len(roi.shape) == 3:
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            _, roi = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            roi = roi[y:y+h, x:x+w]

            pad = int(min(w, h) * 0.2)
            roi = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

            roi = cv2.resize(roi, (28, 28))
            roi = roi.astype('float32') / 255.0

            return roi
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None

    def recognize_digit(self, roi):
        """Improved digit recognition"""
        if self.model is None:
            return None

        try:
            processed_roi = self.preprocess_for_digit(roi)
            if processed_roi is None:
                return None

            processed_roi = np.expand_dims(processed_roi, axis=(0, -1))

            pred = self.model.predict(processed_roi, verbose=0)
            confidence = np.max(pred)
            digit = np.argmax(pred)

            if confidence > 0.8:
                return str(digit)

            return None
        except Exception as e:
            print(f"Error in digit recognition: {e}")
            return None

    def process_stroke(self):
        """Process stroke for recognition"""
        if not self.current_stroke:
            return None

        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for i in range(len(self.current_stroke) - 1):
            cv2.line(mask, self.current_stroke[i], self.current_stroke[i + 1], 255, 5)

        operator = self.recognize_operator(self.current_stroke)
        if operator:
            return operator

        digit = self.recognize_digit(mask)
        return digit

    def process_frame(self):
        """Process a single frame"""
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger = hand_landmarks.landmark[8]
                x = int(index_finger.x * self.width)
                y = int(index_finger.y * self.height)

                should_draw = self.get_finger_state(hand_landmarks)

                if should_draw:
                    if not self.is_drawing:
                        self.is_drawing = True
                        self.current_stroke = [(x, y)]
                    else:
                        self.current_stroke.append((x, y))
                        if len(self.current_stroke) > 1:
                            cv2.line(self.temp_canvas, self.current_stroke[-2], self.current_stroke[-1], (255, 255, 255), 5)

                elif self.is_drawing:  # End of stroke
                    self.is_drawing = False
                    if self.current_stroke:
                        result = self.process_stroke()
                        if result:
                            self.expression += result
                            self.canvas = cv2.add(self.canvas, self.temp_canvas)
                        self.temp_canvas = np.zeros_like(self.temp_canvas)
                        self.current_stroke = []

                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        combined = cv2.addWeighted(frame, 0.5, self.canvas, 0.5, 0)
        combined = cv2.addWeighted(combined, 1.0, self.temp_canvas, 0.5, 0)

        cv2.putText(combined, f"Expression: {self.expression}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return combined

    def evaluate_expression(self):
        """Evaluate the mathematical expression"""
        try:
            expr = self.expression.replace('x', '*').replace('÷', '/')
            result = eval(expr)
            cv2.putText(self.canvas, f"Result: {result}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return result
        except Exception as e:
            cv2.putText(self.canvas, "Invalid Expression", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return None

    def run(self):
        """Main loop"""
        print("Controls:")
        print("- Raise index finger to draw")
        print("- Raise both fingers to move without drawing")
        print("- Press 'c' to clear canvas")
        print("- Press 'e' to evaluate expression")
        print("- Press 'q' to quit")

        while True:
            frame = self.process_frame()
            if frame is None:
                break

            cv2.imshow("Air Canvas", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                self.temp_canvas = np.zeros_like(self.canvas)
                self.expression = ""
                self.current_stroke = []
            elif key == ord('e'):
                self.evaluate_expression()

        self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        canvas = AirCanvas()
        canvas.run()
    except Exception as e:
        print(f"Application error: {e}")
