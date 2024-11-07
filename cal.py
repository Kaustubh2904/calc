import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
from collections import deque
import time
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

class AirCanvas:
    def __init__(self, width=1400, height=800):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.temp_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.prev_point = None
        self.expression = ""
        self.result = None
        self.is_drawing = False
        self.current_stroke = []
        self.symbols = ['+', '-', '*', '/', '=', 'C']
        self.symbol_positions = [(width - 150, 100 + i * 100) for i in range(len(self.symbols))]
        self.symbol_cooldown = 1.0
        self.last_symbol_time = 0
        self.selected_symbol = None
        self.symbol_selection_start = None

        # Debug mode
        self.debug_mode = True

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize camera
        self.cap = None
        self.setup_camera()

        # Load the digit recognition model
        try:
            self.model = load_model('digit_recognition_model.h5')
            print("Model loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Running without digit recognition model")
            self.model = None

    def setup_camera(self):
        """Initialize and configure the webcam"""
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Failed to open webcam")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Check if the camera settings were applied successfully
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if actual_width != self.width or actual_height != self.height:
            print(f"Warning: Camera resolution set to {actual_width}x{actual_height} instead of requested {self.width}x{self.height}")
            self.width = int(actual_width)
            self.height = int(actual_height)
            # Update canvas dimensions
            self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.temp_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # Update symbol positions for new width
            self.symbol_positions = [(self.width - 150, 100 + i * 100) for i in range(len(self.symbols))]

    def get_finger_state(self, hand_landmarks):
        index_tip = hand_landmarks.landmark[8].y
        index_pip = hand_landmarks.landmark[7].y
        middle_tip = hand_landmarks.landmark[12].y
        middle_pip = hand_landmarks.landmark[11].y

        finger_height = abs(hand_landmarks.landmark[8].y - hand_landmarks.landmark[5].y)
        threshold = finger_height * 0.2

        index_up = (index_pip - index_tip) > threshold
        middle_up = (middle_pip - middle_tip) > threshold

        return index_up and not middle_up

    def draw_symbols(self, frame):
        for i, (x, y) in enumerate(self.symbol_positions):
            # Different colors for different button types
            if self.symbols[i] in ['=', 'C']:
                bg_color = (50, 100, 50)  # Green background for action buttons
            else:
                bg_color = (50, 50, 50)   # Original color for operators

            # Create a circle background for each symbol
            cv2.circle(frame, (x, y), 30, bg_color, -1)

            # Highlight the symbol if it's being selected
            if self.selected_symbol == self.symbols[i]:
                cv2.circle(frame, (x, y), 30, (0, 255, 0), -1)

            # Draw the symbol
            cv2.putText(frame, self.symbols[i], (x-15, y+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    def check_symbol_selection(self, x, y):
        current_time = time.time()

        # If we're in cooldown period, return None
        if current_time - self.last_symbol_time < self.symbol_cooldown:
            return None

        for i, (sx, sy) in enumerate(self.symbol_positions):
            if np.sqrt((x - sx)**2 + (y - sy)**2) < 30:
                if self.symbol_selection_start is None:
                    self.symbol_selection_start = current_time
                    self.selected_symbol = self.symbols[i]
                    return None

                # If finger has been hovering for more than 1 second
                elif current_time - self.symbol_selection_start > 1.0:
                    self.last_symbol_time = current_time
                    self.symbol_selection_start = None
                    selected = self.symbols[i]
                    self.selected_symbol = None

                    # Handle action buttons
                    if selected == '=':
                        self.compute_expression()
                        return None
                    elif selected == 'C':
                        self.reset_canvas()
                        return None
                    return selected

                return None

        self.symbol_selection_start = None
        self.selected_symbol = None
        return None

    def reset_canvas(self):
        """Reset the canvas and clear the expression"""
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.expression = ""
        self.result = None

    def compute_expression(self):
        """Compute the current expression"""
        try:
            if self.expression:
                self.result = eval(self.expression)
                print(f"Result: {self.result}")
                self.expression = str(self.result)
        except Exception as e:
            print(f"Error in evaluating expression: {e}")
            self.result = "Error"

    def draw_result(self, frame):
        if self.result is not None:
            # Draw a semi-transparent background rectangle
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, self.height-100), (400, self.height-20),
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # Draw the result text
            cv2.putText(frame, f"Result: {self.result}",
                       (20, self.height-50), cv2.FONT_HERSHEY_SIMPLEX,
                       1.5, (0, 255, 0), 2, cv2.LINE_AA)

    # [Rest of the methods remain the same as in your original code]
    def process_stroke(self):
        if not self.current_stroke:
            return None

        try:
            # Create a mask from the stroke
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            points = np.array(self.current_stroke, dtype=np.int32)

            # Draw the stroke
            if len(points) > 1:
                cv2.polylines(mask, [points], False, 255, 5)

                # Fill small gaps in the stroke
                kernel = np.ones((5,5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find the bounding box of the stroke
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            # Get the largest contour
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            # Extract the region of interest (ROI)
            roi = mask[max(0, y-10):min(self.height, y+h+10),
                      max(0, x-10):min(self.width, x+w+10)]

            if roi.size == 0:
                return None

            # Preprocess the ROI
            processed_roi = self.preprocess_for_digit(roi)
            if processed_roi is None:
                return None

            # Recognize the digit
            digit = self.recognize_digit(processed_roi)

            # Clear the current stroke from the canvas
            if digit is not None:
                cv2.polylines(self.canvas, [points], False, (0, 0, 0), 7)

            return digit

        except Exception as e:
            print(f"Error in process_stroke: {e}")
            return None

    def preprocess_for_digit(self, roi):
        try:
            # Ensure minimum size
            min_size = 28
            if roi.shape[0] < min_size or roi.shape[1] < min_size:
                scale = min_size / min(roi.shape[0], roi.shape[1])
                roi = cv2.resize(roi, None, fx=scale, fy=scale)

            # Convert to grayscale if needed
            if len(roi.shape) == 3:
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Apply thresholding
            _, roi = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY)

            # Find contours and get the largest one
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            # Get the largest contour
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            roi = roi[y:y+h, x:x+w]

            # Add padding
            pad = int(min(w, h) * 0.2)
            roi = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

            # Resize to model input size
            roi = cv2.resize(roi, (28, 28))

            # Normalize
            roi = roi.astype('float32') / 255.0

            # Debug visualization
            if self.debug_mode:
                cv2.imshow('Preprocessed Digit', roi)

            return roi
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None

    def recognize_digit(self, roi):
        if self.model is None:
            return None

        try:
            # Prepare the input
            roi = np.expand_dims(roi, axis=(0, -1))

            # Get prediction
            pred = self.model.predict(roi, verbose=0)
            confidence = np.max(pred)
            digit = np.argmax(pred)

            if self.debug_mode:
                print(f"Confidence: {confidence}, Predicted digit: {digit}")

            # Only return digits with high confidence
            if confidence > 0.8:  # Adjust this threshold as needed
                # Show confidence in UI
                if len(self.current_stroke) > 0:
                    cv2.putText(self.canvas, f"{digit} ({confidence:.2f})",
                              self.current_stroke[-1],
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                return str(digit)

            return None

        except Exception as e:
            print(f"Error in digit recognition: {e}")
            return None

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        # Draw user interface elements
        self.draw_symbols(frame)

        # Draw expression box at the top
        cv2.rectangle(frame, (10, 10), (400, 70), (50, 50, 50), -1)
        cv2.putText(frame, self.expression, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger = hand_landmarks.landmark[8]
                x = int(index_finger.x * self.width)
                y = int(index_finger.y * self.height)

                should_draw = self.get_finger_state(hand_landmarks)

                if should_draw:
                    if not self.is_drawing:
                        self.is_drawing = True
                        self.current_stroke = []
                    self.current_stroke.append((x, y))

                    if self.prev_point is not None:
                        cv2.line(self.canvas, self.prev_point, (x, y), (255, 255, 255), 5)

                    self.prev_point = (x, y)
                else:
                    if self.is_drawing:
                        self.is_drawing = False
                        result = self.process_stroke()
                        if result:
                            self.expression += result
                        self.prev_point = None

                selected_symbol = self.check_symbol_selection(x, y)
                if selected_symbol:
                    self.expression += selected_symbol

                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Combine the drawing canvas with the frame
        frame = cv2.add(frame, self.canvas)

        # Draw the result if available
        self.draw_result(frame)

        cv2.imshow('AirCanvas Calculator', frame)

    def run(self):
        print("Controls:")
        print("Gesture Controls:")
        print("- Hover over '=' button - Compute expression")
        print("- Hover over 'C' button - Clear canvas")
        print("'q' - Quit")
        print("'d' - Toggle debug mode")

        try:
            while True:
                self.process_frame()
                key = cv2.waitKey(1) & 0xFF

                if key == ord('d'):
                    self.debug_mode = not self.debug_mode
                elif key == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    air_canvas = AirCanvas()
    air_canvas.run()