import cv2
import mediapipe as mp
import numpy as np
import pytesseract
from sympy import sympify, solve, Eq
from sympy.parsing.sympy_parser import parse_expr
import re

class LocalAirCalculator:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.expression = ""
        self.result = ""

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def preprocess_canvas(self):
        """Enhanced preprocessing for OCR accuracy, then scaled down for display"""
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=30)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        padding = 50
        padded = cv2.copyMakeBorder(thresh, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        enlarged = cv2.resize(padded, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        final = cv2.fastNlMeansDenoising(enlarged, None, 10, 7, 21)

        # Scaled down for display
        display_final = cv2.resize(final, (self.width // 2, self.height // 2))
        cv2.imshow("Processed Canvas", display_final)

        return final

    def recognize_expression(self):
        """Recognize expression with OCR and improved processing"""
        try:
            processed_img = self.preprocess_canvas()
            custom_config = r'--oem 1 --psm 7 -c tessedit_char_whitelist="0123456789+-*/()=xX. "'
            recognized_text = pytesseract.image_to_string(processed_img, config=custom_config).strip()
            cleaned = re.sub(r'[^0-9+\-*/()=xX\s.]', '', recognized_text).replace("X", "x")
            return cleaned if self.is_valid_expression(cleaned) else None
        except Exception as e:
            print(f"Recognition error: {e}")
            return None

    def is_valid_expression(self, expr):
        valid_chars = set('0123456789+-*/()=x.')
        if expr and all(c in valid_chars for c in expr) and expr.count('(') == expr.count(')'):
            return not any(seq in expr for seq in ['++', '--', '**', '//', '==', '+=', '-=', '*=', '/='])
        return False

    def evaluate_expression(self, expression):
        """Evaluate the mathematical expression with error handling"""
        try:
            if '=' in expression:
                left, right = expression.split('=')
                if not self.is_valid_expression(left) or not self.is_valid_expression(right):
                    return "Invalid equation structure"
                eq = Eq(parse_expr(left), parse_expr(right))
                solution = solve(eq)
                if solution:
                    if isinstance(solution, list):
                        return "x = " + ', '.join(str(sol) for sol in solution)
                    return f"x = {solution}"
                return "No solution found"
            else:
                result = sympify(expression)
                return f"{result}" if isinstance(result, (int, float)) else str(result)
        except Exception as e:
            print(f"Evaluation error: {e}")
            return "Invalid expression"

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger = hand_landmarks.landmark[8]
                x, y = int(index_finger.x * self.width), int(index_finger.y * self.height)
                if self.get_finger_state(hand_landmarks):
                    cv2.circle(self.canvas, (x, y), 5, (255, 255, 255), -1)
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        combined = cv2.addWeighted(frame, 0.5, self.canvas, 0.5, 0)
        if self.expression:
            cv2.putText(combined, f"Expression: {self.expression}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if self.result:
            cv2.putText(combined, f"Result: {self.result}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return combined

    def get_finger_state(self, hand_landmarks):
        """Detect if index finger is raised"""
        index_tip, index_pip = hand_landmarks.landmark[8].y, hand_landmarks.landmark[7].y
        return (index_pip - index_tip) > 0.02

    def run(self):
        print("Controls: - Raise index finger to draw, 'r' to recognize, 'e' to evaluate, 'c' to clear, 'q' to quit")
        while True:
            combined_frame = self.process_frame()
            if combined_frame is not None:
                cv2.imshow("Calculator", combined_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                self.expression, self.result = "", ""
            elif key == ord('r'):
                self.expression = self.recognize_expression()
            elif key == ord('e') and self.expression:
                self.result = self.evaluate_expression(self.expression)

        self.cap.release()
        cv2.destroyAllWindows()

# Instantiate and run the calculator
calculator = LocalAirCalculator()
calculator.run()
