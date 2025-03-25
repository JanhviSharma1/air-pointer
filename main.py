import cv2
import pyautogui
import mediapipe as mp
import numpy as np

capture = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

prev_y = None  # For scrolling detection

while True:
    success, img = capture.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Mirror effect
    img_height, img_width, _ = img.shape

    # Convert image to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = hands.process(rgb_img)

    if output.multi_hand_landmarks:
        for hand in output.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

            # Get important finger landmarks
            index_tip = hand.landmark[8]
            middle_tip = hand.landmark[12]
            thumb_tip = hand.landmark[4]
            ring_tip = hand.landmark[16]
            pinky_tip = hand.landmark[20]
            wrist = hand.landmark[0]

            # Convert coordinates to screen size
            x = int(index_tip.x * screen_width)
            y = int(index_tip.y * screen_height)

            # Move mouse with index finger
            pyautogui.moveTo(x, y, duration=0.1)

            # Convert to image coordinates (for distance check)
            index_x, index_y = int(index_tip.x * img_width), int(index_tip.y * img_height)
            middle_x, middle_y = int(middle_tip.x * img_width), int(middle_tip.y * img_height)
            thumb_x, thumb_y = int(thumb_tip.x * img_width), int(thumb_tip.y * img_height)
            ring_x, ring_y = int(ring_tip.x * img_width), int(ring_tip.y * img_height)
            pinky_x, pinky_y = int(pinky_tip.x * img_width), int(pinky_tip.y * img_height)
            wrist_x, wrist_y = int(wrist.x * img_width), int(wrist.y * img_height)

            # Left Click: If index & thumb are touching
            if abs(index_x - thumb_x) < 40 and abs(index_y - thumb_y) < 40:
                pyautogui.click()

            # Right Click: If index & middle fingers are touching
            if abs(index_x - middle_x) < 40 and abs(index_y - middle_y) < 40:
                pyautogui.rightClick()

            # Screenshot: If full hand is open (All fingers spread apart)
            spread_threshold = 50  # Adjust based on hand size
            if (
                abs(index_x - thumb_x) > spread_threshold and
                abs(middle_x - index_x) > spread_threshold and
                abs(ring_x - middle_x) > spread_threshold and
                abs(pinky_x - ring_x) > spread_threshold
            ):
                pyautogui.screenshot("screenshot.png")
                print("Screenshot Taken!")

            # Scroll Up: If thumb is pointing up (higher than index & middle)
            if thumb_y < index_y and thumb_y < middle_y:
                pyautogui.scroll(10)

            # Scroll Down: If thumb is pointing sideways (Near wrist level)
            if abs(thumb_x - wrist_x) < 40 and abs(thumb_y - wrist_y) < 40:
                pyautogui.scroll(-10)

    # Show camera output
    cv2.imshow("Hand Mouse Control", img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
capture.release()
cv2.destroyAllWindows()