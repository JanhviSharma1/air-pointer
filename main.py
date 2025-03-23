import cv2
import pyautogui
import mediapipe as mp

capture = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

while True:
    success, img = capture.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_height, img_width, _ = img.shape

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = hands.process(rgb_img)

    if output.multi_hand_landmarks:
        for hand in output.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

            index_finger_tip = hand.landmark[8]
            middle_finger_tip = hand.landmark[12]
            thumb_tip = hand.landmark[4]

            x = int(index_finger_tip.x * screen_width)
            y = int(index_finger_tip.y * screen_height)

            pyautogui.moveTo(x, y, duration=0.1)

            index_x, index_y = int(index_finger_tip.x * img_width), int(index_finger_tip.y * img_height)
            middle_x, middle_y = int(middle_finger_tip.x * img_width), int(middle_finger_tip.y * img_height)
            thumb_x, thumb_y = int(thumb_tip.x * img_width), int(thumb_tip.y * img_height)

            # Left Click: If index & middle fingers are close together
            if abs(index_x - middle_x) < 40 and abs(index_y - middle_y) < 40:
                pyautogui.click()

            # Right Click: If index & thumb are close together
            if abs(index_x - thumb_x) < 40 and abs(index_y - thumb_y) < 40:
                pyautogui.rightClick()

    cv2.imshow("Hand Mouse Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()