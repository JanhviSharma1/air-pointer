import cv2
import pyautogui
import mediapipe as mp

capture = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

while True:
    success, img = capture.read()
    if not success:
        break
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = hands.process(rgb_img)
    if output.multi_hand_landmarks:
        for hand in output.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

added landmarks