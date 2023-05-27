import cv2
import mediapipe
import pyautogui


cam = cv2.VideoCapture(0)
faceMesh = mediapipe.solutions.face_mesh.FaceMesh(refine_landmarks=True)
displayH, displayW = pyautogui.size()


while True:
    sucess, image = cam.read()
    image = cv2.flip(image, 1)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = faceMesh.process(imageRGB)
    landmarksA = output.multi_face_landmarks
    imageH, imageW, channel = image.shape

    if (landmarksA):
        landmarks = landmarksA[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x, y = landmark.x * imageW, landmark.y * imageH
            cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), 1)  # img, center, radius, color, thickness
            if (id == 1):
                displayX = displayW / imageW * x
                displayY = displayH / imageH * y
                pyautogui.moveTo(x, y)

        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x, y = landmark.x * imageW, landmark.y * imageH
            cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), 1)  # img, center, radius, color, thickness

        if ((left[0].y - left[1].y) < 0.003):
            pyautogui.click()
            pyautogui.sleep(0.25)

    cv2.imshow("visionMouse (CPU) Preview", image)
    cv2.waitKey(1)
