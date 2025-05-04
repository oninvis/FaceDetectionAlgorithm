import json
import cv2
import FaceAnalysis as fm
from PI import unlock_door
import  time
#from playsound import playsound
# creating a face detector instance from the face mesh module
faceDetector = fm.FaceDetector(False, 1, False, 0.5, 0.5)
mpDraw = faceDetector.mpDraw
mpFace = faceDetector.mpFace

# drawing specs for the points and the connections
pointSpec = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
connectorSpec = mpDraw.DrawingSpec(color=(255, 255, 255), thickness=1)
instruction = input(str('''1) Press S to save a new face
2) Press U to unlock
                        #'''))
state = False
last_denied_time = 0
granted_sound_played = False
face = mpFace.FaceMesh()
if instruction.upper() == 'S':
    stored_landmarks = faceDetector.storeFace()
    print(stored_landmarks)
elif instruction.upper() == 'U':
    try:
        with open('AuthorisedFace.json', 'r') as f:
            stored_landmarks = json.load(f)
        print(f"stored landmarks successfully in stored_landmarks {stored_landmarks}")
    except FileNotFoundError:
        print("Error: file not found")
    except json.JSONDecodeError:
        print("Error : please save a face")
        exit()
    video_path = 'angry_face1.mp4'
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = face.process(imgRGB)
        if results.multi_face_landmarks:
            live_landmarks = []
            for face_landmark in results.multi_face_landmarks:
                mpDraw.draw_landmarks(
                    img,
                    face_landmark,
                    mpFace.FACEMESH_TESSELATION,
                    pointSpec,
                    connectorSpec
                )
                emotion = faceDetector.detectEmotion(img, face_landmark)
                # Display emotion on the video feed
                cv2.putText(img, f"Emotion: {emotion}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                for landmark in face_landmark.landmark:
                    x, y, z = landmark.x, landmark.y, landmark.z
                    live_landmarks.append({
                        'x': x,
                        'y': y,
                        'z': z
                    })
                    #print(live_landmarks)
                    #print(stored_landmarks)
                    dist = faceDetector.computeFace(live_landmarks, stored_landmarks)
                    if dist < 0.2:
                        access = 'Access Granted'
                        state = True
                        print(unlock_door())
                        if not granted_sound_played:  # Play "Access Granted" only once
                            granted_sound_played = True
                        break
                    else:
                        access = 'Access Denied'
                        granted_sound_played = False
            cv2.putText(img, access, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if access == 'Access Granted' else (0, 0, 255), 2)
            if access == "Access Denied":
                current_time = time.time()
                if current_time - last_denied_time >= 5:
                # playsound('access_denied_sound.mp3')
                    last_denied_time = current_time
            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
