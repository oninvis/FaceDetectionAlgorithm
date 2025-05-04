import json
import mediapipe as mp
import cv2
import numpy as np
from deepface import DeepFace


class FaceDetector:
    def __init__(self, mode, num_faces, ref_lm, detectionCon, trackingCon):
        self.mode = mode
        self.num_faces = num_faces
        self.ref_lm = ref_lm
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.mpFace = mp.solutions.face_mesh
        self.face = self.mpFace.FaceMesh(
            static_image_mode=mode,
            max_num_faces=num_faces,
            refine_landmarks=ref_lm,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackingCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def storeFace(self):
        cap = cv2.VideoCapture(0)
        while True:
            success, img = cap.read()
            if not success:
                print("Failed to capture image")
                break
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.face.process(imgRGB)
            if self.results.multi_face_landmarks:
                for face_landmarks in self.results.multi_face_landmarks:
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        landmarks = []
                        for landmark in face_landmarks.landmark:
                            x, y, z = landmark.x, landmark.y, landmark.z
                            landmarks.append({
                                'x': x,
                                'y': y,
                                'z': z
                            })

                        with open('AuthorisedFace.json', 'w') as f:
                            json.dump(landmarks, f)
                        print('Face mesh saved')
                        break
            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        return landmarks

    # Euclidean calculation to calculate distance
    def computeFace(self, landmarks1, landmarks2):
        if len(landmarks1) != len(landmarks2):
            return float("inf")
        total_distance = 0
        for lm1, lm2 in zip(landmarks1, landmarks2):
            dist = np.sqrt(
                (lm1['x'] - lm2['x']) ** 2 +
                (lm1['y'] - lm2['y']) ** 2 +
                (lm1['z'] - lm2['z']) ** 2
            )
            total_distance += dist
        return total_distance / len(landmarks1)

    def detectEmotion(self, img, face_landmarks):
        """
        Detect the emotion of the person in the given image using the detected face landmarks.

        Args:
            img: The current video frame (BGR format).
            face_landmarks: The detected face landmarks from Mediapipe.

        Returns:
            str: The dominant emotion (e.g., 'happy', 'sad', 'angry') or 'Unknown' if detection fails.
        """
        try:
            # Get image dimensions
            h, w, _ = img.shape
            # Convert landmarks to pixel coordinates
            landmarks = []
            for landmark in face_landmarks.landmark:
                x, y = landmark.x, landmark.y
                landmarks.append((int(x * w), int(y * h)))

            # Calculate bounding box for the face
            x_coords = [lm[0] for lm in landmarks]
            y_coords = [lm[1] for lm in landmarks]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Add padding to the bounding box
            padding = 20
            x_min = max(0, x_min - padding)
            x_max = min(w, x_max + padding)
            y_min = max(0, y_min - padding)
            y_max = min(h, y_max + padding)

            # Crop the face from the image
            face_img = img[y_min:y_max, x_min:x_max]

            # Analyze the cropped face for emotion using DeepFace
            emotion_result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            if isinstance(emotion_result, list) and len(emotion_result) > 0:
                emotion = emotion_result[0]['dominant_emotion']
            else:
                emotion = emotion_result['dominant_emotion']
            return emotion
        except Exception as e:
            print(f"Emotion detection failed: {e}")
            return "Unknown"
