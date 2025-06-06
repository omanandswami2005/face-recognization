import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between the horizontal eye landmarks
    C = distance.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    return (A + B) / (2.0 * C)

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start video capture
cap = cv2.VideoCapture(0)
EYE_AR_THRESH = 0.2  # Threshold below which the eye is considered closed
blink_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        shape = predictor(gray, face)
        # Convert the shape (facial landmarks) to a NumPy array
        shape_np = face_utils.shape_to_np(shape)  # You'll need imutils or a similar utility for this conversion
        
        # Assume left eye is landmarks [42:48] and right eye is [36:42]
        leftEye = shape_np[42:48]
        rightEye = shape_np[36:42]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < EYE_AR_THRESH:
            blink_detected = True
        else:
            blink_detected = False

        # Draw eye contours and display EAR for debug purposes
        # (Drawing code omitted for brevity)

    cv2.putText(frame, f"Blink Detected: {blink_detected}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Liveness Check", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()