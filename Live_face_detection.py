import cv2

frontal_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror image

    # face detection is good in grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # flip the grayscale image for grayscale
    gray_flipped = cv2.flip(gray, 1)

    # frontal faces
    faces = frontal_face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7, minSize=(50, 50))

    # left profile faces 
    left_profile_faces = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
    
    # right profile faces 
    right_profile_faces = profile_face_cascade.detectMultiScale(gray_flipped, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
    
    # Convert right profile coordinates back to original image coordinates
    corrected_right_profile_faces = []
    for (x, y, w, h) in right_profile_faces:
        # Adjust for the flipped image
        corrected_x = frame.shape[1] - x - w
        corrected_right_profile_faces.append((corrected_x, y, w, h))

    # frontal faces rectangles(Green)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # left profile faces rectangles(Blue)
    for (x, y, w, h) in left_profile_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    # right profile faces rectangles(Red)
    for (x, y, w, h) in corrected_right_profile_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # text
    cv2.putText(frame, f"Frontal: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Left profile: {len(left_profile_faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Right profile: {len(corrected_right_profile_faces)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Live Face Detection", frame)

    # q to exit 
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()