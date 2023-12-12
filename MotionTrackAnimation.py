# Initialize Mediapipe components for Pose, Face, and Hands
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize Mediapipe models
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face.FaceDetection(min_detection_confidence=0.3) as face_detection, \
     mp_hands.Hands(min_detection_confidence=0.7) as hands:

    # Variables for animation
    animation_started = False
    animation_landmarks = []

    # Load the PNG image of the head
    head_image = cv2.imread('head1.png', cv2.IMREAD_UNCHANGED)

    # Resize the head image to match the video frame size
    head_height, head_width, _ = head_image.shape
    resize_factor = 0.1  # Adjust the factor based on your preference
    head_resized = cv2.resize(head_image, (int(cap.get(3) * resize_factor), int(cap.get(4) * resize_factor)))

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('animation_output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get the pose landmarks
        pose_results = pose.process(rgb_frame)

        # Draw the pose landmarks on the frame and add to animation_landmarks list
        if pose_results.pose_landmarks:
            if animation_started:
                animation_landmarks.append(pose_results.pose_landmarks)

        # Display the output
        cv2.imshow('MediaPipe Pose, Face, Hands', frame)

        # Check for keypress to start/stop animation
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            animation_started = True
            animation_landmarks = []
        elif key == ord('e'):
            animation_started = False
            black_frame = np.zeros_like(frame)

            for landmarks in animation_landmarks:
                frame_copy = black_frame.copy()
                mp_drawing.draw_landmarks(frame_copy, landmarks, mp_pose.POSE_CONNECTIONS)

                # Get the midpoint between the eyes and lips landmarks
                eyes_midpoint = (landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].x +
                                 landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].x) / 2, \
                                (landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].y +
                                 landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].y) / 2

                lips_midpoint = (landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].x +
                                 landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].x) / 2, \
                                (landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].y +
                                 landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].y) / 2

                # Calculate the midpoint between eyes and lips
                face_midpoint = ((eyes_midpoint[0] + lips_midpoint[0]) / 2, (eyes_midpoint[1] + lips_midpoint[1]) / 2)

                # Calculate the position for placing the face
                face_left = int(face_midpoint[0] * frame_copy.shape[1]) - head_resized.shape[1] // 2
                face_top = int(face_midpoint[1] * frame_copy.shape[0]) - head_resized.shape[0] // 2
                face_right = face_left + head_resized.shape[1]
                face_bottom = face_top + head_resized.shape[0]

                # Ensure that the face coordinates are within the frame bounds
                face_left = max(0, face_left)
                face_top = max(0, face_top)
                face_right = min(frame_copy.shape[1], face_right)
                face_bottom = min(frame_copy.shape[0], face_bottom)

                # Get the alpha channel from the resized head image
                head_alpha = head_resized[:, :, 3] / 255.0

                # Blend the head image with the animation frame in the calculated position
                for c in range(0, 3):
                    frame_copy[face_top:face_bottom, face_left:face_right, c] = \
                        frame_copy[face_top:face_bottom, face_left:face_right, c] * (1 - head_alpha) + \
                        head_resized[:, :, c] * head_alpha

                out.write(frame_copy)
                cv2.imshow('Animation', frame_copy)
                cv2.waitKey(30)

        # Break the loop when 'q' is pressed
        elif key == ord('q'):
            break

    # Release the VideoWriter object
    out.release()

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()