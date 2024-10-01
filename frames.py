import cv2
import os

videos_directory = '/home/cortezm/Desktop/Project IA/Real-life_Deception_Detection_2016/Clips/Deceptive'
output_directory = '/home/cortezm/Desktop/Project IA/Real-life_Deception_Detection_2016/Clips/DeceptiveFR'

# Output folder
os.makedirs(output_directory, exist_ok=True)

# Load pre-trained classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

frame_rate = 5
segment_duration = 5  # seconds

for video_file in os.listdir(videos_directory):
    if video_file.endswith('.mp4'):  # Only videos
        video_path = os.path.join(videos_directory, video_file)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        segment_count = 0

        video_output_directory = os.path.join(output_directory, os.path.splitext(video_file)[0])
        os.makedirs(video_output_directory, exist_ok=True)

        while cap.isOpened():
            # Frame per frame
            ret, frame = cap.read()

            if ret:
                if frame_count % frame_rate == 0:
                    # to Gray Scale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Detect faces
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                    for (x, y, w, h) in faces:
                        # Only the face
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_color = frame[y:y+h, x:x+w]

                        # Save frame
                        output_path = os.path.join(video_output_directory, f"frame_{frame_count}.jpg")
                        cv2.imwrite(output_path, roi_color)

                frame_count += 1

                if frame_count % (frame_rate * segment_duration) == 0:
                    segment_count += 1
            else:
                break
        cap.release()

cv2.destroyAllWindows()
