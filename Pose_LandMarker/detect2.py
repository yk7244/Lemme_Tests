import argparse
import sys
import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None

def run(model: str, num_poses: int,
        min_pose_detection_confidence: float,
        min_pose_presence_confidence: float, min_tracking_confidence: float,
        output_segmentation_masks: bool,
        camera_id: int, width: int, height: int) -> None:
    """Continuously run inference on images acquired from the camera.
    Args:
        model: Name of the pose landmarker model bundle.
        num_poses: Max number of poses that can be detected by the landmarker.
        min_pose_detection_confidence: The minimum confidence score for pose
          detection to be considered successful.
        min_pose_presence_confidence: The minimum confidence score of pose
          presence score in the pose landmark detection.
        min_tracking_confidence: The minimum confidence score for the pose
          tracking to be considered successful.
        output_segmentation_masks: Choose whether to output segmentation masks.
        camera_id: The ID of the camera to use for input.
        width: The width of the frame.
        height: The height of the frame.
    """
    global COUNTER, FPS, START_TIME, DETECTION_RESULT

    # Initialize the pose detector
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=output_segmentation_masks,
        min_detection_confidence=min_pose_detection_confidence,
        min_tracking_confidence=min_tracking_confidence) as pose:

        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a later selfie-view display
            # Also convert the BGR image to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable
            image.flags.writeable = False
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                # Draw the coordinates of each joint
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.putText(image, f'{idx}: ({x}, {y})', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

            # Calculate FPS
            COUNTER += 1
            if (time.time() - START_TIME) > 1:
                FPS = COUNTER / (time.time() - START_TIME)
                COUNTER = 0
                START_TIME = time.time()

            # Show the FPS
            cv2.putText(image, f'FPS: {FPS:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the resulting frame
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        help='Name of the pose landmarker model bundle.',
        required=False,
        default='pose_landmarker_full.task')
    parser.add_argument(
        '--numPoses',
        help='Max number of poses that can be detected by the landmarker.',
        required=False,
        default=1)
    parser.add_argument(
        '--minPoseDetectionConfidence',
        help='The minimum confidence score for pose detection to be considered successful.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minPosePresenceConfidence',
        help='The minimum confidence score of pose presence score in the pose landmark detection.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minTrackingConfidence',
        help='The minimum confidence score for the pose tracking to be considered successful.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--outputSegmentationMasks',
        help='Set this if you would also like to visualize the segmentation mask.',
        required=False,
        action='store_true')
    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        default=1280)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        default=960)
    args = parser.parse_args()

    run(args.model, int(args.numPoses), float(args.minPoseDetectionConfidence),
        float(args.minPosePresenceConfidence), float(args.minTrackingConfidence),
        args.outputSegmentationMasks,
        int(args.cameraId), int(args.frameWidth), int(args.frameHeight))

if __name__ == '__main__':
    main()

