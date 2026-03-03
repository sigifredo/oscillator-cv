#!/usr/bin/env python3

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import mediapipe as mp
import numpy as np
import praxis.log as log


FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)
MARGIN = 10


class HandsDrawer:
    def __init__(self):
        self.mp_hands = mp.tasks.vision.HandLandmarksConnections
        self.mp_drawing = mp.tasks.vision.drawing_utils
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            self.mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(
                annotated_image,
                f'{handedness[0].category_name}',
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                HANDEDNESS_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return annotated_image


def cv_image_to_mp_image(frame: cv2.Mat) -> mp.Image:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)


def main():
    hands_drawer: HandsDrawer = HandsDrawer()

    base_options = python.BaseOptions(model_asset_path='assets/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        log.error('No se pudo acceder a la cámara.')
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()

        if not ret:
            log.error('No se pudo leer el frame.')
            break

        image: mp.Image = cv_image_to_mp_image(frame)
        detection_result = detector.detect(image)
        annotated_image = hands_drawer.draw_landmarks_on_image(image.numpy_view(), detection_result)

        cv2.imshow('img', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
