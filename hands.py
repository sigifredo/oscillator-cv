#!/usr/bin/env python3

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import mediapipe as mp
import numpy as np
import praxis.log as log
import utils

FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)
MARGIN = 10


def get_right_index_tip(detection_result, image_width: int, image_height: int) -> tuple[int, int] | None:
    '''
    Retorna la posición en píxeles de la punta del índice derecho.

    Args:
        detection_result: HandLandmarkerResult de MediaPipe
        image_width:  ancho del frame en píxeles
        image_height: alto del frame en píxeles

    Returns:
        (x, y) en píxeles, o None si no se detecta mano derecha
    '''

    for i, handedness_list in enumerate(detection_result.handedness):
        # Cada elemento es una lista con un Category; tomamos el de mayor score
        hand_label = handedness_list[0].category_name  # 'Right' o 'Left'

        if hand_label == 'Left':
            landmark = detection_result.hand_landmarks[i][8]  # INDEX_FINGER_TIP

            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)

            return (x, y)

    return None


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

    def draw_index(self, rgb_image: mp.Image, detection_result):
        tip_pos = get_right_index_tip(detection_result, 1280, 720)

        if tip_pos:
            annotated_image = np.copy(rgb_image)

            cv2.circle(annotated_image, center=tip_pos, radius=10, color=(0, 255, 0), thickness=-1)

            cv2.putText(
                annotated_image,
                "index",
                (tip_pos[0] + 12, tip_pos[1]),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                HANDEDNESS_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

            return annotated_image, tip_pos

        return rgb_image, None


def check_exit() -> bool:
    key = cv2.waitKey(1)
    return key == ord('q') or key == 27


def cv_image_to_mp_image(frame: cv2.Mat) -> mp.Image:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)


def main():
    hands_drawer: HandsDrawer = HandsDrawer()

    base_options = python.BaseOptions(model_asset_path='assets/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(1)

    osc = utils.Oscillator(440, 1, 'sawtooth')
    lfo = utils.Oscillator(2, 1)

    osc.play()
    lfo.play()

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
        annotated_image, index_pos = hands_drawer.draw_index(image.numpy_view(), detection_result)
        # annotated_image = hands_drawer.draw_landmarks_on_image(image.numpy_view(), detection_result)

        if annotated_image is not None:
            cv2.imshow('img', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        else:
            cv2.imshow('img', frame)

        if index_pos:
            # osc.set_frequency(utils.map_range(index_pos[1], 0, 720, 100, 10000))
            lfo.set_frequency(utils.map_range(index_pos[1], 0, 720, 0, 10))

        if check_exit():
            break

    # Liberar recursos
    osc.stop()
    cap.release()
    cv2.destroyAllWindows()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
