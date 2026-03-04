#!/usr/bin/env python3

from dataclasses import dataclass
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import enum
import mediapipe as mp
import numpy as np
import praxis.log as log
import utils

FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)
MARGIN = 10

CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720

OSC_MIN_FREQ = 82
OSC_MAX_FREQ = 400
LFO_MIN_FREQ = 1
LFO_MAX_FREQ = 3


class HandType(enum.Enum):
    LEFT = 'left'
    RIGHT = 'right'


@dataclass
class Point:
    x: int
    y: int


class Hand:
    def __init__(self, type: HandType, point: Point | None = None):
        self.type = type
        self.point = point


def get_index(detection_result, image_width: int, image_height: int) -> dict[HandType, Point | None]:
    '''
    Retorna la posición en píxeles de la punta del índice para ambas manos.

    Args:
        detection_result: HandLandmarkerResult de MediaPipe
        image_width:      ancho del frame en píxeles
        image_height:     alto del frame en píxeles

    Returns:
        dict con HandType.LEFT y HandType.RIGHT como claves.
        Valor Point(x, y) en píxeles, o None si la mano no fue detectada.
    '''

    result: dict[HandType, Point | None] = {
        HandType.LEFT: None,
        HandType.RIGHT: None,
    }

    for i, handedness_list in enumerate(detection_result.handedness):
        label = handedness_list[0].category_name  # 'Left' o 'Right'
        hand_type = HandType(label.lower())

        landmark = detection_result.hand_landmarks[i][8]  # INDEX_FINGER_TIP
        result[hand_type] = Point(
            x=int(landmark.x * image_width),
            y=int(landmark.y * image_height),
        )

    return result


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
                'lfo' if handedness[0].category_name.lower() == HandType.LEFT.value else 'osc',
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                HANDEDNESS_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    def draw_index(self, rgb_image: mp.Image, positions):
        annotated_image = np.copy(rgb_image)

        for hand_type, point in positions.items():
            if point is None:
                continue

            cv2.circle(
                annotated_image,
                center=(point.x, point.y),
                radius=10,
                color=(0, 255, 0),
                thickness=-1,
            )

            cv2.putText(
                annotated_image,
                hand_type.value,
                (point.x + 12, point.y),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                HANDEDNESS_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return annotated_image


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

    osc1 = utils.Oscillator(440, 1, utils.Waveform.SQUARE)
    osc2 = utils.Oscillator(440 * 6 / 5, 1, utils.Waveform.SAWTOOTH)
    osc3 = utils.Oscillator(440 * 3 / 2, 1, utils.Waveform.SAWTOOTH)
    lfo = utils.Oscillator(2, 1, utils.Waveform.SINE)

    osc1.set_amplitude(lfo)
    osc2.set_amplitude(lfo)
    osc3.set_amplitude(lfo)

    osc1.play()
    osc2.play()
    lfo.play()

    if not cap.isOpened():
        log.error('No se pudo acceder a la cámara.')
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            log.error('No se pudo leer el frame.')
            break

        image: mp.Image = cv_image_to_mp_image(frame)
        detection_result = detector.detect(image)
        index_pos = get_index(detection_result, CANVAS_WIDTH, CANVAS_HEIGHT)
        # annotated_image = hands_drawer.draw_index(image.numpy_view(), index_pos)
        annotated_image = hands_drawer.draw_landmarks_on_image(image.numpy_view(), detection_result)

        if annotated_image is not None:
            frame = annotated_image

        cv2.imshow('img', frame)

        for hand_type, point in index_pos.items():
            if point is None:
                continue

            if hand_type == HandType.LEFT:
                lfo.set_frequency(utils.map_range(point.y, 0, CANVAS_HEIGHT, LFO_MIN_FREQ, LFO_MAX_FREQ))
            elif hand_type == HandType.RIGHT:
                osc1.set_frequency(utils.map_range(point.y, 0, CANVAS_HEIGHT, OSC_MIN_FREQ, OSC_MAX_FREQ))
                osc2.set_frequency(osc1.frequency * 6 / 5)
                osc3.set_frequency(osc1.frequency * 3 / 2)

        if check_exit():
            break

    # Liberar recursos
    osc1.stop()
    osc2.stop()
    osc3.stop()
    cap.release()
    cv2.destroyAllWindows()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
