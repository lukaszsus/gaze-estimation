import cv2
import numpy as np

from application.data_processing import convert_to_model_format
from data_processing.head_pose import estimate_head_pose
from data_processing.algebraic_transformations import processed_both_eyes_rgb, parse_both_eyes_rgb_landmark
from models.face_detectors.face_detector import FaceDetector
from models.landmarks_detectors.landmarks_detector import LandmarksDetector, filter_landmarks, \
    MOUTH_EYES_CORNERS_HEAD_POSE, EYES_LANDMARKS
from models.model_loaders import load_best_modal3_conv_net
from scripts.create_dataset.create_dataset_mpiigaze_processed_both_from_single_eye import resultant_angles
from scripts.create_dataset.create_dataset_mpiigaze_processed_both_rgb import load_face_model, norm_landmarks

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

# USE_GPU = False  # just for logging some metrics correctly (for example forward_pass_time)
"""
That's for GPU training and maintaining one session and nice cuda lib loading.
"""
USE_GPU = True
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
AUTOTUNE = tf.data.experimental.AUTOTUNE


class Pipeline:
    def __init__(self, face_detector, landmarks_detector,
                 eye_image_width, eye_image_height, camera_matrix,
                 screen_size):
        self.face_model = load_face_model()
        self.face_detector: FaceDetector = face_detector
        self.landmarks_detector: LandmarksDetector = landmarks_detector
        self.eye_image_width = eye_image_width
        self.eye_image_height = eye_image_height
        self.camera_matrix = camera_matrix
        self.gaze_estimation_model = load_best_modal3_conv_net(test=False)
        self.screen_size = screen_size

        self.image = None
        self.im_height = None
        self.im_width = None

    def process(self, image):
        self.image = image
        self.im_height, self.im_width, _ = image.shape

        # grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # detect face
        faces = self.face_detector.detect(image_gray)

        # Take only first face
        if len(faces) > 1:
            faces = faces[0].reshape(1, -1)
        elif len(faces) == 0:  # No face detected -> assume that face is on the full image
            return None

        # detect landmarks
        self.landmarks = self.landmarks_detector.detect(image_gray, faces)

        # make algebraic magic
        data = self._algebraic_tranformations()

        return data

    def predict(self, image):
        """
        Return (y, x)
        y - shorter edge (up and down)
        x - longer edge (left and right)
        """
        self.image = image
        self.im_height, self.im_width, _ = image.shape

        # grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # detect face
        faces = self.face_detector.detect(image_gray)

        # Take only first face
        if len(faces) > 1:
            faces = faces[0].reshape(1, -1)
        elif len(faces) == 0:  # No face detected -> assume that face is on the full image
            return None

        # detect landmarks
        self.landmarks = self.landmarks_detector.detect(image_gray, faces)

        # make algebraic magic
        data = self._algebraic_tranformations()

        # make prediction
        prediction = self._predict_coordinates(data)

        # some post processing if needed
        prediction = prediction.numpy()
        prediction[:, 0] = prediction[:, 0] * self.screen_size[0]
        prediction[:, 1] = prediction[:, 1] * self.screen_size[1]
        prediction[prediction < 0] = 0.

        # return prediction
        return prediction

    def _algebraic_tranformations(self):
        landmarks_head_pose = filter_landmarks(self.landmarks, indices=MOUTH_EYES_CORNERS_HEAD_POSE)

        headpose_hr, headpose_ht = estimate_head_pose(self.image, landmarks_head_pose, self.camera_matrix,
                                                      face_model_points=np.transpose(self.face_model))

        data = parse_both_eyes_rgb_landmark(self.image, self.face_model,
                                            self.camera_matrix,
                                            headpose_hr, headpose_ht,
                                            eye_image_width=self.eye_image_width,
                                            eye_image_height=self.eye_image_height)

        # additional landmarks
        landmarks_eyes = filter_landmarks(self.landmarks, EYES_LANDMARKS)
        landmarks_eyes = np.reshape(landmarks_eyes, (1, -1)).astype(np.float)
        landmarks_eyes = norm_landmarks(landmarks_eyes, height=self.im_height, width=self.im_width)
        data["landmarks"] = landmarks_eyes.reshape(1, -1)

        # convert to model format
        data['right_image'] = data['right_image'].reshape(1, data['right_image'].shape[0],
                                                  data['right_image'].shape[1],
                                                  data['right_image'].shape[2])
        data['left_image'] = data['left_image'].reshape(1, data['left_image'].shape[0],
                                                data['left_image'].shape[1],
                                                data['left_image'].shape[2])
        data["pose"] = np.array(data["pose"]).reshape(1, -1)
        resultant_pose = resultant_angles(data["pose"][:, 0:2], data["pose"][:, 2:4])
        data["pose"] = np.concatenate([data["pose"], resultant_pose], axis=1)
        data["pose"] = np.concatenate([data["pose"], data["landmarks"]], axis=1)

        return data

    def _predict_coordinates(self, data):
        right_image, left_image, pose = convert_to_model_format(data)
        X = tf.data.Dataset.from_tensor_slices((right_image, left_image, pose))
        X = X.batch(batch_size=1)
        for x in X:     # only one iteration always
            prediction = self.gaze_estimation_model.predict(x)
        return prediction
