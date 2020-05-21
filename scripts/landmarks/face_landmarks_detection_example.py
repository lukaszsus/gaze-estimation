import cv2
import urllib.request as urlreq
import os
import matplotlib.pyplot as plt

from models.face_landmarks_detectors import get_haarcascade_detector, get_lbf_model
from settings import DATA_PATH
from utils.landmarks import visualize_faces, visualize_landmarks


def load_example_image():
    # save picture's url in pics_url variable
    pics_url = "https://upload.wikimedia.org/wikipedia/commons/archive/7/76/20170623195932%21Donald_Trump_Justin_Trudeau_2017-02-13_02.jpg"

    # save picture's name as pic
    pic_dir = os.path.join(DATA_PATH, "for_tests")
    os.makedirs(pic_dir, exist_ok=True)
    pic_path = os.path.join(pic_dir, "landmarks_image.jpg")
    print(pic_path)

    # download picture from url and save locally as image.jpg
    urlreq.urlretrieve(pics_url, pic_path)
    # read image with openCV
    image = cv2.imread(pic_path)

    return image


if __name__ == "__main__":
    ### Download the image
    image = load_example_image()

    # plot image with matplotlib package
    plt.imshow(image)
    plt.show()

    ### Image processing
    # convert image to RGB colour
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    image_cropped = image_rgb
    # # set dimension for cropping image
    # x, y, width, depth = 50, 200, 950, 500
    # image_cropped = image_rgb[y:(y + depth), x:(x + width)]
    # create a copy of the cropped image to be used later
    image_template = image_cropped.copy()
    # convert image to Grayscale
    image_gray = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR)
    # remove axes and show image
    plt.axis("off")
    plt.imshow(image_gray, cmap="gray")
    plt.show()

    ########################################################################################################
    detector = get_haarcascade_detector()
    landmark_detector = get_lbf_model()

    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = detector.detectMultiScale(image_gray)
    visualize_faces(faces, image_template)

    # Detect landmarks on "image_gray"
    _, landmarks = landmark_detector.fit(image_gray, faces)
    visualize_landmarks(landmarks, image_cropped)

    print(landmarks)