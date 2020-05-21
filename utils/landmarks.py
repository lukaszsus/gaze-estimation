import cv2
from matplotlib import pyplot as plt


def visualize_faces(faces: list, image, color=(255, 255, 255), show=True):
    """
    Shows a photo with face rectangles.
    :param faces: List of lists - one list contains 4 numbers which are upper-left and down-right corners of face
                bounding box.
    :param image: Image to draw bounding box on it.
    :param color: Color to draw the bounding box.
    """
    # Print coordinates of detected faces
    if show:
        print("Faces:\n", faces)

    for face in faces:
        # save the coordinates in x, y, w, d variables
        (x, y, w, d) = face
        # Draw a white coloured rectangle around each face using the face's coordinates
        # on the "image_template" with the thickness of 2
        cv2.rectangle(image, (x, y), (x + w, y + d), color, 2)

    if show:
        plt.axis("off")
        plt.imshow(image)
        plt.show()
        plt.title('Face Detection')


def visualize_landmarks(landmarks: list, image, numbers=False, color=(255, 255, 255)):
    for landmark in landmarks:
        for i, (x, y) in enumerate(landmark):
            if numbers:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, str(i), (x, y), fontFace=font, fontScale=1, color=(255, 255, 255), lineType=cv2.LINE_AA)
            else:
                cv2.circle(image, (x, y), 2, color, 2)
    plt.axis("off")
    plt.imshow(image)
    plt.show()


def visualize_landmarks_mpii_gaze_format(landmarks: list, image, numbers=False):
    for landmark in landmarks:
        for i, (x, y) in enumerate(landmark):
            if numbers:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, str(i), (x, y), fontFace=font, fontScale=1, color=(255, 255, 255), lineType=cv2.LINE_AA)
            else:
                cv2.circle(image, (x, y), 2, (255, 255, 255), 2)
    plt.axis("off")
    plt.imshow(image)
    plt.show()