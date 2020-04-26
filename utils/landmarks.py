import cv2
from matplotlib import pyplot as plt


def visualize_faces(faces: list, image):
    """
    Shows a photo with face rectangles.
    :param faces: List of lists - one list contains 4 numbers which are upper-left and down-right corners of face
                bounding box.
    :param image: Image to draw bounding box on it.
    """
    # Print coordinates of detected faces
    print("Faces:\n", faces)

    for face in faces:
        # save the coordinates in x, y, w, d variables
        (x, y, w, d) = face
        # Draw a white coloured rectangle around each face using the face's coordinates
        # on the "image_template" with the thickness of 2
        cv2.rectangle(image, (x, y), (x + w, y + d), (255, 255, 255), 2)

    plt.axis("off")
    plt.imshow(image)
    plt.show()
    plt.title('Face Detection')


def visualize_landmarks(landmarks: list, image):
    for landmark in landmarks:
        for x, y in landmark[0]:
            # display landmarks on "image_cropped"
            # with white colour in BGR and thickness 1
            cv2.circle(image, (x, y), 1, (255, 255, 255), 1)
    plt.axis("off")
    plt.imshow(image)
    plt.show()