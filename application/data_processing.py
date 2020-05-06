from data_loader.mpiigaze_processed_loader import _prepare_images, _prepare_headposes


def convert_to_model_format(data: dict, grayscale=False):
    right_image = _prepare_images(data["right_image"], grayscale)
    left_image = _prepare_images(data["left_image"], grayscale)
    pose = _prepare_headposes(data["pose"], normalize=False)

    return right_image, left_image, pose