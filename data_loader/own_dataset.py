from sklearn.model_selection import train_test_split

from data_loader.mpiigaze_both_from_single import _load_all_people, _load_one_person, \
    _load_all_people_reject_suspicious, _filter_subject_02, _filter_subject_10
from data_loader.mpiigaze_processed_loader import prepare_dataset


def load_own_dataset_one_person(dataset_name, person_id, val_split=0.2, batch_size=128,
                                                 grayscale=True):
    test_subject_ids = None
    if person_id is None:
        right_images, left_images, poses, gazes, subject_ids = _load_all_people(dataset_name, grayscale)
        (right_eye_train, right_eye_test,
         left_eye_train, left_eye_test,
         headpose_train, headpose_test,
         y_train, y_test,
         train_subject_ids, test_subject_ids) = train_test_split(right_images, left_images, poses, gazes, subject_ids,
                                                                 test_size=val_split,
                                                                 random_state=42, stratify=subject_ids)
    else:
        right_images, left_images, poses, gazes = _load_one_person(dataset_name, person_id, grayscale)
        (right_eye_train, right_eye_test,
         left_eye_train, left_eye_test,
         headpose_train, headpose_test,
         y_train, y_test) = train_test_split(right_images, left_images, poses, gazes, test_size=val_split,
                                             random_state=42)

    train_dataset = prepare_dataset((right_eye_train, left_eye_train, headpose_train), y_train, batch_size)
    # don't shuffle test dataset to have consistent outcomes
    test_dataset = prepare_dataset((right_eye_test, left_eye_test, headpose_test), y_test, batch_size, shuffle=False)

    return train_dataset, test_dataset, test_subject_ids


def load_train_test_ds_reject_suspicious(dataset_name, person_id, val_split=0.2, batch_size=128,
                                                 grayscale=True, all_subjects=None):
    if all_subjects is None:
        all_subjects = list(range(0, 7)) + list(range(8, 15)) + [24, 25]

    test_subject_ids = None
    if person_id is None:
        right_images, left_images, poses, gazes, subject_ids = _load_all_people_reject_suspicious(dataset_name,
                                                                                                  grayscale,
                                                                                                  subjects=all_subjects)
        (right_eye_train, right_eye_test,
         left_eye_train, left_eye_test,
         headpose_train, headpose_test,
         y_train, y_test,
         train_subject_ids, test_subject_ids) = train_test_split(right_images, left_images, poses, gazes, subject_ids,
                                                                 test_size=val_split,
                                                                 random_state=42, stratify=subject_ids)
    else:
        right_images, left_images, poses, gazes = _load_one_person(dataset_name, person_id, grayscale)
        (right_eye_train, right_eye_test,
         left_eye_train, left_eye_test,
         headpose_train, headpose_test,
         y_train, y_test) = train_test_split(right_images, left_images, poses, gazes, test_size=val_split,
                                             random_state=42)
        # filter suspicious
        if person_id == 7:
            raise ValueError("Person 7 is rejected.")
        right_images, left_images, poses, gazes = _load_one_person(dataset_name, i, grayscale)
        if person_id == 2:
            right_images, left_images, poses, gazes = _filter_subject_02(right_images, left_images, poses, gazes)
        if person_id == 10:
            right_images, left_images, poses, gazes = _filter_subject_10(right_images, left_images, poses, gazes)

    train_dataset = prepare_dataset((right_eye_train, left_eye_train, headpose_train), y_train, batch_size)
    # don't shuffle test dataset to have consistent outcomes
    test_dataset = prepare_dataset((right_eye_test, left_eye_test, headpose_test), y_test, batch_size, shuffle=False)

    return train_dataset, test_dataset, test_subject_ids


def load_train_test_ds_reject_suspicious(dataset_name, person_id, val_split=0.2, batch_size=128,
                                                 grayscale=True, all_subjects=None):
    if all_subjects is None:
        all_subjects = list(range(0, 7)) + list(range(8, 15)) + [24, 25]

    test_subject_ids = None
    if person_id is None:
        right_images, left_images, poses, gazes, subject_ids = _load_all_people_reject_suspicious(dataset_name,
                                                                                                  grayscale,
                                                                                                  subjects=all_subjects)
        (right_eye_train, right_eye_test,
         left_eye_train, left_eye_test,
         headpose_train, headpose_test,
         y_train, y_test,
         train_subject_ids, test_subject_ids) = train_test_split(right_images, left_images, poses, gazes, subject_ids,
                                                                 test_size=val_split,
                                                                 random_state=42, stratify=subject_ids)
    else:
        right_images, left_images, poses, gazes = _load_one_person(dataset_name, person_id, grayscale)
        (right_eye_train, right_eye_test,
         left_eye_train, left_eye_test,
         headpose_train, headpose_test,
         y_train, y_test) = train_test_split(right_images, left_images, poses, gazes, test_size=val_split,
                                             random_state=42)
        # filter suspicious
        if person_id == 7:
            raise ValueError("Person 7 is rejected.")
        right_images, left_images, poses, gazes = _load_one_person(dataset_name, i, grayscale)
        if person_id == 2:
            right_images, left_images, poses, gazes = _filter_subject_02(right_images, left_images, poses, gazes)
        if person_id == 10:
            right_images, left_images, poses, gazes = _filter_subject_10(right_images, left_images, poses, gazes)

    train_dataset = prepare_dataset((right_eye_train, left_eye_train, headpose_train), y_train, batch_size)
    # don't shuffle test dataset to have consistent outcomes
    test_dataset = prepare_dataset((right_eye_test, left_eye_test, headpose_test), y_test, batch_size, shuffle=False)

    return train_dataset, test_dataset, test_subject_ids