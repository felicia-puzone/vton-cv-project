import glob

from dataset_loading import class_bounding_box_extraction
import numpy as np
import os
import cv2

images_patt = "D:\\Tumor\\DressCodeDoubleResized\\dresses"

# Ex: your custom label format
# import fiftyone as fo
label_map = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "head": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
}
keypoints_map = {
    0.0: "nose",
    1.0: "neck",
    2.0: "right shoulder",
    3.0: "right elbow",
    4.0: "right wrist",
    5.0: "left shoulder",
    6.0: "left elbow",
    7.0: "left wrist",
    8.0: "right hip",
    9.0: "right knee",
    10.0: "right ankle",
    11.0: "left hip",
    12.0: "left knee",
    13.0: "left ankle",
    14.0: "right eye",
    15.0: "left eye",
    16.0: "right ear",
    17.0: "left ear"
}


def label_box_extraction(image: np.ndarray):
    print(len(image.shape))
    if len(image.shape)==2:
        classes = np.unique(image.reshape((-1,1)), axis=0)
    elif len(image.shape)==3:
        classes = np.unique(image.reshape(-1, 3), axis=0)
    else:
        print("can only support 2 or 3 dimensional arrays")
        return None
    print(classes)
    print(image.shape)
    cv2.imshow("lol",image)
    cv2.waitKey()

"""
annotations = {
    "/path/to/images/000001.jpg": [
        {"bbox": ..., "label": ...},
        ...
    ],
    ...
}

# Create dataset
dataset = fo.Dataset(name="my-detection-dataset")

# Persist the dataset on disk in order to
# be able to load it in one line in the future
dataset.persistent = True

# Add your samples to the dataset
for filepath in glob.glob(images_patt):
    sample = fo.Sample(filepath=filepath)

    # Convert detections to FiftyOne format
    detections = []
    for obj in annotations[filepath]:
        label = obj["label"]

        # Bounding box coordinates should be relative values
        # in [0, 1] in the following format:
        # [top-left-x, top-left-y, width, height]
        bounding_box = obj["bbox"]

        detections.append(
            fo.Detection(label=label, bounding_box=bounding_box)
        )

    # Store detections in a field name of your choice
    sample["ground_truth"] = fo.Detections(detections=detections)

    dataset.add_sample(sample)
"""

if __name__ == "__main__":
    for image_file in os.listdir(os.path.join(images_patt, "label_maps")):
        path=os.path.join(images_patt, "label_maps",image_file)
        print(image_file)
        image=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        label_box_extraction(image)
