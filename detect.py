import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
#sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import custom

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
RESULT_DIR = os.path.join(ROOT_DIR, "result")



# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
CUSTOM_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_object_0025.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

print("MODEL_DIR: ", MODEL_DIR)
print("CUSTOM_MODEL_PATH: ", CUSTOM_MODEL_PATH)
print("IMAGE_DIR: ", IMAGE_DIR)

class CustomInferenceConfig(custom.CustomConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = CustomInferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(CUSTOM_MODEL_PATH, by_name=True)

# Load a random image from the images folder
class_names = custom.classess
file_names = glob(os.path.join(IMAGE_DIR, "*.JPEG"))
colors = visualize.random_colors(len(class_names))
for file_name in file_names:
    image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    filename = os.path.join(RESULT_DIR, os.path.basename(file_name))
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'], colors=colors, filename=filename, auto_show=False)

