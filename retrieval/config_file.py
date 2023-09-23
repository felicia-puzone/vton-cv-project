from pathlib import Path
from torch.cuda import is_available as cuda_is_available

DATASET_PATH_STR = "D:\\IngMagistrale\\Computer_vision\\project\\dress_code_dataset_example\\DressCodeFinal3.0" \
                   "\\DressCodeFinal3.0"
DATASET_PATH = Path("D:\\IngMagistrale\\Computer_vision\\project\\dress_code_dataset_example\\DressCodeFinal3.0"
                    "\\DressCodeFinal3.0\\upper_body")

DATASET_IMAGES_PATH = Path("D:\\IngMagistrale\\Computer_vision\\project\\dress_code_dataset_example\\DressCodeFinal3"
                           ".0\\DressCodeFinal3.0\\upper_body\\images")
DATASET_LABEL_MAPS_PATH = Path("D:\\IngMagistrale\\Computer_vision\\project\\dress_code_dataset_example"
                               "\\DressCodeFinal3.0\\DressCodeFinal3.0\\upper_body\\label_maps")

DST_DATA = Path("data")
DST_WORN_CLOTH_PATH = Path("data/worn-cloth")
DST_IN_SHOP_CLOTH_PATH = Path("data/in-shop-cloth")

TRAIN_SAMPLES_FILE = Path("data/train_samples.txt")
TEST_SAMPLES_FILE = Path("data/test_samples.txt")
TRAIN_FILE = Path("data/train.txt")
TEST_FILE = Path("data/test.txt")

N_NEGATIVE_EXAMPLES = 5
N_POSITIVE_EXAMPLES = N_NEGATIVE_EXAMPLES

CLASS_NAMES = ['no match', 'match']

MODEL_NAME_DEFAULT = "SimilarityNet_synth_data_6.pth"
CHECKPOINT_PATH_DEFAULT = "models/" + MODEL_NAME_DEFAULT

# N_BEST_KEYPOINTS = 200
N_BEST_KEYPOINTS = 50
N_MAX_ORB_FEATURES = 500
N_MAX_ORB_POINT_FEATURES = 32
# N_HISTOGRAM_BINS = 0
N_HISTOGRAM_BINS = 256
N_FEATURES = N_BEST_KEYPOINTS * (2 + 1 + 1 + N_MAX_ORB_POINT_FEATURES) + N_HISTOGRAM_BINS
# N_FEATURES = N_BEST_KEYPOINTS * (2 + 1 + 1 + N_MAX_ORB_POINT_FEATURES)
IN_FEATURES = N_FEATURES * 6

RESIZE_IMG_DIMS = (384, 512)  # (W,H) in opencv format
TARGET_RGB_COLOR = [0, 0, 128]
# [254, 85, 0]

# NUM_WORKERS = os.cpu_count()
# lo tengo ad 1 perché il mio computer sembra essere scarso
NUM_WORKERS = 1
NUM_EPOCHS = 50
BATCH_SIZE = 32
HIDDEN_UNITS = 512
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-6
PATIENCE = 3
MIN_DELTA = 2e-3

DEVICE = "cuda" if cuda_is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

TOP_K = 5