import os

import tensorflow as tf
from tensorflow.python.keras.backend import set_session

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.WARN)

# for rtx 20xx cards
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

# image config
INPUT_IMG_SHAPE = (512, 512, 1)
OUTPUT_IMG_SHAPE = (224, 224, 12)
CROP_SIZE = (128, 128, 12)
CROP = False

# nn config
HASH_BUCKET_SIZE = 10
WIDE_NEURONS = 10
WIDE_DEEP_NEURONS = 6
# https://www.quora.com/How-do-I-determine-the-number-of-dimensions-for-word-embedding
EMBEDDING_DIMS = int(HASH_BUCKET_SIZE ** 0.25) + 1
DEEP_HIDDEN_UNITS = [6, 6]

# net parameters
REGULARIZATION = 0.05
LR = 0.01

# net type
CONV_TYPE = "irn2"

# batch config
BATCH_SIZE = 32

# training config
EPOCHS = 100
SHUFFLE_BUFFER_SIZE = 487_344
TENSORBOARD_UPDATE_FREQUENCY = 500  # setting this too low will slow training!
RANDOM_SPLIT_SEED = 55
TRAIN = True
GPU = True
# TRAIN = False
# config = {
#     "INPUT_IMG_SHAPE": INPUT_IMG_SHAPE,
#     "OUTPUT_IMG_SHAPE": OUTPUT_IMG_SHAPE,
#     "CROP_SIZE": CROP_SIZE,
#     "CROP": CROP,
#     "BATCH_SIZE": BATCH_SIZE,
#     "EPOCHS": EPOCHS,
#     "HASH_BUCKET_SIZE": HASH_BUCKET_SIZE,
#     "EMBEDDING_DIMS": EMBEDDING_DIMS,
#     "HIDDEN_UNITS": HIDDEN_UNITS,
#     "CONCAT_HIDDEN_UNITS": CONCAT_HIDDEN_UNITS,
#     "SHUFFLE_BUFFER_SIZE": SHUFFLE_BUFFER_SIZE,
#     "TENSORBOARD_UPDATE_FREQUENCY": TENSORBOARD_UPDATE_FREQUENCY,
#     "RANDOM_SPLIT_SEED": RANDOM_SPLIT_SEED,
#     "TRAIN": TRAIN,
# }

if not GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # For CPU
