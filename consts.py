INPUT_IMG_SHAPE = (512, 512, 1)
OUTPUT_IMG_SHAPE = (224, 224, 6)
CROP_SIZE = (224, 224, 1)
CROP = False
BATCH_SIZE = 32
EPOCHS = 100
HASH_BUCKET_SIZE = 200
# https://www.quora.com/How-do-I-determine-the-number-of-dimensions-for-word-embedding
EMBEDDING_DIMS = int(HASH_BUCKET_SIZE ** 0.25) + 1
HIDDEN_UNITS = [6, 6]
CONCAT_HIDDEN_UNITS = [100] 
SHUFFLE_BUFFER_SIZE = 487_344
TENSORBOARD_UPDATE_FREQUENCY = 500  # setting this too low will slow training!
RANDOM_SPLIT_SEED = 55
TRAIN = True
# TRAIN = False
TRAIN_CONV = False
