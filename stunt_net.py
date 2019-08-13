import os
import sys

from default_config import DF_LOCATION

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pathlib
import time

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.WARN)
from sklearn.model_selection import train_test_split

from consts import BATCH_SIZE, EPOCHS, EMBEDDING_DIMS, HASH_BUCKET_SIZE, HIDDEN_UNITS, SHUFFLE_BUFFER_SIZE, \
    TENSORBOARD_UPDATE_FREQUENCY, OUTPUT_IMG_SHAPE, CROP, CROP_SIZE, RANDOM_SPLIT_SEED, CONCAT_HIDDEN_UNITS
from rxrx1_df import get_dataframe
from rxrx1_ds import get_ds
from utils import get_random_string, get_number_of_target_classes

# for rtx 20xx cards
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def wide_and_deep_classifier(
        inputs, linear_feature_columns, dnn_feature_columns,
        number_of_target_classes
):
    deep = tf.keras.layers.DenseFeatures(dnn_feature_columns, name='deep_inputs')(inputs)
    for layerno, numnodes in enumerate(HIDDEN_UNITS):
        deep = tf.keras.layers.Dense(numnodes, activation='relu', name='dnn_{}'.format(layerno + 1))(deep)
    wide = tf.keras.layers.DenseFeatures(linear_feature_columns, name='wide_inputs')(inputs)

    img_net = tf.keras.applications.InceptionResNetV2(
        include_top=False,
        weights=None,
        # weights='imagenet',
        input_tensor=inputs["img"],
        input_shape=None,
        pooling="max"
    )

    output = tf.keras.layers.concatenate([deep, wide, img_net.output], name='both')

    for layerno, numnodes in enumerate(CONCAT_HIDDEN_UNITS):
        output = tf.keras.layers.Dense(numnodes, activation='relu', name=f'cnn_{layerno + 1}')(output)
    output = tf.keras.layers.Dense(number_of_target_classes, activation='softmax', name='pred')(output)
    model = tf.keras.Model(inputs, output)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


def get_features(ds):
    real = {name: tf.feature_column.numeric_column(name)
            for name, dtype in ds.output_types[0].items()
            if name != "img" and dtype.name in ["int64"]}
    sparse = {name: tf.feature_column.categorical_column_with_hash_bucket(name, hash_bucket_size=HASH_BUCKET_SIZE)
              for name, dtype in ds.output_types[0].items()
              if dtype.name in ["string"]}

    inputs = {colname: tf.keras.layers.Input(name=colname, shape=(), dtype='float32')
              for colname in real.keys()}
    inputs.update({colname: tf.keras.layers.Input(name=colname, shape=(), dtype='string')
                   for colname in sparse.keys()})

    # we should have a crossed column
    sparse['crossed'] = tf.feature_column.crossed_column(
        ['well_column', 'well_row'],
        int(HASH_BUCKET_SIZE ** 2.0)
    )

    # embed all the sparse columns
    embed = {'embed_{}'.format(colname): tf.feature_column.embedding_column(col, EMBEDDING_DIMS)
             for colname, col in sparse.items()}
    real.update(embed)

    # one-hot encode the sparse columns
    sparse = {colname: tf.feature_column.indicator_column(col)
              for colname, col in sparse.items()}

    inputs.update({
        "img": tf.keras.layers.Input(name="img", shape=CROP_SIZE if CROP else OUTPUT_IMG_SHAPE, dtype='float32')
    })
    return inputs, sparse, real


def train_model(model, train_ds, test_ds, run_id, steps_per_epoch, validation_steps_per_epoch):
    model_path = os.path.join("models", run_id)
    pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        monitor='val_acc',
        filepath=os.path.join(model_path, 'model.cpt'),
        save_weights_only=False,
        save_best_only=True,
        verbose=1,
        load_weights_on_restart=True
    )
    callback = tf.keras.callbacks.TensorBoard(model_path, update_freq=TENSORBOARD_UPDATE_FREQUENCY)
    callback.set_model(model)

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps_per_epoch,
        callbacks=[cp_callback, callback]
    )
    return history


def export_saved_model(run_id, model, feature_columns):
    export_dir = os.path.join('models', run_id, f'model_{time.strftime("%Y%m%d-%H%M%S")}')
    print('Exporting to {}'.format(export_dir))
    tf.keras.experimental.export_saved_model(model, export_dir, custom_objects={
        'feature_columns': feature_columns
    })


def run_inferrance():
    # todo:
    # load test dataset (similar to training data, but without SIRNA)
    # run predict with our trained model
    # write to csv
    # upload to kaggle
    pass


def main(_run_id=None):
    df = get_dataframe(DF_LOCATION)
    number_of_target_classes = get_number_of_target_classes(df)

    if _run_id is None:
        run_id = get_random_string(8)
    else:
        run_id = _run_id

    # no need for ID
    df.pop("id_code")

    # df.pop("site_num")
    # df.pop("microscope_channel")
    # df.pop("well_type")

    train_df, test_df = train_test_split(df, random_state=RANDOM_SPLIT_SEED)
    training_samples = len(train_df.index)
    training_steps_per_epoch = training_samples // BATCH_SIZE
    validation_samples = len(test_df.index)
    validation_steps_per_epoch = validation_samples // BATCH_SIZE

    print(f"""
    number_of_target_classes: {number_of_target_classes}
    total_samples: {training_samples + validation_samples}
    training_samples: {training_samples}
    training_steps_per_epoch: {training_steps_per_epoch}
    validation_samples: {validation_samples}
    validation_steps_per_epoch: {validation_steps_per_epoch}
    run_id: {run_id}
    gpu: {tf.test.is_gpu_available()}
    cuda: {tf.test.is_built_with_cuda()}
    """)

    train_ds = get_ds(
        train_df, number_of_target_classes=number_of_target_classes,
        training=True, shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
        perform_img_augmentation=True
    )
    test_ds = get_ds(
        test_df, number_of_target_classes=number_of_target_classes,
        perform_img_augmentation=False
    )

    inputs, sparse, real = get_features(train_ds)
    model = wide_and_deep_classifier(
        inputs,
        linear_feature_columns=sparse.values(),
        dnn_feature_columns=real.values(),
        number_of_target_classes=number_of_target_classes
    )

    train_model(model, train_ds, test_ds, run_id, training_steps_per_epoch, validation_steps_per_epoch)
    run_inferrance()
    export_saved_model(run_id, model, real)
    print("")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        _run_id = sys.argv[1]
    else:
        _run_id = None
    main(_run_id=_run_id)
