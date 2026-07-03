"""TensorFlow/Keras GRD/GEE Back2Coh ResUNet architecture."""

from __future__ import annotations

from typing import Sequence


def _indexed_name(base: str, index: int) -> str:
    return base if index == 0 else f"{base}_{index}"


def _tf():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "The GRD/GEE Back2Coh model uses TensorFlow/Keras weights. "
            "Install TensorFlow in the runtime before loading the model."
        ) from exc
    return tf


def _residual_block(x, filters: int, block_index: int, kernel_size: tuple[int, int] = (3, 3)):
    """Residual block matching the released Keras GRD/GEE weight layout."""

    tf = _tf()
    layers = tf.keras.layers
    conv_base = block_index * 3
    norm_base = block_index * 2
    act_base = block_index * 2

    y = layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        kernel_initializer="he_normal",
        name=_indexed_name("conv2d", conv_base + 1),
    )(x)
    y = layers.BatchNormalization(
        axis=3,
        momentum=0.9,
        epsilon=1e-4,
        name=_indexed_name("batch_normalization", norm_base),
    )(y)
    y = layers.Activation("relu", name=_indexed_name("activation", act_base))(y)
    y = layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        kernel_initializer="he_normal",
        name=_indexed_name("conv2d", conv_base + 2),
    )(y)
    shortcut = layers.Conv2D(
        filters,
        (1, 1),
        padding="same",
        kernel_initializer="he_normal",
        name=_indexed_name("conv2d", conv_base),
    )(x)
    y = layers.BatchNormalization(
        axis=3,
        momentum=0.9,
        epsilon=1e-4,
        name=_indexed_name("batch_normalization", norm_base + 1),
    )(y)
    y = layers.Add(name=_indexed_name("add", block_index))([shortcut, y])
    return layers.Activation("relu", name=_indexed_name("activation", act_base + 1))(y)


def build_resunet(
    input_shape: Sequence[int] = (128, 128, 2),
    output_channels: int = 2,
):
    """Build the GRD/GEE Back2Coh ResUNet used by ``model.weights.h5``.

    The model accepts preprocessed GRD/GEE patches with NHWC layout and two
    channels: primary/t1 and secondary/t2. The first output channel is the
    predicted coherence mean. The second channel is the training uncertainty
    head retained in the original weights and ignored during public inference.
    """

    if output_channels != 2:
        raise ValueError("The released GRD/GEE Keras weights expect two output channels.")

    tf = _tf()
    layers = tf.keras.layers

    inputs = layers.Input(shape=tuple(input_shape), name="input")

    enc32 = _residual_block(inputs, 32, 0)
    x = layers.MaxPool2D(pool_size=(2, 2), name="maxpool_32-64")(enc32)
    enc64 = _residual_block(x, 64, 1)
    x = layers.MaxPool2D(pool_size=(2, 2), name="maxpool_64-128")(enc64)
    enc128 = _residual_block(x, 128, 2)
    x = layers.MaxPool2D(pool_size=(2, 2), name="maxpool_128-256")(enc128)
    enc256 = _residual_block(x, 256, 3)
    x = layers.MaxPool2D(pool_size=(2, 2), name="maxpool_256-512")(enc256)

    x = _residual_block(x, 512, 4)

    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="up_sampling2d")(x)
    x = layers.Conv2D(
        256,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        name="upsampling_256-128",
    )(x)
    x = layers.Concatenate(axis=3, name="concat256")([enc256, x])
    x = _residual_block(x, 256, 5)

    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="up_sampling2d_1")(x)
    x = layers.Conv2D(
        128,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        name="upsampling_128-64",
    )(x)
    x = layers.Concatenate(axis=3, name="concat128")([enc128, x])
    x = _residual_block(x, 128, 6)

    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="up_sampling2d_2")(x)
    x = layers.Conv2D(
        64,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        name="upsampling_64-32",
    )(x)
    x = layers.Concatenate(axis=3, name="concat64")([enc64, x])
    x = _residual_block(x, 64, 7)

    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="up_sampling2d_3")(x)
    x = layers.Conv2D(
        32,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        name="upsampling_32-16",
    )(x)
    x = layers.Concatenate(axis=3, name="concat32")([enc32, x])
    x = _residual_block(x, 32, 8)

    mean = layers.Conv2D(
        1,
        (1, 1),
        activation="sigmoid",
        padding="same",
        kernel_initializer="he_normal",
        name="mean",
    )(x)
    log_variance = layers.Conv2D(
        1,
        (1, 1),
        activation="sigmoid",
        padding="same",
        kernel_initializer="he_normal",
        name="log_variance",
    )(x)
    outputs = layers.Concatenate(axis=3, name="output")([mean, log_variance])

    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name="Back2Coh_GRD_GEE")
