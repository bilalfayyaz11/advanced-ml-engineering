import os
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

print("===== GAN INITIALIZATION =====")

(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

X_train = X_train / 127.5 - 1.0
X_train = np.expand_dims(X_train, axis=-1)

# Fast CPU training subset
X_train = X_train[:10000]

BUFFER_SIZE = 10000
BATCH_SIZE = 128
NOISE_DIM = 100
EPOCHS = 3
IMG_SHAPE = (28, 28, 1)

print(f"Training Dataset Shape: {X_train.shape}")

train_dataset = (
    tf.data.Dataset.from_tensor_slices(X_train)
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)

def build_generator():
    model = tf.keras.Sequential([
        layers.Input(shape=(NOISE_DIM,)),
        layers.Dense(7 * 7 * 256, use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Reshape((7, 7, 256)),

        layers.Conv2DTranspose(
            128,
            (5, 5),
            strides=(1, 1),
            padding="same",
            use_bias=False
        ),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(
            64,
            (5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False
        ),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(
            1,
            (5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            activation="tanh"
        )
    ])

    return model

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Input(shape=IMG_SHAPE),

        layers.Conv2D(
            64,
            (5, 5),
            strides=(2, 2),
            padding="same"
        ),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),

        layers.Conv2D(
            128,
            (5, 5),
            strides=(2, 2),
            padding="same"
        ),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])

    return model

generator = build_generator()
discriminator = build_discriminator()

print("\n===== GENERATOR SUMMARY =====")
generator.summary()

print("\n===== DISCRIMINATOR SUMMARY =====")
discriminator.summary()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(images):
    noise = tf.random.normal([tf.shape(images)[0], NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = cross_entropy(
            tf.ones_like(fake_output),
            fake_output
        )

        disc_loss = (
            cross_entropy(tf.ones_like(real_output), real_output)
            +
            cross_entropy(tf.zeros_like(fake_output), fake_output)
        )

    gradients_of_generator = gen_tape.gradient(
        gen_loss,
        generator.trainable_variables
    )

    gradients_of_discriminator = disc_tape.gradient(
        disc_loss,
        discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )

    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )

    return gen_loss, disc_loss

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    os.makedirs("generated_images", exist_ok=True)

    plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(
            predictions[i, :, :, 0] * 127.5 + 127.5,
            cmap="gray"
        )
        plt.axis("off")

    plt.savefig(f"generated_images/image_at_epoch_{epoch:04d}.png")
    plt.close()

def train(dataset, epochs):
    seed = tf.random.normal([16, NOISE_DIM])

    for epoch in range(epochs):
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Generator Loss: {gen_loss:.4f} | "
            f"Discriminator Loss: {disc_loss:.4f}"
        )

        generate_and_save_images(generator, epoch + 1, seed)

    generator.save("mnist_gan_generator.keras")
    discriminator.save("mnist_gan_discriminator.keras")

train(train_dataset, EPOCHS)

print("GAN Training Complete!")
