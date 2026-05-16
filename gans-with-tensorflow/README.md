# MNIST GAN Image Generation with TensorFlow

## Objective
Build and train a Generative Adversarial Network using TensorFlow to generate synthetic handwritten digit images from random noise.

## Project Overview
This project implements a basic GAN architecture using the MNIST handwritten digit dataset. The Generator learns to create fake digit images, while the Discriminator learns to distinguish real MNIST digits from generated ones.

## Tools Used
- Python 3
- TensorFlow
- Keras
- NumPy
- Matplotlib
- MNIST Dataset
- Linux Virtual Environment

## Key Skills Demonstrated
- Generative Adversarial Network architecture
- TensorFlow model implementation
- Generator and Discriminator design
- Custom training loop with GradientTape
- Image generation from random noise
- Model saving in Keras format
- CPU-optimized ML training workflow

## Outputs
- Trained Generator model
- Trained Discriminator model
- Generated handwritten digit images across training epochs

## Troubleshooting Log

### CPU-Optimized Training
The original full MNIST training setup was reduced from 60,000 samples and 10 epochs to 10,000 samples and 3 epochs to complete efficiently in a CPU-only lab environment.

### GPU Warning Handling
TensorFlow displayed CUDA-related warnings because no NVIDIA GPU or CUDA drivers were available. This was expected and did not prevent CPU-based training.

### Modern TensorFlow API Update
Updated deprecated LeakyReLU parameter usage from alpha to negative_slope for compatibility with modern TensorFlow/Keras versions.

## Files
- `mnist_gan.py` — GAN training script
- `mnist_gan_generator.keras` — saved generator model
- `mnist_gan_discriminator.keras` — saved discriminator model
- `generated_images/` — generated digit images from training epochs
