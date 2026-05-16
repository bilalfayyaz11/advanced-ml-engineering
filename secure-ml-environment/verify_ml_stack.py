import tensorflow as tf
import torch
import numpy as np
import pandas as pd
import sklearn
import cv2

print("===== ML STACK VERIFICATION =====")

print(f"TensorFlow Version: {tf.__version__}")
print(f"PyTorch Version: {torch.__version__}")
print(f"NumPy Version: {np.__version__}")
print(f"Pandas Version: {pd.__version__}")
print(f"Scikit-learn Version: {sklearn.__version__}")
print(f"OpenCV Version: {cv2.__version__}")

print("\nTensorFlow GPU Available:", tf.config.list_physical_devices('GPU'))
print("PyTorch GPU Available:", torch.cuda.is_available())

print("\nAll libraries loaded successfully!")
