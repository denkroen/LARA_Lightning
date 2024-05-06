import os

# Training hyperparameters
INPUT_SIZE = 150
NUM_CLASSES = 7
LEARNING_RATE = 0.001
BATCH_SIZE = 100
NUM_EPOCHS = 50
VAL_BATCHES = 200

# Dataset
DATA_DIR = ""
NUM_WORKERS = os.cpu_count()-2

WINDOW_LENGTH = 100
NUM_SENSORS = 126


# Compute related
ACCELERATOR = "cpu"
DEVICE = "cpu"
PRECISION = 32

# TCNN Related
MODE = "attribute" # / "classification"
NUM_FILTERS = 64
FILTER_SIZE = 5

#if attributes
NUM_ATTRIBUTES = 16
PATH_ATTRIBUTES = ""



