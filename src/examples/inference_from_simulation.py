import numpy as np
import scipy.io as spio
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from taser.helpers.matplotlib import plot_covariances, plot_two_data_scales
from taser.helpers.numpy import get_one_hot
from taser.helpers.tensorflow import gpu_growth
from taser.inference import gmm
from taser.inference.functions import get_alpha_order
from taser.inference.models.inference_rnn import InferenceRNN
from taser.simulation import Simulation
from taser.training.trainers import AnnealingTrainer

# Limit the amount of memory that TensorFlow consumes by default
gpu_growth()

# Simulation parameters
