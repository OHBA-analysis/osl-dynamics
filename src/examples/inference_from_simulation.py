import numpy as np
import scipy.io as spio
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from taser.helpers.array_ops import get_one_hot
from taser.helpers.plotting import plot_covariances, plot_two_data_scales
from taser.helpers.tf_ops import gpu_growth
from taser.inference import gmm
from taser.data_manipulation import get_alpha_order
from taser.inference.models.inference_rnn import InferenceRNN
from taser.simulation import Simulation
from taser.training.trainers import AnnealingTrainer

# Limit the amount of memory that TensorFlow consumes by default
gpu_growth()

# Simulation parameters
