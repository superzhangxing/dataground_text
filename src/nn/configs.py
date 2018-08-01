import tensorflow as tf
import os
import time

# -------------- Parameters---------------

# Load data


# Model Hyperparameters
tf.flags.DEFINE_integer('embedding_dim', 300, "Dimensionality of word embedding(default: 300)")
tf.flags.DEFINE_string('filter_sizes', '3,4,5', "Comma-separated filter sizes(default:'3,4,5')")
tf.flags.DEFINE_integer('num_filters', 128, 'Number of filters per filter size(default: 100)')
tf.flags.DEFINE_float('dropout', 0.7, 'Dropout keep probability(default: 0.7)')
tf.flags.DEFINE_float('l2_reg_lambda', 5e-5, "L2 regularization lambda(default: 5e-5)")

# Training parameters
tf.flags.DEFINE_integer('batch_size', 64, "Batch Size(fault: 64)")
tf.flags.DEFINE_integer('num_epochs', 100, "Training epochs(default: 100)")
tf.flags.DEFINE_integer('evaluate_every', "Evaluate model on dev set after this many steps(default: 100)")
tf.flags.DEFINE_integer('checkpoint_every', 100, "Save model after this many steps(default: 100)")
tf.flags.DEFINE_integer('num_checkpoints', 5, 'Number of checkpoints to store (default: 5)')

# Misc Parameters

FLAGS = tf.flags.FLAGS
