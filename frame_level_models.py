# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils

import tensorflow.contrib.slim as slim
from tensorflow import flags

from loupe import NetVLAD

FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_integer("vlad_cluster_size", 128,
                     "Number of units in the VLAD cluster layer.")
flags.DEFINE_integer("vlad_hidden_size", 1024,
                     "Number of units in the VLAD hidden layer.")
flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")

class FrameLevelLogisticModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the
    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input,
                               axis=[1]) / denominators

    output = slim.fully_connected(
        avg_pooled, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(1e-8))
    return {"predictions": output}

class DbofModel(models.BaseModel):
  """Creates a Deep Bag of Frames model.

  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.

  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    cluster_weights = tf.get_variable("cluster_weights",
      [feature_size, cluster_size],
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
    tf.summary.histogram("cluster_weights", cluster_weights)
    activation = tf.matmul(reshaped_input, cluster_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="cluster_bn")
    else:
      cluster_biases = tf.get_variable("cluster_biases",
        [cluster_size],
        initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
      tf.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("cluster_output", activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])
    activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)

    hidden1_weights = tf.get_variable("hidden1_weights",
      [cluster_size, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
    tf.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(activation, hidden1_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("hidden1_output", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        **unused_params)

class LstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        **unused_params)

class LiteLstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0

    pooled = tf.nn.pool(model_input, (5,), 'MAX', 'VALID', strides=(5,))
    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, pooled,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        **unused_params)

class LstmWeightPoolingModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    hidden_size = 8192
    max_frames = model_input.get_shape().as_list()[1]

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0

    pooled = tf.nn.pool(model_input, (5,), 'MAX', 'VALID', strides=(5,))
    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, pooled,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    cat1 = tf.concat((state[0].h, state[1].h), axis=1)

    hidden_weights = tf.get_variable("hidden_weights",
      [2*lstm_size, hidden_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(2*lstm_size)))
    hidden_biases = tf.get_variable("hidden_biases",
      [hidden_size],
      initializer = tf.random_normal_initializer(stddev=0.01))
    tf.summary.histogram("hidden_weights", hidden_weights)
    tf.summary.histogram("hidden_biases", hidden_biases)

    hidden = tf.nn.relu6(tf.matmul(cat1, hidden_weights) + hidden_biases)

    gate_weights = tf.get_variable("gate_weights",
      [hidden_size, max_frames],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(hidden_size)))
    gate_biases = tf.get_variable("gate_biases",
      [max_frames],
      initializer = tf.random_normal_initializer(stddev=0.01))
    tf.summary.histogram("gate_weights", gate_weights)
    tf.summary.histogram("gate_biases", gate_biases)

    gate = tf.nn.sigmoid(tf.matmul(hidden, gate_weights) + gate_biases)
    gate = tf.reshape(gate, (-1, max_frames, 1))

    weight_pooled = utils.FramePooling(gate * model_input, 'average')

    cat2 = tf.concat((cat1, weight_pooled), axis=1)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=cat2,
        vocab_size=vocab_size,
        **unused_params)

class LstmWeightPoolingV2Model(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """TODO: description
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    hidden_size = 8192
    max_frames = model_input.get_shape().as_list()[1]

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0

    pooled = tf.nn.pool(model_input, (5,), 'MAX', 'VALID', strides=(5,))
    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, pooled,
                                       sequence_length=num_frames,
                                       dtype=tf.float32,
                                       scope="LSTM")

    cat1 = tf.concat((state[0].h, state[1].h), axis=1)

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])
    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, pooled,
                                       sequence_length=num_frames,
                                       dtype=tf.float32,
                                       scope="gate_LSTM")

    cat2 = tf.concat((state[0].h, state[1].h), axis=1)

    hidden_weights = tf.get_variable("hidden_weights",
      [2*lstm_size, hidden_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(2*lstm_size)))
    hidden_biases = tf.get_variable("hidden_biases",
      [hidden_size],
      initializer = tf.random_normal_initializer(stddev=0.01))
    tf.summary.histogram("hidden_weights", hidden_weights)
    tf.summary.histogram("hidden_biases", hidden_biases)

    hidden = tf.nn.relu6(tf.matmul(cat2, hidden_weights) + hidden_biases)

    gate_weights = tf.get_variable("gate_weights",
      [hidden_size, max_frames],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(hidden_size)))
    gate_biases = tf.get_variable("gate_biases",
      [max_frames],
      initializer = tf.random_normal_initializer(stddev=0.01))
    tf.summary.histogram("gate_weights", gate_weights)
    tf.summary.histogram("gate_biases", gate_biases)

    gate = tf.nn.sigmoid(tf.matmul(hidden, gate_weights) + gate_biases)
    gate = tf.reshape(gate, (-1, max_frames, 1))

    weight_pooled = utils.FramePooling(gate * model_input, 'average')

    cat3 = tf.concat((cat1, weight_pooled), axis=1)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=cat3,
        vocab_size=vocab_size,
        **unused_params)

class LstmWeightPoolingV3Model(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
    """TODO: Descriptionx
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    hidden_size = 8192
    max_frames = model_input.get_shape().as_list()[1]

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0

    pooled = tf.nn.pool(model_input, (5,), 'MAX', 'VALID', strides=(5,))
    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, pooled,
                                       sequence_length=num_frames,
                                       dtype=tf.float32,
                                       scope="LSTM")

    cat1 = tf.concat((state[0].h, state[1].h), axis=1)
    tf.summary.histogram("cat1", cat1)

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])
    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, pooled,
                                       sequence_length=num_frames,
                                       dtype=tf.float32,
                                       scope="gate_LSTM")

    cat2 = tf.concat((state[0].h, state[1].h), axis=1)
    tf.summary.histogram("cat2", cat2)

    hidden_weights = tf.get_variable("hidden_weights",
      [2*lstm_size, hidden_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(2*lstm_size)))
    hidden_biases = tf.get_variable("hidden_biases",
      [hidden_size],
      initializer = tf.random_normal_initializer(stddev=0.01))
    tf.summary.histogram("hidden_weights", hidden_weights)
    tf.summary.histogram("hidden_biases", hidden_biases)

    hidden = tf.nn.relu(tf.matmul(cat2, hidden_weights) + hidden_biases)
    tf.summary.histogram("hidden", hidden)

    gate_weights = tf.get_variable("gate_weights",
      [hidden_size, max_frames],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(hidden_size)))
    gate_biases = tf.get_variable("gate_biases",
      [max_frames],
      initializer = tf.random_normal_initializer(stddev=0.01))
    tf.summary.histogram("gate_weights", gate_weights)
    tf.summary.histogram("gate_biases", gate_biases)

    gate = slim.batch_norm(tf.matmul(hidden, gate_weights) + gate_biases,
                           center=True,
                           scale=True,
                           is_training=is_training,
                           scope='gate_bn')
    gate = tf.reshape(tf.nn.sigmoid(gate), (-1, max_frames, 1))
    tf.summary.histogram("gate_values", gate)

    weight_pooled = utils.FramePooling(gate * model_input, 'max')

    cat3 = tf.concat((cat1, weight_pooled), axis=1)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=cat3,
        vocab_size=vocab_size,
        **unused_params)


class Conv1dLstmModel(models.BaseModel):
  """ TODO: description
  """

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    filter_width = 5
    feature_size = model_input.get_shape().as_list()[2]
   
    conv_weights = tf.get_variable("conv_weights",
      [filter_width, feature_size, feature_size],
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
    conv_biases = tf.get_variable("conv_biases",
      [feature_size],
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
    tf.summary.histogram("conv_weights", conv_weights)
    tf.summary.histogram("conv_biases", conv_biases)
    
    conv = tf.nn.conv1d(model_input, conv_weights, filter_width, 'VALID') + conv_biases
    pooled = tf.nn.relu(conv)

    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])
   
    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, pooled,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        **unused_params)

class LiteLstmCGModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, add_batch_norm=None, is_training=True, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0

    pooled = tf.nn.pool(model_input, (5,), 'MAX', 'VALID', strides=(5,))
    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, pooled,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    hidden = state[-1].h

    gating_weights = tf.get_variable("gating_weights",
      [lstm_size, lstm_size],
      initializer = tf.random_normal_initializer(
      stddev=1 / math.sqrt(lstm_size)))

    gates = tf.matmul(hidden, gating_weights)

    if add_batch_norm:
      gates = slim.batch_norm(
        gates,
        center=True,
        scale=True,
        is_training=is_training,
        scope="gating_bn")
    else:
      gating_biases = tf.get_variable("gating_biases",
        [input_dim],
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(input_dim)))
      gates += gating_biases

    gates = tf.sigmoid(gates)

    activation = tf.multiply(hidden,gates)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=activation, 
        vocab_size=vocab_size,
        **unused_params)

class NetVladModel(models.BaseModel):
  """TODO: Description
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.vlad_cluster_size
    hidden1_size = hidden_size or FLAGS.vlad_hidden_size

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    cluster_weights = tf.get_variable("cluster_weights",
                                      [feature_size, cluster_size],
                                      initializer = tf.random_normal_initializer(
                                        stddev=1 / math.sqrt(feature_size)))
        
    activation = tf.matmul(reshaped_input, cluster_weights)
        
    if add_batch_norm:
      activation = slim.batch_norm(
        activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="cluster_bn")
    else:
      cluster_biases = tf.get_variable("cluster_biases",
                                       [cluster_size],
                                       initializer = tf.random_normal_initializer(
                                         stddev=1 / math.sqrt(feature_size)))
      activation += cluster_biases
          
    activation = tf.nn.softmax(activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])

    a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

    cluster_weights2 = tf.get_variable("cluster_weights2",
                                       [1, feature_size, cluster_size],
                                       initializer = tf.random_normal_initializer(
                                         stddev=1 / math.sqrt(feature_size)))
        
    a = tf.multiply(a_sum,cluster_weights2)
        
    activation = tf.transpose(activation,perm=[0,2,1])
        
    reshaped_input = tf.reshape(reshaped_input,[-1, max_frames, feature_size])

    vlad = tf.matmul(activation,reshaped_input)
    vlad = tf.transpose(vlad,perm=[0,2,1])
    vlad = tf.subtract(vlad,a)
        
    vlad = tf.nn.l2_normalize(vlad,1)

    vlad = tf.reshape(vlad,[-1, cluster_size*feature_size])
    vlad = tf.nn.l2_normalize(vlad,1)
    tf.summary.histogram("vlad", vlad)

    hidden1_weights = tf.get_variable("hidden1_weights",
      [cluster_size*feature_size, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
    #tf.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(vlad, hidden1_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    activation = tf.nn.relu(activation)
    tf.summary.histogram("hidden1_output", activation)

    gating_weights = tf.get_variable("gating_weights",
      [hidden1_size, hidden1_size],
      initializer = tf.random_normal_initializer(
      stddev=1 / math.sqrt(hidden1_size)))

    gates = tf.matmul(activation, gating_weights)
    tf.summary.histogram("gates", gates)

    if add_batch_norm:
      gates = slim.batch_norm(
        gates,
        center=True,
        scale=True,
        is_training=is_training,
        scope="gating_bn")
    else:
      gating_biases = tf.get_variable("gating_biases",
        [input_dim],
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(input_dim)))
      gates += gating_biases

    gates = tf.sigmoid(gates)

    activation = tf.multiply(activation,gates)
    tf.summary.histogram("activation", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        **unused_params)

class NetVladHighwayModel(models.BaseModel):
  """TODO: Description
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.vlad_cluster_size
    hidden1_size = hidden_size or FLAGS.vlad_hidden_size

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    cluster_weights = tf.get_variable("cluster_weights",
                                      [feature_size, cluster_size],
                                      initializer = tf.random_normal_initializer(
                                        stddev=1 / math.sqrt(feature_size)))
        
    activation = tf.matmul(reshaped_input, cluster_weights)
        
    if add_batch_norm:
      activation = slim.batch_norm(
        activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="cluster_bn")
    else:
      cluster_biases = tf.get_variable("cluster_biases",
                                       [cluster_size],
                                       initializer = tf.random_normal_initializer(
                                         stddev=1 / math.sqrt(feature_size)))
      activation += cluster_biases
          
    activation = tf.nn.softmax(activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])

    a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

    cluster_weights2 = tf.get_variable("cluster_weights2",
                                       [1, feature_size, cluster_size],
                                       initializer = tf.random_normal_initializer(
                                         stddev=1 / math.sqrt(feature_size)))
        
    a = tf.multiply(a_sum,cluster_weights2)
        
    activation = tf.transpose(activation,perm=[0,2,1])
        
    reshaped_input = tf.reshape(reshaped_input,[-1, max_frames, feature_size])

    vlad = tf.matmul(activation,reshaped_input)
    vlad = tf.transpose(vlad,perm=[0,2,1])
    vlad = tf.subtract(vlad,a)
        
    vlad = tf.nn.l2_normalize(vlad,1)

    vlad = tf.reshape(vlad,[-1, cluster_size*feature_size])
    vlad = tf.nn.l2_normalize(vlad,1)
    tf.summary.histogram("vlad", vlad)

    hidden1_weights = tf.get_variable("hidden1_weights",
      [cluster_size*feature_size, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
    #tf.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(vlad, hidden1_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    activation = tf.nn.relu(activation)
    tf.summary.histogram("hidden1_output", activation)

    carry_weights = tf.get_variable("carry_weights",
      [hidden1_size, hidden1_size],
      initializer = tf.random_normal_initializer(
      stddev=1 / math.sqrt(hidden1_size)))

    xform_weights = tf.get_variable("xform_weights",
      [hidden1_size, hidden1_size],
      initializer = tf.random_normal_initializer(
      stddev=1 / math.sqrt(hidden1_size)))

    hidden_weights = tf.get_variable("hidden_weights",
      [hidden1_size, hidden1_size],
      initializer = tf.random_normal_initializer(
      stddev=1 / math.sqrt(hidden1_size)))

    carry = tf.matmul(activation, carry_weights)
    xform = tf.matmul(activation, xform_weights)
    hidden = tf.matmul(activation, hidden_weights)
    tf.summary.histogram("carry", carry)
    tf.summary.histogram("xform", xform)

    if add_batch_norm:
      carry = slim.batch_norm(
        carry,
        center=True,
        scale=True,
        is_training=is_training,
        scope="carry_bn")
      xform = slim.batch_norm(
        xform,
        center=True,
        scale=True,
        is_training=is_training,
        scope="xform_bn")
      hidden = slim.batch_norm(
        hidden,
        center=True,
        scale=True,
        is_training=is_training,
        scope="hidden_bn")
    else:
      carry_biases = tf.get_variable("carry_biases",
        [input_dim],
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(input_dim)))
      carry += carry_biases
      xform_biases = tf.get_variable("xform_biases",
        [input_dim],
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(input_dim)))
      xform += carry_biases
      hidden_biases = tf.get_variable("hidden_biases",
        [input_dim],
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(input_dim)))
      hidden += hidden_biases

    carry = tf.sigmoid(carry)
    xform = tf.sigmoid(xform)
    hidden = tf.nn.relu(hidden)

    activation = tf.multiply(activation, carry) + tf.multiply(hidden, xform)
    tf.summary.histogram("activation", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        **unused_params)

class NetVladLongHighwayModel(models.BaseModel):
  """TODO: Description
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.vlad_cluster_size
    hidden1_size = hidden_size or FLAGS.vlad_hidden_size

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    cluster_weights = tf.get_variable("cluster_weights",
                                      [feature_size, cluster_size],
                                      initializer = tf.random_normal_initializer(
                                        stddev=1 / math.sqrt(feature_size)))
        
    activation = tf.matmul(reshaped_input, cluster_weights)
        
    if add_batch_norm:
      activation = slim.batch_norm(
        activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="cluster_bn")
    else:
      cluster_biases = tf.get_variable("cluster_biases",
                                       [cluster_size],
                                       initializer = tf.random_normal_initializer(
                                         stddev=1 / math.sqrt(feature_size)))
      activation += cluster_biases
          
    activation = tf.nn.softmax(activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])

    a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

    cluster_weights2 = tf.get_variable("cluster_weights2",
                                       [1, feature_size, cluster_size],
                                       initializer = tf.random_normal_initializer(
                                         stddev=1 / math.sqrt(feature_size)))
        
    a = tf.multiply(a_sum,cluster_weights2)
        
    activation = tf.transpose(activation,perm=[0,2,1])
        
    reshaped_input = tf.reshape(reshaped_input,[-1, max_frames, feature_size])

    vlad = tf.matmul(activation,reshaped_input)
    vlad = tf.transpose(vlad,perm=[0,2,1])
    vlad = tf.subtract(vlad,a)
        
    vlad = tf.nn.l2_normalize(vlad,1)

    vlad = tf.reshape(vlad,[-1, cluster_size*feature_size])
    vlad = tf.nn.l2_normalize(vlad,1)
    tf.summary.histogram("vlad", vlad)

    hidden1_weights = tf.get_variable("hidden1_weights",
      [cluster_size*feature_size, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
    activation = tf.matmul(vlad, hidden1_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    activation = tf.nn.relu(activation)
    tf.summary.histogram("hidden1_output", activation)

    for layer in range(4):
      carry_weights = tf.get_variable("carry_weights_%d" % layer,
                                      [hidden1_size, hidden1_size],
                                      initializer = tf.random_normal_initializer(
                                        stddev=1 / math.sqrt(hidden1_size)))
      hidden_weights = tf.get_variable("hidden_weights_%d" % layer,
                                       [hidden1_size, hidden1_size],
                                       initializer = tf.random_normal_initializer(
                                         stddev=1 / math.sqrt(hidden1_size)))
      carry_biases = tf.get_variable("carry_biases_%d" % layer,
                                     [hidden1_size],
                                     initializer = tf.constant_initializer(1.0))
      hidden_biases = tf.get_variable("hidden_biases_%d" % layer,
                                      [hidden1_size],
                                      initializer = tf.constant_initializer(0.1))

      carry = tf.sigmoid(tf.matmul(activation, carry_weights) + carry_biases)
      hidden = tf.nn.relu(tf.matmul(activation, hidden_weights) + hidden_biases)
      activation = tf.multiply(activation, carry) + tf.multiply(hidden, (1.0 - carry))

      tf.summary.histogram("carry_%d" % layer, carry)
      tf.summary.histogram("highway_%d" % layer, activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        **unused_params)

class NetVladInputHighwayModel(models.BaseModel):
  """TODO: Description
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.vlad_cluster_size
    hidden1_size = hidden_size or FLAGS.vlad_hidden_size

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    for layer in range(3):
      carry_weights = tf.get_variable("carry_weights_%d" % layer,
                                      [feature_size, feature_size],
                                      initializer = tf.random_normal_initializer(
                                        stddev=1 / math.sqrt(feature_size)))
      hidden_weights = tf.get_variable("hidden_weights_%d" % layer,
                                       [feature_size, feature_size],
                                       initializer = tf.random_normal_initializer(
                                         stddev=1 / math.sqrt(feature_size)))
      carry_biases = tf.get_variable("carry_biases_%d" % layer,
                                     [feature_size],
                                     initializer = tf.constant_initializer(1.0))
      hidden_biases = tf.get_variable("hidden_biases_%d" % layer,
                                      [feature_size],
                                      initializer = tf.constant_initializer(0.1))

      carry = tf.sigmoid(tf.matmul(reshaped_input, carry_weights) + carry_biases)
      hidden = tf.nn.relu(tf.matmul(reshaped_input, hidden_weights) + hidden_biases)
      reshaped_input = tf.multiply(reshaped_input, carry) + tf.multiply(hidden, (1.0 - carry))

      tf.summary.histogram("carry_%d" % layer, carry)
      tf.summary.histogram("highway_%d" % layer, reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    cluster_weights = tf.get_variable("cluster_weights",
                                      [feature_size, cluster_size],
                                      initializer = tf.random_normal_initializer(
                                        stddev=1 / math.sqrt(feature_size)))
        
    activation = tf.matmul(reshaped_input, cluster_weights)
        
    if add_batch_norm:
      activation = slim.batch_norm(
        activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="cluster_bn")
    else:
      cluster_biases = tf.get_variable("cluster_biases",
                                       [cluster_size],
                                       initializer = tf.random_normal_initializer(
                                         stddev=1 / math.sqrt(feature_size)))
      activation += cluster_biases
          
    activation = tf.nn.softmax(activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])

    a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

    cluster_weights2 = tf.get_variable("cluster_weights2",
                                       [1, feature_size, cluster_size],
                                       initializer = tf.random_normal_initializer(
                                         stddev=1 / math.sqrt(feature_size)))
        
    a = tf.multiply(a_sum,cluster_weights2)
        
    activation = tf.transpose(activation,perm=[0,2,1])
        
    reshaped_input = tf.reshape(reshaped_input,[-1, max_frames, feature_size])

    vlad = tf.matmul(activation,reshaped_input)
    vlad = tf.transpose(vlad,perm=[0,2,1])
    vlad = tf.subtract(vlad,a)
        
    vlad = tf.nn.l2_normalize(vlad,1)

    vlad = tf.reshape(vlad,[-1, cluster_size*feature_size])
    vlad = tf.nn.l2_normalize(vlad,1)
    tf.summary.histogram("vlad", vlad)

    hidden1_weights = tf.get_variable("hidden1_weights",
      [cluster_size*feature_size, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
    activation = tf.matmul(vlad, hidden1_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    activation = tf.nn.relu(activation)
    tf.summary.histogram("hidden1_output", activation)

    carry_weights = tf.get_variable("carry_weights",
      [hidden1_size, hidden1_size],
      initializer = tf.random_normal_initializer(
      stddev=1 / math.sqrt(hidden1_size)))

    xform_weights = tf.get_variable("xform_weights",
      [hidden1_size, hidden1_size],
      initializer = tf.random_normal_initializer(
      stddev=1 / math.sqrt(hidden1_size)))

    hidden_weights = tf.get_variable("hidden_weights",
      [hidden1_size, hidden1_size],
      initializer = tf.random_normal_initializer(
      stddev=1 / math.sqrt(hidden1_size)))

    carry = tf.matmul(activation, carry_weights)
    xform = tf.matmul(activation, xform_weights)
    hidden = tf.matmul(activation, hidden_weights)

    if add_batch_norm:
      carry = slim.batch_norm(
        carry,
        center=True,
        scale=True,
        is_training=is_training,
        scope="carry_bn")
      xform = slim.batch_norm(
        xform,
        center=True,
        scale=True,
        is_training=is_training,
        scope="xform_bn")
      hidden = slim.batch_norm(
        hidden,
        center=True,
        scale=True,
        is_training=is_training,
        scope="hidden_bn")
    else:
      carry_biases = tf.get_variable("carry_biases",
        [input_dim],
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(input_dim)))
      carry += carry_biases
      xform_biases = tf.get_variable("xform_biases",
        [input_dim],
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(input_dim)))
      xform += carry_biases
      hidden_biases = tf.get_variable("hidden_biases",
        [input_dim],
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(input_dim)))
      hidden += hidden_biases

    carry = tf.sigmoid(carry)
    xform = tf.sigmoid(xform)
    hidden = tf.nn.relu(hidden)
    tf.summary.histogram("carry", carry)
    tf.summary.histogram("xform", xform)

    activation = tf.multiply(activation, carry) + tf.multiply(hidden, xform)
    tf.summary.histogram("activation", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        **unused_params)

