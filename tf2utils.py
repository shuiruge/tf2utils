# Utils for TensorFlow. Modified from https://www.tensorflow.org/guide/core/mlp_core

import tensorflow as tf


def xavier_init(shape):
  # Computes the xavier initialization values for a weight matrix
  in_dim, out_dim = shape
  xavier_lim = tf.sqrt(6.)/tf.sqrt(tf.cast(in_dim + out_dim, tf.float32))
  weight_vals = tf.random.uniform(shape=(in_dim, out_dim),
                                  minval=-xavier_lim, maxval=xavier_lim)
  return weight_vals


class DenseLayer(tf.Module):

  def __init__(self, out_dim, weight_init=xavier_init, activation=tf.identity):
    # Initialize the dimensions and activation functions
    self.out_dim = out_dim
    self.weight_init = weight_init
    self.activation = activation
    self.built = False

  def __call__(self, x):
    if not self.built:
      # Infer the input dimension based on first call
      self.in_dim = x.shape[1]
      # Initialize the weights and biases
      self.W = tf.Variable(self.weight_init(shape=(self.in_dim, self.out_dim)))
      self.b = tf.Variable(tf.zeros(shape=(self.out_dim,)))
      self.built = True
    # Compute the forward pass
    z = tf.add(tf.matmul(x, self.W), self.b)
    return self.activation(z)


def cross_entropy_loss(logits, labels):
  """Compute cross entropy loss with a sparse operation."""
  sparse_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
  return tf.reduce_mean(sparse_ce)


def mse_loss(y_pred, y_true):
  """Compute mean square error loss."""
  return tf.reduce_mean(tf.square(y_true - y_pred))


def accuracy(y_pred, y_true):
  """Compute accuracy after extracting class predictions."""
  class_preds = tf.argmax(tf.nn.softmax(y_pred), axis=1)
  is_equal = tf.equal(y_true, class_preds)
  return tf.reduce_mean(tf.cast(is_equal, tf.float32))


class Adam(tf.Module):

  def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, ep=1e-7):
    # Initialize the Adam parameters
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.learning_rate = learning_rate
    self.ep = ep
    self.t = 1.
    self.v_dvar, self.s_dvar = [], []
    self.built = False

  def apply_gradients(self, grads, vars):
    # Set up moment and RMSprop slots for each variable on the first call
    if not self.built:
      for var in vars:
        v = tf.Variable(tf.zeros(shape=var.shape))
        s = tf.Variable(tf.zeros(shape=var.shape))
        self.v_dvar.append(v)
        self.s_dvar.append(s)
      self.built = True
    # Perform Adam updates
    for i, (d_var, var) in enumerate(zip(grads, vars)):
      # Moment calculation
      self.v_dvar[i] = self.beta_1*self.v_dvar[i] + (1-self.beta_1)*d_var
      # RMSprop calculation
      self.s_dvar[i] = self.beta_2*self.s_dvar[i] + (1-self.beta_2)*tf.square(d_var)
      # Bias correction
      v_dvar_bc = self.v_dvar[i]/(1-(self.beta_1**self.t))
      s_dvar_bc = self.s_dvar[i]/(1-(self.beta_2**self.t))
      # Update model variables
      var.assign_sub(self.learning_rate*(v_dvar_bc/(tf.sqrt(s_dvar_bc) + self.ep)))
    # Increment the iteration counter
    self.t += 1.


def train_step(x_batch, y_batch, loss, model, optimizer, accuracy=None):
  # Update the model state given a batch of data
  with tf.GradientTape() as tape:
    y_pred = model(x_batch)
    batch_loss = loss(y_pred, y_batch)
  batch_acc = accuracy(y_pred, y_batch) if accuracy else None
  grads = tape.gradient(batch_loss, model.variables)
  optimizer.apply_gradients(grads, model.variables)
  return batch_loss, batch_acc


def val_step(x_batch, y_batch, loss, model, accuracy=None):
  # Evaluate the model on given a batch of validation data
  y_pred = model(x_batch)
  batch_loss = loss(y_pred, y_batch)
  batch_acc = accuracy(y_pred, y_batch) if accuracy else None
  return batch_loss, batch_acc

