import os
import pathlib
import pickle
from collections import namedtuple
from datetime import datetime

import annoy
import apache_beam as beam
import numpy as np
import sklearn
from sklearn import *
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
import tempfile
import itertools

encoder = None
module_url = 'https://tfhub.dev/google/universal-sentence-encoder/2'  # @param {type:"string"}
projected_dim = 64  # @param {type:"number"}

def load_module(module_url):
  embed_module = hub.Module(module_url)
  placeholder = tf.placeholder(dtype=tf.string)
  embed = embed_module(placeholder)
  session = tf.Session()
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  print('TF-Hub module is loaded.')

  def _embeddings_fn(sentences):
    computed_embeddings = session.run(
        embed, feed_dict={placeholder: sentences})
    return computed_embeddings

  return _embeddings_fn


def generate_random_projection_weights(original_dim, projected_dim):
    random_projection_matrix = None
    if projected_dim and original_dim > projected_dim:
        random_projection_matrix = sklearn.random_projection._gaussian_random_matrix(
            n_components=projected_dim, n_features=original_dim).T
        print("A Gaussian random weight matrix was creates with shape of {}".format(random_projection_matrix.shape))
        print('Storing random projection matrix to disk...')
        with open('random_projection_matrix', 'wb') as handle:
            pickle.dump(random_projection_matrix,
                        handle, protocol=pickle.HIGHEST_PROTOCOL)

    return random_projection_matrix

#======================================================================
output_dir = pathlib.Path(tempfile.mkdtemp())
temporary_dir = pathlib.Path(tempfile.mkdtemp())
g = tf.Graph()
with g.as_default():
  original_dim = load_module(module_url)(['']).shape[1]
  random_projection_matrix = None

  if projected_dim:
    random_projection_matrix = generate_random_projection_weights(
        original_dim, projected_dim)

args = {
    'job_name': 'hub2emb-{}'.format(datetime.utcnow().strftime('%y%m%d-%H%M%S')),
    'runner': 'DirectRunner',
    'batch_size': 1024,
    'data_dir': 'corpus/*.txt',
    'output_dir': output_dir,
    'temporary_dir': temporary_dir,
    'module_url': module_url,
    'random_projection_matrix': random_projection_matrix,
}

print("Pipeline args are set.")
print(args)

