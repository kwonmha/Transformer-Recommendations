#-*- coding: utf-8 -*-
import copy
import json

import tensorflow as tf
from util import *


def sasrec_model_fn():
  return


def sasrec_input_fn():
  return


def bert4rec_model_fn():
  return


def bert4rec_input_fn():
  return


class SasrecTrainingInstance(object):


  def __init__(self, seq):
    self.seq = seq


class Bert4recTrainingInstance(object):


  def __init__(self, seq):
    self.seq = seq


model_fn_map = {'sasrec': sasrec_model_fn,
                'bert4rec': bert4rec_model_fn}

input_fn_map = {'sasrec': sasrec_input_fn,
                'bert4rec': bert4rec_input_fn}

training_instance = {'sasrec': SasrecTrainingInstance,
                     'bert4rec': Bert4recTrainingInstance}


class Model:

  def __init__(self, FLAGS):
    self.type = FLAGS.type

    self.pool_size = FLAGS.pool_size
    self.max_seq_length = FLAGS.max_seq_length

    self.record_file = self.type + "+ml" + self.max_seq_length
    self.input_fn = None

    # 공유 가능?
    self.config = ModelConfig.from_json_file(FLAGS.config_file)

    # is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    self.run_config = tf.estimator.RunConfig(
      model_dir=FLAGS.checkpointDir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps)


  def store_data_as_record(self, data, rng):
    pass

  def set_input_fn(self):
    self.input_fn = input_fn_map[self.type](is_training=)

  def train(self):
    self.estimator.train(input_fn=self.input_fn)

  def predict(self):
    self.estimator.evaluate(input_fn=self.input_fn)


class SASRec(Model):

  def __init__(self, FLAGS):
    super().__init__(FLAGS)
    self.model_fn = model_fn_map[self.type]()
    self.instance = training_instance[self.type]

    self.estimator = tf.estimator.Estimator(model_fn=)

  def store_data_as_record(self, data, rng):
    user_seqs_train, user_valid, user_test, usernum, itemnum = data
    instances = self._create_sasrec_training_instances(
        user_seqs_train, self.max_seq_length, rng)

    tf.logging.info("*** Writing to output files ***")
    tf.logging.info("  %s", output_filename)

    write_sasrec_instance_to_example_files(instances)

  def _create_sasrec_training_instances(self, user_seqs_train, max_seq_length, rng):
    instances = []

    rng.shuffle(instances)
    return instances

  def _sample_function(self, user_seqs, ):


class BERT4Rec(Model):

  def __init__(self, FLAGS):
    super().__init__(FLAGS)
    max_predictions_per_seq = FLAGS.max_predictions_per_seq
    masked_lm_prob = FLAGS.masked_lm_prob
    mask_prob = FLAGS.mask_prob
    dupe_factor = FLAGS.dupe_factor
    prop_sliding_window = FLAGS.prop_sliding_window

    self.record_file += "mxp" + max_predictions_per_seq +\
                        "mlmp" + masked_lm_prob +\
                        "mp" + mask_prob +\
                        "dup" + dupe_factor

    model_fn = model_fn_map[self.type](
                                  bert_config=bert_config,
                                  init_checkpoint=FLAGS.init_checkpoint,
                                  learning_rate=FLAGS.learning_rate,
                                  num_train_steps=FLAGS.num_train_steps,
                                  num_warmup_steps=FLAGS.num_warmup_steps,
                                  use_tpu=FLAGS.use_tpu,
                                  use_one_hot_embeddings=FLAGS.use_tpu,
                                  item_size=item_size)

    self.estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=self.run_config,
        params={
            "batch_size": FLAGS.batch_size
        })

  def store_data_as_record(self, data, rng):
    user_train, user_valid, user_test, usernum, itemnum = data
    instances = create_training_instances(
        data, max_seq_length, dupe_factor, short_seq_prob, masked_lm_prob,
        max_predictions_per_seq, vocab, mask_prob, prop_sliding_window,
        rng, pool_size, force_last)

    tf.logging.info("*** Writing to output files ***")
    tf.logging.info("  %s", output_filename)

    write_instance_to_example_files(instances, max_seq_length,
                                    max_predictions_per_seq, vocab,
                                    [output_filename])


class ModelConfig(object):
  """Configuration for `Model`."""

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):
    """Constructs BertConfig.

    Args:
    vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
    hidden_size: Size of the encoder layers and the pooler layer.
    num_hidden_layers: Number of hidden layers in the Transformer encoder.
    num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
    intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
    hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
    hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
    max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
    type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
    initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = ModelConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"