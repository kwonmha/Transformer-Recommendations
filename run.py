#-*- coding: utf-8 -*-
import os

import tensorflow as tf
from models import SASRec, BERT4Rec
from util import data_partition


random_seed = 12345

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "pool_size", 10,
    "multiprocesses pool size.")

flags.DEFINE_integer(
    "max_seq_length", 200,
    "max sequence length.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "max_predictions_per_seq.")

flags.DEFINE_float(
    "masked_lm_prob", 0.15,
    "Masked LM probability.")

flags.DEFINE_float(
    "mask_prob", 1.0,
    "mask probabaility")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_string(
    "data_dir", './data/',
    "data dir.")

flags.DEFINE_string(
    "output_dir", './result/',
    "output dir.")

flags.DEFINE_string(
    "dataset_name", 'ml-1m',
    "dataset name.")


model_map = {'sasrec': SASRec,
             'bert4rec': BERT4Rec}




# ===========================
# config_file, data_file, do_train, do_eval

def main():
  model_type = FLAGS.type
  data_dir = FLAGS.data_dir
  datafile = FLAGS.datafile
  output_dir = FLAGS.output_dir

  if not os.path.isdir(data_dir):
    print(data_dir + ' is not exist in {}'.format(os.getcwd()))
    exit(1)

  model = model_map[model_type](FLAGS)

  # if tf record exists, pass
  if not os.path.isfile(model.record_file):
    dataset = data_partition(output_dir + datafile)
    model.store_data_as_record(dataset)

  if FLAGS.do_train:
    model.train()
  else:
    model.predict()


if __name__ == "__main__":
    main()

