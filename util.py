#-*- coding: utf-8 -*-
import collections
import multiprocessing
import time
from collections import defaultdict

import six


def create_training_instances(seq_by_user,
                              max_seq_length,
                              dupe_factor,
                              short_seq_prob,
                              masked_lm_prob,
                              max_predictions_per_seq,
                              vocab,
                              mask_prob,
                              prop_sliding_window,
                              rng,
                              pool_size,
                              force_last=False):
  """Create `TrainingInstance`s from raw text.
    all_documents : item sequence of each user.
  """
  # seq_by_user = {}  # already dictionary

  if force_last:
    for user, item_seq in seq_by_user.items():
      if len(item_seq) == 0:
        print("got empty seq:" + user)
        continue
      seq_by_user[user] = [item_seq[-max_seq_length:]]
  else:
    max_num_tokens = max_seq_length  # we need two sentence

    sliding_step = (int)(
      prop_sliding_window *
      max_num_tokens) if prop_sliding_window != -1.0 else max_num_tokens
    for user, item_seq in all_documents_raw.items():
      if len(item_seq) == 0:
        print("got empty seq:" + user)
        continue

      # todo: add slide
      if len(item_seq) <= max_num_tokens:
        all_documents[user] = [item_seq]
      else:
        beg_idx = range(len(item_seq) - max_num_tokens, 0, -sliding_step)
        beg_idx.append(0)
        all_documents[user] = [item_seq[i:i + max_num_tokens] for i in beg_idx[::-1]]

  instances = []
  if force_last:
    for user in seq_by_user:
      instances.extend(
        create_instances_from_document_test(
          seq_by_user, user, max_seq_length))
    print("num of instance:{}".format(len(instances)))
  else:
    start_time = time.clock()
    pool = multiprocessing.Pool(processes=pool_size)
    instances = []
    print("document num: {}".format(len(all_documents)))

    def log_result(result):
      print("callback function result type: {}, size: {} ".format(type(result), len(result)))
      instances.extend(result)

    for step in range(dupe_factor):
      pool.apply_async(
        create_instances_threading, args=(
          all_documents, max_seq_length, short_seq_prob,
          masked_lm_prob, max_predictions_per_seq, vocab, mask_prob, step),
          callback=log_result)
    pool.close()
    pool.join()

    for user in all_documents:
      instances.extend(
        mask_last(
          all_documents, user, max_seq_length, short_seq_prob, vocab))

    print("num of instance:{}; time:{}".format(len(instances), time.clock() - start_time))
  rng.shuffle(instances)
  return instances


class BertTrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, info, tokens, masked_lm_positions, masked_lm_labels):
      self.info = info  # info = [user]
      self.tokens = tokens
      self.masked_lm_positions = masked_lm_positions
      self.masked_lm_labels = masked_lm_labels

    def __str__(self):
      s = ""
      s += "info: %s\n" % (" ".join([printable_text(x) for x in self.info]))
      s += "tokens: %s\n" % (
        " ".join([printable_text(x) for x in self.tokens]))
      s += "masked_lm_positions: %s\n" % (
        " ".join([str(x) for x in self.masked_lm_positions]))
      s += "masked_lm_labels: %s\n" % (
        " ".join([printable_text(x) for x in self.masked_lm_labels]))
      s += "\n"
      return s

    def __repr__(self):
      return self.__str__()


def mask_last(
      all_documents, user, max_seq_length, short_seq_prob, vocab):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[user]
    max_num_tokens = max_seq_length

    instances = []
    info = [int(user.split("_")[1])]

    for tokens in document:
      assert len(tokens) >= 1 and len(tokens) <= max_num_tokens

      (tokens, masked_lm_positions,
       masked_lm_labels) = create_masked_lm_predictions_force_last(tokens)
      instance = BertTrainingInstance(
        info=info,
        tokens=tokens,
        masked_lm_positions=masked_lm_positions,
        masked_lm_labels=masked_lm_labels)
      instances.append(instance)

    return instances


def create_instances_from_document_test(seq_by_user, user, max_seq_length):
  """Creates `TrainingInstance`s for a single document."""
  seq = seq_by_user[user]
  max_num_tokens = max_seq_length

  assert len(seq) == 1 and len(seq[0]) <= max_num_tokens

  tokens = seq[0]
  assert len(tokens) >= 1

  (tokens, masked_lm_positions,
   masked_lm_labels) = create_masked_lm_predictions_force_last(tokens)

  info = [int(user.split("_")[1])]
  instance = BertTrainingInstance(
    info=info,
    tokens=tokens,
    masked_lm_positions=masked_lm_positions,
    masked_lm_labels=masked_lm_labels)

  return [instance]


def create_instances_from_document_train(
    all_documents, user, max_seq_length, short_seq_prob, masked_lm_prob,
    max_predictions_per_seq, vocab, rng, mask_prob):
  """Creates `TrainingInstance`s for a single document."""
  document = all_documents[user]

  max_num_tokens = max_seq_length

  instances = []
  info = [int(user.split("_")[1])]
  vocab_items = vocab.get_items()

  for tokens in document:
    assert len(tokens) >= 1 and len(tokens) <= max_num_tokens

    (tokens, masked_lm_positions,
     masked_lm_labels) = create_masked_lm_predictions(
      tokens, masked_lm_prob, max_predictions_per_seq,
      vocab_items, rng, mask_prob)
    instance = BertTrainingInstance(
      info=info,
      tokens=tokens,
      masked_lm_positions=masked_lm_positions,
      masked_lm_labels=masked_lm_labels)
    instances.append(instance)

  return instances


def create_masked_lm_predictions_force_last(tokens):
    """Creates the predictions for the masked LM objective."""

    last_index = -1
    for (i, token) in enumerate(tokens):
      if token == "[CLS]" or token == "[PAD]" or token == '[NO_USE]':
        continue
      last_index = i

    assert last_index > 0

    output_tokens = list(tokens)  # already list?
    output_tokens[last_index] = "[MASK]"

    masked_lm_positions = [last_index]
    masked_lm_labels = [tokens[last_index]]

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng,
                                 mask_prob):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token not in vocab_words:
      continue
    cand_indexes.append(i)

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    if index in covered_indexes:
      continue
    covered_indexes.add(index)

    masked_token = None
    # 80% of the time, replace with [MASK]
    if rng.random() < mask_prob:
      masked_token = "[MASK]"
    else:
      # 10% of the time, keep original
      if rng.random() < 0.5:
        masked_token = tokens[index]
      # 10% of the time, replace with random word
      else:
        # masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
        masked_token = rng.choice(vocab_words)

    output_tokens[index] = masked_token

    masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_instances_threading(all_documents, max_seq_length, short_seq_prob,
                               masked_lm_prob, max_predictions_per_seq, vocab, rng,
                               mask_prob, step):
  cnt = 0
  start_time = time.clock()
  instances = []
  for user in all_documents:
    cnt += 1
    if cnt % 1000 == 0:
      print("step: {}, name: {}, step: {}, time: {}".format(step, multiprocessing.current_process().name, cnt,
                                                            time.clock() - start_time))
      start_time = time.clock()
    instances.extend(create_instances_from_document_train(
      all_documents, user, max_seq_length, short_seq_prob,
      masked_lm_prob, max_predictions_per_seq, vocab, rng,
      mask_prob))

  return instances


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
      if isinstance(text, str):
        return text
      elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
      else:
        raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
      if isinstance(text, str):
        return text
      elif isinstance(text, unicode):
        return text.encode("utf-8")
      else:
        raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
      raise ValueError("Not running on Python2 or Python 3?")


def data_partition(fname):
  usernum = 0
  itemnum = 0
  User = defaultdict(list)
  user_train = {}
  user_valid = {}
  user_test = {}
  # assume user/item index starting from 1
  f = open(fname, 'r')
  for line in f:
    u, i = line.rstrip().split(' ')
    u = int(u)
    i = int(i)
    usernum = max(u, usernum)
    itemnum = max(i, itemnum)
    User[u].append(i)

  for user in User:
    nfeedback = len(User[user])
    if nfeedback < 3:
      user_train[user] = User[user]
      user_valid[user] = []
      user_test[user] = []
    else:
      user_train[user] = User[user][:-2]
      user_valid[user] = []
      user_valid[user].append(User[user][-2])
      user_test[user] = []
      user_test[user].append(User[user][-1])
  return [user_train, user_valid, user_test, usernum, itemnum]