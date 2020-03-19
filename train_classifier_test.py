#!/usr/bin/env python
# coding: utf-8

# In[22]:


import os
import sys
import json
import datetime
import pprint
import tensorflow as tf

import modeling
import optimization
import tokenization
import run_classifier
import classifier_util as util


# We will use base uncased bert model, you can give try with large models
# For large model TPU is necessary
#BERT_MODEL = 'uncased_L-12_H-768_A-12' #@param {type:"string"}

# BERT checkpoint bucket
BERT_PRETRAINED_DIR = '/Users/nirahurk/Documents/PersonalDocs/Learning/Stanford/CS224n/Project/multi_cased_L-12_H-768_A-12'
print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))


# In[23]:


# Model Hyper Parameters
TRAIN_BATCH_SIZE = 12 # For GPU, reduce to 16
EVAL_BATCH_SIZE = 8
PREDICT_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 2.0
WARMUP_PROPORTION = 0.1

# Model configs
SAVE_CHECKPOINTS_STEPS = 1000
ITERATIONS_PER_LOOP = 1000
NUM_TPU_CORES = 8
VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
DO_LOWER_CASE = False


# In[24]:


train_file = '/Users/nirahurk/Documents/PersonalDocs/Learning/Stanford/CS224n/Project/tydiqa-goldp-dev-v1.0b.json'
label_file = '/Users/nirahurk/Documents/PersonalDocs/Learning/Stanford/CS224n/Project/labels-classifier-qa-f1.json'
OUTPUT_DIR = '/Users/nirahurk/Documents/PersonalDocs/Learning/Stanford/CS224n/Project/output'
labels = []
with open(label_file) as f:
    labels = json.load(f)


# In[25]:


print("################  Processing Training Data #####################")

label_list = [0,1]
tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)
train_examples = util.read_classifier_examples(train_file, labels, is_training=True)
num_train_examples = len(train_examples)
train_batch_size = 12
num_train_epochs = 2.0
warmup_proportion = 0.1
num_train_steps = int(len(train_examples) / train_batch_size * num_train_epochs)
num_warmup_steps = int(num_train_steps * warmup_proportion)
MAX_SEQ_LENGTH = 384
# Converting training examples to features
TRAIN_TF_RECORD = os.path.join(OUTPUT_DIR, "train.tf_record")
run_classifier.file_based_convert_examples_to_features(train_examples, label_list, MAX_SEQ_LENGTH, tokenizer, TRAIN_TF_RECORD)


# In[26]:


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  # Bert Model instant 
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # Getting output for last layer of BERT
  output_layer = model.get_pooled_output()
  
  # Number of outputs for last layer
  hidden_size = output_layer.shape[-1].value
  
  # We will use one layer on top of BERT pretrained for creating classification model
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    
    # Calcaulte prediction probabilites and loss
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)


# In[27]:


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  
    """The `model_fn` for TPUEstimator."""

    # reading features input
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
    
    # checking if training mode
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    
    # create simple classification model
    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)
    
    # getting variables for intialization and using pretrained init checkpoint
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      # defining optimizar function
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
      
      # Training estimator spec
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:
      # accuracy, loss, auc, F1, precision and recall metrics for evaluation
      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        f1_score = tf.contrib.metrics.f1_score(
            label_ids,
            predictions)
        auc = tf.metrics.auc(
            label_ids,
            predictions)
        recall = tf.metrics.recall(
            label_ids,
            predictions)
        precision = tf.metrics.precision(
            label_ids,
            predictions) 
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
            "f1_score": f1_score,
            "auc": auc,
            "precision": precision,
            "recall": recall
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      # estimator spec for evalaution
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      # estimator spec for predictions
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# In[28]:


# Define TPU configs
tpu_cluster_resolver = None
run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=OUTPUT_DIR,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=ITERATIONS_PER_LOOP,
        num_shards=NUM_TPU_CORES,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))


# In[29]:


# create model function for estimator using model function builder
model_fn = model_fn_builder(
    bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
    num_labels=len(label_list),
    init_checkpoint=INIT_CHECKPOINT,
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=False,
    use_one_hot_embeddings=True)


# In[20]:


# Defining TPU Estimator
estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=False,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    predict_batch_size=PREDICT_BATCH_SIZE)


# In[ ]:


# Train the model.
print('QQP on BERT base model normally takes about 1 hour on TPU and 15-20 hours on GPU. Please wait...')
print('***** Started training at {} *****'.format(datetime.datetime.now()))
print('  Num examples = {}'.format(num_train_examples))
print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
tf.logging.info("  Num steps = %d", num_train_steps)
# we are using `file_based_input_fn_builder` for creating input function from TF_RECORD file
train_input_fn = run_classifier.file_based_input_fn_builder(TRAIN_TF_RECORD,
                                                            seq_length=MAX_SEQ_LENGTH,
                                                            is_training=True,
                                                            drop_remainder=True)
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print('***** Finished training at {} *****'.format(datetime.datetime.now()))


# In[ ]:




