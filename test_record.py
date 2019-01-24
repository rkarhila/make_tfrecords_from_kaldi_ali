#/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tf.enable_eager_execution()

from scipy.io import wavfile

import numpy as np
import IPython.display as display
import subprocess

import sys

if len(sys.argv) > 1:
  utterance_filenames = [ sys.argv[1] ]
else:
  utterance_filenames = ['/teamwork/t40511_asr/scratch/rkarhila/corpora_for_new_siak/new_tfrecords/utterance_collection.pfstar_enuk____train_clean_wav__________.1.50s.tfrecord.1']

if len(sys.argv) > 2:
  phoneme_filenames = [sys.argv[2] ]
else:
  phoneme_filenames = ['/teamwork/t40511_asr/scratch/rkarhila/corpora_for_new_siak/new_tfrecords/phoneme_collection.tidigits_______children_test_wav________.0.75s.tfrecord.0']


raw_utterance_dataset = tf.data.TFRecordDataset(utterance_filenames)

#for raw_record in raw_dataset.take(1):
#  print(repr(raw_record))

#for raw_record in raw_dataset.take(10):
#  print(repr(raw_record))

# Create a description of the features.  
utterance_feature_description = {
  'sourcefile' : tf.FixedLenFeature([], tf.string, default_value=''),
  'lng' : tf.FixedLenFeature([], tf.string, default_value=''),
  'wordseq' :  tf.FixedLenFeature([], tf.string, default_value=''),
  'phoneseq' : tf.FixedLenFeature([], tf.string, default_value=''),
  'classseq' : tf.FixedLenSequenceFeature(([]), tf.int64,  allow_missing=True),# tf.VarLenFeature(tf.int64),
  'starttime' : tf.FixedLenFeature([], tf.float32, default_value=0.0),
  'duration' : tf.FixedLenFeature([], tf.float32, default_value=0.0),
  'gender' : tf.FixedLenFeature([], tf.string, default_value=''),
  'age' : tf.FixedLenFeature([], tf.int64, default_value=0),
  'endsillength' : tf.FixedLenFeature([], tf.float32, default_value=0.0),
  'bucketlength' :tf.FixedLenFeature([], tf.float32, default_value=0.0),
  'audio' : tf.FixedLenSequenceFeature(([]), tf.float32, allow_missing=True ), #tf.VarLenFeature(tf.float32),
  'alignment' : tf.FixedLenSequenceFeature(([]), tf.float32, allow_missing=True ), #tf.VarLenFeature(tf.float32),
}

def _utterance_parse_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.parse_single_example(example_proto, utterance_feature_description)

parsed_utterance_dataset = raw_utterance_dataset.map(_utterance_parse_function)

if True:
  for record in parsed_utterance_dataset.take(10):
    print("======================================")
    print(record['wordseq'].numpy().decode('utf-8'))
    print(record['phoneseq'].numpy().decode('utf-8'))
    print(record['duration'].numpy())
    print(record['alignment'].numpy().reshape([-1,3]))
    a = record['audio'].numpy()
    wavfile.write('/tmp/python-audio.wav', 16000,a)
    subprocess.call(["play", '/tmp/python-audio.wav', 'norm', '-10'])




    
raw_phone_dataset = tf.data.TFRecordDataset(phone_filenames)

#for raw_record in raw_phone_dataset.take(1):
#  print(repr(raw_record))

#for raw_record in raw_dataset.take(10):
#  print(repr(raw_record))

# Create a description of the features.  
phone_feature_description = {
  'sourcefile' : tf.FixedLenFeature([], tf.string, default_value=''),
  'starttime' : tf.FixedLenFeature([], tf.float32, default_value=0.0),
  'duration' : tf.FixedLenFeature([], tf.float32, default_value=0.0),
  #'class' : tf.FixedLenFeature([], tf.int64, default_value=0.0),
  'gender' : tf.FixedLenFeature([], tf.string, default_value=''),
  'phone' : tf.FixedLenFeature([], tf.string, default_value=''),
  #'age' : tf.FixedLenFeature([], tf.int64, default_value=0),
  'audio' : tf.FixedLenSequenceFeature(([]), tf.float32, allow_missing=True ), #tf.VarLenFeature(tf.float32),
  'end_padding_samples' : tf.FixedLenFeature([], tf.int64, default_value=0),
  'start_padding_samples' : tf.FixedLenFeature([], tf.int64, default_value=0),
}

def _phone_parse_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.parse_single_example(example_proto, phone_feature_description)

parsed_phone_dataset = raw_phone_dataset.map(_phone_parse_function)

for record in parsed_phone_dataset.take(1000):
  phone = record['phone'].numpy().decode('utf-8')
  if phone[:2] == 'əʊ' or phone[:2] == 'i_':
    print("======================================")
    print(phone, "Duration", record['duration'].numpy())
    print(len(record['audio'].numpy()), record['start_padding_samples'].numpy(),record['end_padding_samples'].numpy())
    a = record['audio'].numpy()[ record['start_padding_samples'].numpy(): -record['end_padding_samples'].numpy()]
    wavfile.write('/tmp/python-audio.wav', 16000,a)
    subprocess.call(["play", '/tmp/python-audio.wav', 'norm', '-10'])

