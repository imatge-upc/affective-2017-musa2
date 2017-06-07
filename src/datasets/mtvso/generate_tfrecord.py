# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Original code: https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py
Modified by Victor Campos

Converts image data to TFRecords file format with Example protos.
The image data set is given by a text file with the following structure:
    path1 label1_anp label1_noun label1_adj
    path2 label2_anp label2_noun label2_adj
    ...
    pathN labelN_anp labelN_noun labelN_adj

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of TFRecord files
    output_directory/name-00000-of-N_shards
    output_directory/name-N_shards-of-N_shards
    ...
    output_directory/name-N_shards-1-of-N_shards
Each record within the TFRecord file is a serialized Example proto. The
Example proto contains the following fields:
    image/encoded: string containing JPEG encoded image in RGB colorspace
    image/height: integer, image height in pixels
    image/width: integer, image width in pixels
    image/colorspace: string, specifying the colorspace, always 'RGB'
    image/channels: integer, specifying the number of channels, always 3
    image/format: string, specifying the format, always'JPEG'
    image/filename: string containing the basename of the image file
        e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
    image/class/label: list of integers specifying the indices in a classification layer.
    The label ranges from [0, num_labels-1] and are given as [anp noun adjective].
    image/class/text: string specifying the human-readable version of the labels
        e.g. 'dog'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string('images_directory', None, 'Image data directory')
tf.app.flags.DEFINE_string('input_file', None, 'Image data directory')
tf.app.flags.DEFINE_string('output_directory', None, 'Output data directory')
tf.app.flags.DEFINE_string('name', None, 'Name for the subset')
tf.app.flags.DEFINE_string('anp_list', None, 'File with the ANP labels')
tf.app.flags.DEFINE_string('noun_list', None, 'File with the Noun labels')
tf.app.flags.DEFINE_string('adj_list', None, 'File with the Adjective labels')

tf.app.flags.DEFINE_integer('num_shards', 2, 'Number of shards in training TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 2, 'Number of threads to preprocess the images.')

tf.app.flags.DEFINE_integer('anp_offset', 0, 'Label offset for ANPs.')
tf.app.flags.DEFINE_integer('noun_offset', 0, 'Label offset for Nouns.')
tf.app.flags.DEFINE_integer('adj_offset', 0, 'Label offset for Adjectives.')
tf.app.flags.DEFINE_boolean('anp_only', False, 'Encode only ANPs, setting Noun and Adj labels to -1.')


FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if isinstance(value, list):
        value = value[0]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, text, height, width):
    """Build an Example proto for an example.
    Args:
        filename: string, path to an image file, e.g., '/path/to/example.JPG'
        image_buffer: string, JPEG encoding of RGB image
        label: integer, identifier for the ground truth for the network
        text: string, unique human-readable, e.g. 'dog'
        height: integer, image height in pixels
        width: integer, image width in pixels
    Returns:
        Example proto
    """

    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/text': _bytes_feature(text),
      'image/format': _bytes_feature(image_format),
      'image/filename': _bytes_feature(os.path.basename(filename)),
      'image/encoded': _bytes_feature(image_buffer)}))
    return example


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _is_png(filename):
    """Determine if a file contains a PNG format image.
    Args:
    filename: string, path of the image file.
    Returns:
    boolean indicating if the image is a PNG.
    """
    return '.png' in filename


def _process_image(filename, coder):
    """Process a single image file.
    Args:
        filename: string, path to an image file e.g., '/path/to/example.JPG'.
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
        image_buffer: string, JPEG encoding of RGB image.
        height: integer, image height in pixels.
        width: integer, image width in pixels.
    """
    # Read the image file.
    image_data = tf.gfile.FastGFile(filename, 'r').read()

    # Convert any PNG to JPEG's for consistency.
    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards):
    """Processes and saves list of images as TFRecord in 1 thread.
    Args:
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
        thread_index: integer, unique batch to run index is within [0, len(ranges)).
        ranges: list of pairs of integers specifying ranges of each batches to
          analyze in parallel.
        name: string, unique identifier specifying the data set
        filenames: list of strings; each string is a path to an image file
        texts: list of strings; each string is human readable, e.g. 'dog'
        labels: list of integer; each integer identifies the ground truth
        num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in xrange(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            text = texts[i]

            image_buffer, height, width = _process_image(filename, coder)

            example = _convert_to_example(filename, image_buffer, label,
                                        text, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files(name, filenames, texts, labels, num_shards):
    """Process and save list of images as TFRecord of Example protos.
    Args:
        name: string, unique identifier specifying the data set
        filenames: list of strings; each string is a path to an image file
        texts: list of strings; each string is human readable, e.g. 'dog'
        labels: list of integer; each integer identifies the ground truth
        num_shards: integer number of shards for this data set.
    """
    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in xrange(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames,
                texts, labels, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()


def _find_image_files(input_file, anp_list_path, noun_list_path, adj_list_path, dataset_dir):
    """Build a list of all images files and labels in the data set.
    Args:
        input_file: path to the file listing (path, anp_label, noun_label, adj_label) tuples
        anp_list_path: path to the file with the class id -> class name mapping for ANPs
        noun_list_path: path to the file with the class id -> class name mapping for Nouns
        adj_list_path: path to the file with the class id -> class name mapping for Adjectives
    Returns:
        filenames: list of strings; each string is a path to an image file.
        texts: list of string tuples; each string is the tuple of classes, e.g. ('happy dog', 'happy', 'dog')
        labels: list of integer tuples; each tuple identifies the ground truth: (anp_id, noun_id, adj_id)
    """
    lines = [line.strip() for line in open(input_file, 'r')]
    anp_list = [line.strip() for line in open(anp_list_path, 'r')]
    if not FLAGS.anp_only:
        noun_list = [line.strip() for line in open(noun_list_path, 'r')]
        adj_list = [line.strip() for line in open(adj_list_path, 'r')]
    filenames = list()
    texts = list()
    labels = list()
    for line in lines:
        if FLAGS.anp_only:
            img, anp_id = line.split()
        else:
            img, anp_id, noun_id, adj_id = line.split()
        filenames.append(os.path.join(dataset_dir, img))
        if FLAGS.anp_only:
            labels.append([int(anp_id) + FLAGS.anp_offset, -1, -1])
            texts.append([anp_list[int(anp_id)], 'no_noun_label', 'no_adjective_label'])
        else:
            labels.append([int(anp_id)+FLAGS.anp_offset, int(noun_id)+FLAGS.noun_offset, int(adj_id)+FLAGS.adj_offset])
            texts.append([anp_list[int(anp_id)], noun_list[int(noun_id)], adj_list[int(adj_id)]])

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = range(len(filenames))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Found %d JPEG files.' % len(filenames))

    return filenames, texts, labels


def _process_dataset(name, input_file, dataset_dir, num_shards, anp_list_path, noun_list_path, adj_list_path):
    """Process a complete data set and save it as a TFRecord.
    Args:
        name: string, unique identifier specifying the data set.
        input_file: path to the file listing (path, anp_label, noun_label, adj_label) tuples
        num_shards: integer number of shards for this data set.
        anp_list_path: string, path to the labels file.
        noun_list_path: string, path to the labels file.
        adj_list_path: string, path to the labels file.
    """
    filenames, texts, labels = _find_image_files(input_file, anp_list_path, noun_list_path, adj_list_path, dataset_dir)
    _process_image_files(name, filenames, texts, labels, num_shards)


def main(unused_argv):
    assert not FLAGS.num_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
    print('Saving results to %s' % FLAGS.output_directory)

    # Run it!
    _process_dataset(FLAGS.name, FLAGS.input_file, FLAGS.images_directory, FLAGS.num_shards,
                     FLAGS.anp_list, FLAGS.noun_list, FLAGS.adj_list)


if __name__ == '__main__':
    tf.app.run()
