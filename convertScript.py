import tensorflow as tf
import datetime
import glob
import os

from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

compositPath= [
    './composite/crossword',
    './composite/document',
    './composite/eraser',
    './composite/pen',
    './composite/website',
]


def create_tf_example(input_file_path):
    # TODO(user): Populate the following variables from your example.
    #   height = None # Image height
    #   width = None # Image width
    #   filename = None # Filename of the image. Empty if image is not from file
    #   encoded_image_data = None # Encoded image bytes
    #   image_format = None # b'jpeg' or b'png'

    #   xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    #   xmaxs = [] # List of normalized right x coordinates in bounding box
    #              # (1 per box)
    #   ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    #   ymaxs = [] # List of normalized bottom y coordinates in bounding box
    #              # (1 per box)
    #   classes_text = [] # List of string class name of bounding box (1 per box)
    #   classes = [] # List of integer class id of bounding box (1 per box)
    # >>> file_name = "composite_208jreu.jpg_1024_768_322_818_54_586.jpg"
    # >>> file_name.split("_")
    # ['composite', '208jreu.jpg', '1024', '768', '322', '818', '54', '586.jpg']
    split_file_name = os.path.basename(input_file_path).split("_")

    filename = str.encode(split_file_name[1])
    image_format = b'jpeg'
    width = int(split_file_name[2])
    height = int(split_file_name[3])
    xmins = [int(split_file_name[4])/width]
    xmaxs = [int(split_file_name[5])/width]
    ymins = [int(split_file_name[6])/height]
    ymaxs = [int(split_file_name[7].split(".")[0])/height]
    classes_text = [b"sudoku"]
    classes = [1]
    with tf.gfile.GFile(input_file_path, 'rb') as fid:
        encoded_image_data = fid.read()

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    
    composit_path= [
        './composite/crossword',
        './composite/document',
        './composite/eraser',
        './composite/pen',
        './composite/website',
    ]

    writer = tf.python_io.TFRecordWriter("eval.record")

    for i, input_dir_path in enumerate(compositPath):
        fileList = glob.glob(input_dir_path + '/*')
        for input_file_path in fileList:
            tf_example = create_tf_example(input_file_path)
            writer.write(tf_example.SerializeToString())
    writer.close()

if __name__ == '__main__':
  tf.app.run()