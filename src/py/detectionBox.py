import tensorflow as tf
import cv2
import numpy as np


# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    step = "2913"
    with tf.gfile.GFile("/Users/kitamurataku/work/sudoku-recognition-model/tf/models/model/frozenModel/sudoku-" + step + "/frozen_inference_graph.pb", 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

sampleImg = cv2.imread("./sample.jpg")

height, width = sampleImg.shape[:2]
print(height, width)

sampleImg= cv2.resize(sampleImg,dsize=(299,299), interpolation = cv2.INTER_CUBIC)

np_image_data = np.asarray(sampleImg)

np_final = np.expand_dims(np_image_data, axis=0)

(boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: np_final})

boxes = np.squeeze(boxes)
scores = np.squeeze(scores)
print(scores)
max_boxes_to_draw = boxes.shape[0]
min_score_thresh = 0.01
for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
        box = tuple(boxes[i].tolist())
        # print(box)
        ymin = box[0]*height
        xmin = box[1]*width
        ymax = box[2]*height
        xmax = box[3]*width
        print(xmin ,xmax, ymin, ymax)

# saver = tf.train.Saver()
# sess = tf.Session()

# saver = tf.train.import_meta_graph("model.ckpt-10277.meta")
# saver.restore(sess, 'model.ckpt-10277')

# estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)

# preds = estimator.predict(input_fn=train_input_fn)
#     for pred in preds:
#     print(pred)

# def visualize_boxes_and_labels_on_image_array(
#     image,
#     boxes,
#     classes,
#     scores,
#     category_index,
#     instance_masks=None,
#     instance_boundaries=None,
#     keypoints=None,
#     use_normalized_coordinates=False,
#     max_boxes_to_draw=20,
#     min_score_thresh=.5,
#     agnostic_mode=False,
#     line_thickness=4,
#     groundtruth_box_visualization_color='black',
#     skip_scores=False,
#     skip_labels=False):
#   """Overlay labeled boxes on an image with formatted scores and label names.
#   This function groups boxes that correspond to the same location
#   and creates a display string for each detection and overlays these
#   on the image. Note that this function modifies the image in place, and returns
#   that same image.
#   Args:
#     image: uint8 numpy array with shape (img_height, img_width, 3)
#     boxes: a numpy array of shape [N, 4]
#     classes: a numpy array of shape [N]. Note that class indices are 1-based,
#       and match the keys in the label map.
#     scores: a numpy array of shape [N] or None.  If scores=None, then
#       this function assumes that the boxes to be plotted are groundtruth
#       boxes and plot all boxes as black with no classes or scores.
#     category_index: a dict containing category dictionaries (each holding
#       category index `id` and category name `name`) keyed by category indices.
#     instance_masks: a numpy array of shape [N, image_height, image_width] with
#       values ranging between 0 and 1, can be None.
#     instance_boundaries: a numpy array of shape [N, image_height, image_width]
#       with values ranging between 0 and 1, can be None.
#     keypoints: a numpy array of shape [N, num_keypoints, 2], can
#       be None
#     use_normalized_coordinates: whether boxes is to be interpreted as
#       normalized coordinates or not.
#     max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
#       all boxes.
#     min_score_thresh: minimum score threshold for a box to be visualized
#     agnostic_mode: boolean (default: False) controlling whether to evaluate in
#       class-agnostic mode or not.  This mode will display scores but ignore
#       classes.
#     line_thickness: integer (default: 4) controlling line width of the boxes.
#     groundtruth_box_visualization_color: box color for visualizing groundtruth
#       boxes
#     skip_scores: whether to skip score when drawing a single detection
#     skip_labels: whether to skip label when drawing a single detection
#   Returns:
#     uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
#   """
#   # Create a display string (and color) for every box location, group any boxes
#   # that correspond to the same location.
#   box_to_display_str_map = collections.defaultdict(list)
#   box_to_color_map = collections.defaultdict(str)
#   box_to_instance_masks_map = {}
#   box_to_instance_boundaries_map = {}
#   box_to_keypoints_map = collections.defaultdict(list)
#   if not max_boxes_to_draw:
#     max_boxes_to_draw = boxes.shape[0]
#   for i in range(min(max_boxes_to_draw, boxes.shape[0])):
#     if scores is None or scores[i] > min_score_thresh:
#       box = tuple(boxes[i].tolist())
#       if instance_masks is not None:
#         box_to_instance_masks_map[box] = instance_masks[i]
#       if instance_boundaries is not None:
#         box_to_instance_boundaries_map[box] = instance_boundaries[i]
#       if keypoints is not None:
#         box_to_keypoints_map[box].extend(keypoints[i])
#       if scores is None:
#         box_to_color_map[box] = groundtruth_box_visualization_color
#       else:
#         display_str = ''
#         if not skip_labels:
#           if not agnostic_mode:
#             if classes[i] in category_index.keys():
#               class_name = category_index[classes[i]]['name']
#             else:
#               class_name = 'N/A'
#             display_str = str(class_name)
#         if not skip_scores:
#           if not display_str:
#             display_str = '{}%'.format(int(100*scores[i]))
#           else:
#             display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
#         box_to_display_str_map[box].append(display_str)
#         if agnostic_mode:
#           box_to_color_map[box] = 'DarkOrange'
#         else:
#           box_to_color_map[box] = STANDARD_COLORS[
#               classes[i] % len(STANDARD_COLORS)]

#   # Draw all boxes onto image.
#   for box, _ in box_to_color_map.items():
#     ymin, xmin, ymax, xmax = box

#   return image
