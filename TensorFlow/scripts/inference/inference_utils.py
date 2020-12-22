import cv2
import tensorflow as tf
from datetime import datetime
import time
import numpy as np

from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util

PATH_TO_LABELS = 'TensorFlow/models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


def visualize_results(image, results):

  # convert result to numpy values
  result = {key:value.numpy() for key,value in results.items()}

  image_with_detections = np.asarray(image).copy()

  viz_utils.visualize_boxes_and_labels_on_image_array(
      image = image_with_detections,
      boxes = result['detection_boxes'][0],
      classes = result['detection_classes'][0].astype(int),
      scores = result['detection_scores'][0],
      category_index = category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=4,
      min_score_thresh=.50
  )

  return image_with_detections

def run_inference_on_videostream(model_fn = None, num_warmup_rounds = 300, streaming_adress = None,
                                 save_video_with_detections = False, target_size = (320,320), num_frames = 300):
  
  def preprocess_frame(frame = None, target_size = None, rotate_90deg = False):
    frame = cv2.resize(frame, dsize = target_size)
    if rotate_90deg: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) 
    return tf.convert_to_tensor(frame, dtype = tf.uint8)

  cap = cv2.VideoCapture(streaming_adress)

  # create output video
  if save_video_with_detections:
    dateTimeObj = datetime.now()
    fname = 'result_{}.mp4'. format(dateTimeObj.strftime("%d_%m_%Y_%H_%M_%S"))
    out = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'mp4v'), 15, target_size) 

  # test streaming rate
  print('testing streamingrate...')
  elapsed_time = []
  for i in range(100):
    start_time = time.time()
    ret, frame = cap.read()
    end_time = time.time()
    elapsed_time = np.append(elapsed_time, end_time - start_time)
  print('Streaming with: %.0d fps'% (100 / elapsed_time.sum()))
  
  # warmup
  print('running warmup...')
  # run num_warmup_rounds
  for i in range(num_warmup_rounds):
    ret, frame = cap.read()
    frame = preprocess_frame(frame, target_size)
    results = model_fn(frame[tf.newaxis,...])
  
  # inference
  print('running inference...')
  elapsed_time = []
  counter = 0
  while(cap.isOpened()):
      ret, frame = cap.read()
      if ret == True:
        counter += 1

        # preprocess frame
        frame = preprocess_frame(frame, target_size)

        # capture inference time
        start_time = time.time()
        results = model_fn(frame[tf.newaxis,...])
        end_time = time.time()
        elapsed_time = np.append(elapsed_time, end_time - start_time)

        # visualizer results
        if save_video_with_detections:
          image_with_detections = visualize_results(frame, results)
          out.write(image_with_detections)

        # print intermediate result
        if counter % 200 == 0:
          print('Step {}: {:4.1f}ms'.format(counter, (elapsed_time[-50:].mean()) * 1000))
        
        if counter == num_frames: break

      else:
          break

  cap.release()
  out.release() 

  print('Throughput: {:.0f} fps'.format(num_frames / elapsed_time.sum()))



def run_inference_on_video(model_fn = None, num_warmup_rounds = 300, filename = None,
                           save_video_with_detections = False, target_size = (320,320)):
  
  def preprocess_frame(frame = None, target_size = None):
    frame = cv2.resize(frame, dsize = target_size)
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return tf.convert_to_tensor(frame, dtype = tf.uint8)

  elapsed_time = []

  cap = cv2.VideoCapture(filename)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  # create output video
  if save_video_with_detections:
    dateTimeObj = datetime.now()
    fname = 'result_{}.mp4'. format(dateTimeObj.strftime("%d_%m_%Y_%H_%M_%S"))
    out = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'MP4V'), 15, target_size) 

  # warmup
  print('running warmup...')
  # get warmup image
  _, warmup_image = cap.read()
  warmup_image = preprocess_frame(warmup_image, target_size)
  # run num_warmup_rounds
  for i in range(num_warmup_rounds):
    results = model_fn(warmup_image[tf.newaxis,...])
  # reset video
  cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

  # inference
  print('running inference...')
  counter = 0
  while(cap.isOpened()):
      ret, frame = cap.read()
      if ret == True:
        counter += 1

        # preprocess frame
        frame = preprocess_frame(frame, target_size)

        # capture inference time
        start_time = time.time()
        results = model_fn(frame[tf.newaxis,...])
        end_time = time.time()
        elapsed_time = np.append(elapsed_time, end_time - start_time)

        # visualizer results
        if save_video_with_detections:
          image_with_detections = visualize_results(frame, results)
          out.write(image_with_detections)

        # print intermediate result
        if counter % 200 == 0:
          print('Step {}: {:4.1f}ms'.format(counter, (elapsed_time[-50:].mean()) * 1000))

      else:
          break

  cap.release()
  out.release() 

  print('Throughput: {:.0f} fps'.format(total_frames / elapsed_time.sum()))

  if save_video_with_detections:
    files.download(fname)