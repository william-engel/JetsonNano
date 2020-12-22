import os
import tensorflow as tf
from inference_utils import run_inference_on_videostream

tf.get_logger().setLevel('ERROR')

project_path = os.getcwd()
workspace_path = os.path.join(project_path, 'Tensorflow/workspace/training_demo')

# Load detection model
model_dir = os.path.join(workspace_path, 'pre-trained-models/ssd_mobilenet_v2_320x320_coco17_tpu-8')
detection_model = tf.saved_model.load(os.path.join(model_dir, 'saved_model'))
detection_model_fn = detection_model.signatures['serving_default']

rtsp_url = 'rtsp://192.168.2.112:8080/h264_ulaw.sdp'

run_inference_on_videostream(model_fn = detection_model_fn,
                             num_warmup_rounds = 300, 
                             streaming_adress = rtsp_url,
                             save_video_with_detections = True, 
                             target_size = (320,320), 
                             num_frames = 300)
