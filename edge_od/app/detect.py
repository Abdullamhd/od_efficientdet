import sys
import time
import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
import os
import log

from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor()


from dotenv import load_dotenv
load_dotenv()

selected_model = os.getenv("MODEL_NAME")


ip_camera_url = os.getenv("CAMERA_URL2")

trushold = float(os.getenv("TRUSHOLD"))

default_width = 640
default_height = 480


def run(model: str = selected_model, camera_id:str = ip_camera_url, width: int =default_width, height: int = default_height, num_threads: int = 4) -> None:

  counter, fps = 0, 0
  start_time = time.time()

  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  row_size = 20  
  left_margin = 24 
  text_color = (0, 0, 255)  
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  base_options = core.BaseOptions(
      file_name=model, use_coral=False, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=trushold)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from camera feed'
      )
    # reduce the loop speed 
    # time.sleep(0.01)

    image = cv2.flip(image, 1)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    detection_result = detector.detect(input_tensor)

    # image = utils.visualize(image, detection_result)
    if len(detection_result.detections) > 0:
        executor.submit(utils.send_data, image,detection_result.detections)

    # if counter % fps_avg_frame_count == 0:
    #   end_time = time.time()
    #   fps = fps_avg_frame_count / (end_time - start_time)
    #   start_time = time.time()

    # fps_text = 'FPS = {:.1f}'.format(fps)
    # text_location = (left_margin, row_size)
    # cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
    #             font_size, text_color, font_thickness)

    # if cv2.waitKey(1) == 27:
    #   break
    # cv2.imshow('', image)

  cap.release()
  cv2.destroyAllWindows()








def main():
  run()
if __name__ == '__main__':
  main()
