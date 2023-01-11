
import cv2
import numpy as np
from tflite_support.task import processor
import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
import time
import os
import log 

from dotenv import load_dotenv
load_dotenv()

time_delay = float(os.getenv("TIME_DELAY"))


_MARGIN = 10  
_ROW_SIZE = 10 
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 255, 0)

def visualize(
    image: np.ndarray,
    detection_result: processor.DetectionResult,
) -> np.ndarray:
  for detection in detection_result.detections:
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (_MARGIN + bbox.origin_x,
                     _MARGIN + _ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

  return image





last_time = None

def send_data(image,detections):
    global last_time
    current_time = time.perf_counter()

    # write status to stdout without log and print function 

    if last_time is None:
        last_time = current_time
        class_name = detections[0].categories[0].category_name
        # log(f"sending detection for {class_name}")
        os.write(1, f"Sending data for {class_name} \n".encode()) # print & log not working from docker 

        
    elif current_time - last_time > 5: # seconds
        last_time = current_time
        class_name = detections[0].categories[0].category_name
        os.write(1, f"Sending data for {class_name} \n".encode())
