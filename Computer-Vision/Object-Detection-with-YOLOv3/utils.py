#!/usr/bin/env python
# coding: utf-8

# In[13]:


import tensorflow as tf
import numpy as np
import cv2
import time


# In[14]:


#non_max_suppression funtion

def YOLO_nms(inputs, model_size, 
             max_output_size, max_output_size_per_class, 
             iou_threshold, confidence_threshold):
    
    #Extracting box features, confidence and class probs from yolo output
    bbox, confs, class_probs = tf.split(inputs, [4, 1, -1], axis=-1)
    bbox=bbox/model_size[0]
    
    #Scores are defined as object confidence x class probabilities
    scores = confs * class_probs
    
    #Using tensorflow non_max_suppression func
    boxes, scores, classes, valid_detections =         tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1,
                                   tf.shape(scores)[-1])),
        max_output_size_per_class=max_output_size_per_class,
        max_total_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=confidence_threshold
    )
    return boxes, scores, classes, valid_detections

#Reseizing image to the model_Size (416 x 416)
def resize_image(inputs, modelsize):
    inputs= tf.image.resize(inputs, modelsize)
    return inputs

#Loading coco class names
def load_class_names(file_name):
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

# This function is used to convert the boxes into the format of 
#  (top-left-corner, bottom-right-corner), 
#  following by applying the NMS function and returning the proper bounding boxes.

def bounding_boxes(inputs,model_size, max_output_size, max_output_size_per_class,
                 iou_threshold, confidence_threshold):
    center_x, center_y, width, height, confidence, classes =         tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)
    top_left_x = center_x - width / 2.0
    top_left_y = center_y - height / 2.0
    bottom_right_x = center_x + width / 2.0
    bottom_right_y = center_y + height / 2.0
    inputs = tf.concat([top_left_x, top_left_y, bottom_right_x,
                        bottom_right_y, confidence, classes], axis=-1)
    boxes_dicts = YOLO_nms(inputs, model_size, max_output_size,
                                      max_output_size_per_class, iou_threshold, confidence_threshold)
    return boxes_dicts

#Drawing Boxes
def draw_boxes(img, boxes, objectness, classes, nums, class_names):
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    boxes=np.array(boxes)
    for i in range(nums):
        x1y1 = tuple((boxes[i,0:2] * [img.shape[1],img.shape[0]]).astype(np.int32))
        x2y2 = tuple((boxes[i,2:4] * [img.shape[1],img.shape[0]]).astype(np.int32))
        img = cv2.rectangle(img, (x1y1), (x2y2), (0,0,255), 1)
        img = cv2.putText(img, (class_names[int(classes[i])]),
                          (x1y1), cv2.FONT_HERSHEY_PLAIN , 1, (255, 255, 0), 2)
    return img


# In[ ]:




