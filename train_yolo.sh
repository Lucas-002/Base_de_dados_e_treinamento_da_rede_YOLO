#!/bin/bash
# Treinamento do YOLO no Darknet

./darknet detector train data/obj.data cfg/yolov4.cfg darknet53.conv.74 -dont_show