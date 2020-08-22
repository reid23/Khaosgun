#!/bin/sh
cd /Users/reiddye/darknet
./darknet detect cfg/yolov3-tiny.cfg yolov3-tiny.weights data/person.jpg
cd
