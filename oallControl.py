#oall control

import RPi.GPIO as GPIO
import time
import os
import subprocess
from picamera import PiCamera
from enum import Enum

GPIO.setmode(GPIO.BCM)

GPIO.setup(6, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(16, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_UP)

camera = PiCamera()
photobutton = photoButton()

bool aividMode #true means squirrel/photo, false means human/video
bool pbState
bool oldpbState
int controlState #0 means none, 1 means web, 2 means AI
int oldControlState




class photoButton:
	recording = false
	videoNumber = 0
	photoNumber = 0
	def takeWebPhoto(self):
		subprocess.call("~/var/www/html/photo.sh")
	def takeAIPhoto:

	def takePhoto(self):
		camera.capture("Photo" & photoNumber)
		photoNumber++
	def videoAction(self):
		if(recording == false):
			camera.start_recording("video" & videoNumber)
			videoNumber++
			recording = true
		else if(recording == true):
			camera.stop_recording()
			recording = false

def startAI():
	subprocess.call("sudo kill -9 `pidof mjpg_streamer`")
	#add ai start
def startWebControl():
	subprocess.call("sudo /usr/local/bin/mjpg_streamer -i 'input_uvc.so -r 1280x720 -d /dev/video0 -f 30 -q 80' -o 'output_http.so -p 8080 -w /usr/local/share/mjpg-streamer/www'")
def startManual();
	#kill ai
	subprocess.call("sudo kill -9 `pidof mjpg_streamer`")

while True:
	aividMode = GPIO.input(16)
	oldpbState = pbState
    pbState = GPIO.input(6)
    oldControlState = controlState
    if(GPIO.input(27) == true):
		controlState = 1
	else if(GPIO.input(22)):
	   	controlState = 2
	else:
	    controlState = 0

    if(pbState = oldpbState):
    	
    else:
	    if(pbState == true):
	    	if(aividMode == true):
	    		if(controlState == 0):
	    			photobutton.takePhoto()
	    		else if(controlState == 1):
	    			photobutton.takeWebPhoto()
	    		else:
	    			photobutton.takeAIPhoto()
	    	else:
	    		if(controlState == 0):
	    			photoButton.videoaction()
	    		else if(controlState == 2):
    if(oldControlState == controlState):

    else:
    	if(controlState == 0):
    		startManual()
    	else if(controlState == 1):
    		startWebControl()
    	else if(controlState == 2):
    		startAI()
    	else:
    		startManual()



















