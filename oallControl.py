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

#variables and comments
#aividMode     #False means squirrel/photo, True means human/video
#pbState       #the state of the photo button.  False means pressed, True means unpressed
#oldpbState    #the state of the photo button last iteration, to make sure action is only taken when things change
#controlState  #0 means none, 1 means web, 2 means AI
#oldControlState




class photoButton:
	recording = False
	videoNumber = 0
	photoNumber = 0
	def takeWebPhoto(self):
		subprocess.call("~/var/www/html/photo.sh")
	def takeAIPhoto:

	def takePhoto(self):
		camera.capture("Photo" & photoNumber)
		photoNumber+=1
	def videoAction(self):
		if(recording == False):
			camera.start_recording("video" & videoNumber)
			videoNumber+=1
			recording = True
		elif(recording == True):
			camera.stop_recording()
			recording = False

def startAI():
	subprocess.call("sudo kill -9 `pidof mjpg_streamer`")
	#add ai start
def startWebControl():
	subprocess.call("sudo /usr/local/bin/mjpg_streamer -i 'input_uvc.so -r 1280x720 -d /dev/video0 -f 30 -q 80' -o 'output_http.so -p 8080 -w /usr/local/share/mjpg-streamer/www'")
	#add kill ai 
def startManual();
	#add kill ai
	subprocess.call("sudo kill -9 `pidof mjpg_streamer`")

while True:
	aividMode = GPIO.input(16)
	oldpbState = pbState
    pbState = GPIO.input(6)
    oldControlState = controlState
    if(GPIO.input(27) == False):
		controlState = 1
	elif(GPIO.input(22) == False):
	   	controlState = 2
	else:
	    controlState = 0

    if(pbState == oldpbState):
    	pass
    else:
	    if(pbState == True):
	    	if(aividMode == False):
	    		if(controlState == 0):
	    			photobutton.takePhoto()
	    		elif(controlState == 1):
	    			photobutton.takeWebPhoto()
	    		else:
	    			photobutton.takeAIPhoto()
	    	else:
	    		if(controlState == 0):
	    			photoButton.videoaction()
	    		elif(controlState == 2):
    if(oldControlState == controlState):
    	pass
    else:
    	if(controlState == 0):
    		startManual()
    	elif(controlState == 1):
    		startWebControl()
    	elif(controlState == 2):
    		startAI()
    	else:
    		startManual()
   	time.sleep(0.2)



















