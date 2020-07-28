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

photoNumber = 0
videoNumber = 0
iscamera = False

#variables and comments
#aividMode     #False means squirrel/photo, True means human/video
#pbState       #the state of the photo button.  False means pressed, True means unpressed
#oldpbState    #the state of the photo button last iteration, to make sure action is only take$
#controlState  #0 means none, 1 means web, 2 means AI
#oldControlState


recording = False

class photoButton:
    videoNumber = 0
    photoNumber = 0
    def takeWebPhoto(self):
        os.system("/var/www/html/photo.sh &")
    def takeAIPhoto(self):
        pass
    def takePhoto(self):
        global photoNumber
        global camera
        camera.capture("/var/www/html/photos/Photo{}.jpg".format(photoNumber))
        photoNumber+=1
    def videoAction(self):
        global videoNumber
        global camera
        global recording
        if(recording == False):
            camera.start_recording("/var/www/html/photos/Video{}.h264".format(videoNumber))
            videoNumber+=1
            recording = True
            print("recording...")
        elif(recording == True):
            camera.stop_recording()
            recording = False
            print("stopping recording")

photobutton = photoButton()


def startAI():
    global camera
    print("starting AI control")
    os.system("sudo kill -9 `pidof mjpg_streamer` &")
    global iscamera
    if(iscamera == True):
        camera.close()
        iscamera = False
    #add ai start
def startWebControl():
    global camera
    print("starting web control")
    global iscamera
    if(iscamera == True):
        camera.close()
        time.sleep(0.2)
        iscamera == False
    os.system("sudo /usr/local/bin/mjpg_streamer -i 'input_uvc.so -r 1280x720 -d /dev/video0 -f 30 -q 80' -o 'output_http.so -p 8080 -w /usr/local/share/mjpg-streamer/www' &")
    #add kill ai
def startManual():
    global camera
    #add kill ai
    print("starting manual mode")
    global iscamera
    os.system("sudo kill -9 `pidof mjpg_streamer` &")
    time.sleep(0.5)
    camera = PiCamera()
    iscamera = True

pbState = True
controlState = 4

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
                    photobutton.videoAction()
                elif(controlState == 2):
                    pass
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
    time.sleep(0.05)

