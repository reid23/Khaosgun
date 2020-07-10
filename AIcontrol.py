#switch: AI vs Web

#importing libraries
#import RPi.GPIO as GPIO
import time

#setup
#GPIO.setmode(GPIO.BCM)
#GPIO.setup(26, GPIO.IN, pull_up_down = GPIO.PUD_UP)

#AI:
class AI:
	attribute = 5

object = AI()
print(object.attribute)


#code
#while True:
#        inputState = GPIO.input(26)
#        if inputState == False:
#                
#        else:
#                
#        time.sleep(0.2)
#