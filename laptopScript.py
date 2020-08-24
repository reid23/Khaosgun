import os
import http.client
import time
from PIL import Image
import tempfile
import subprocess
def fire():
    print("shoot!!!!!!")


while True:
    os.system("ffmpeg -y -i http://khaosgun.local:8080/?action=stream /Users/reiddye/darknet/data/mjpgStream.jpg")
    im = Image.open("/Users/reiddye/darknet/data/mjpgStream.jpg")
    width, height = im.size

    # Setting the points for cropped image
    left = 1070
    top = 991
    right = 1415
    bottom = 1286

    # Cropped image of above dimension
    im = im.crop((left, top, right, bottom))

    # Shows the image in image viewer
    im.save("/Users/reiddye/darknet/data/mjpgStream.jpg")

    #Mink
    #Meerkat
    #Weasel
    #fox_squirrel
    #Hare
    #Marmot
    #wood_rabbit
    with tempfile.TemporaryFile() as tempf:
        human = http.client.HTTPConnection("khaosgun.local/human.html")
        squirrel = http.client.HTTPConnection("khaosgun.local/squirrel.html")
        if(squirrel.getresponse().status == 200):
            proc = subprocess.Popen(['./detectSquirrel.sh'], stdout=tempf)
            proc.wait()
            tempf.seek(0)
            print(tempf.read())
            result = tempf.read()
            if(result.find("fox_squirrel") != -1):
                numberIndex = result.find("fox_squirrel")-8
                result = result[numberIndex:numberIndex+5]
                if(result.find(" ") != -1 or result.find("'") != -1):
                    result = result[:4]
                if(float(result)>20): #change the 20 if u want.  it's the threshold for what counts as a detection.
                    fire()
            elif(result.find("wood_rabbit") != -1):
                numberIndex = result.find("wood_rabbit")-8
                result = result[numberIndex:numberIndex+5]
                if(result.find(" ") != -1 or result.find("'") != -1):
                    result = result[:4]
                # change the 20 if u want.  it's the threshold for what counts as a detection.
                if(float(result) > 20):
                    fire()
            elif(result.find("hare") != -1):
                numberIndex = result.find("hare")-8
                result = result[numberIndex:numberIndex+5]
                if(result.find(" ") != -1 or result.find("'") != -1):
                    result = result[:4]
                # change the 20 if u want.  it's the threshold for what counts as a detection.
                if(float(result) > 20):
                    fire()
            elif(result.find("marmot") != -1):
                numberIndex = result.find("marmot")-8
                result = result[numberIndex:numberIndex+5]
                if(result.find(" ") != -1 or result.find("'") != -1):
                    result = result[:4]
                # change the 20 if u want.  it's the threshold for what counts as a detection.
                if(float(result) > 20):
                    fire()
            elif(result.find("mink") != -1):
                numberIndex = result.find("mink")-8
                result = result[numberIndex:numberIndex+5]
                if(result.find(" ") != -1 or result.find("'") != -1):
                    result = result[:4]
                # change the 20 if u want.  it's the threshold for what counts as a detection.
                if(float(result) > 20):
                    fire()
            elif(result.find("meerkat") != -1):
                numberIndex = result.find("meerkat")-8
                result = result[numberIndex:numberIndex+5]
                if(result.find(" ") != -1 or result.find("'") != -1):
                    result = result[:4]
                # change the 20 if u want.  it's the threshold for what counts as a detection.
                if(float(result) > 20):
                    fire()
            elif(result.find("weasel") != -1):
                numberIndex = result.find("weasel")-8
                result = result[numberIndex:numberIndex+5]
                if(result.find(" ") != -1 or result.find("'") != -1):
                    result = result[:4]
                # change the 20 if u want.  it's the threshold for what counts as a detection.
                if(float(result) > 20):
                    fire()
            else:
                pass
        elif(human.getresponse().status == 200):
            proc = subprocess.Popen(['./detectHuman.sh'], stdout=tempf)
            proc.wait()
            tempf.seek(0)
            result = tempf.read()
            numberIndex = result.find("person")
            if(numberIndex != -1):
                numberIndex += 8
                result = result[numberIndex:numberIndex+2]
                if(result.find("%") != -1)):
                    result = result[:1]
                if(float(result)<20): #change the 20 for changing sensetivity
                    fire()
                
        else:
            time.sleep(0.1)
