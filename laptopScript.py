import os
import http.client
import time
from PIL import Image
import tempfile
import subprocess
human = http.client.HTTPConnection("khaosgun.local/human.html")
squirrel = http.client.HTTPConnection("khaosgun.local/squirrel.html")
human.request("HEAD", '')
def fire():
    #shoot the thing
    pass


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
    im.save()

    #Mink
    #Meerkat
    #Weasel
    #fox_squirrel
    #Hare
    #Marmot
    #wood_rabbit
    with tempfile.TemporaryFile() as tempf:
        if(human.getresponse().status == 200):
            proc = subprocess.Popen(['./detectSquirrel.sh'], stdout=tempf)
            proc.wait()
            tempf.seek(0)
            print(tempf.read())
            result = tempf.read()
            if(result.find("fox_squirrel") == True):
                numberIndex = result.find("fox_squirrel")-8
                result = result[numberIndex:numberIndex+5]
                if(result.find(" ") != -1 or result.find("'") != -1):
                    result = result[:4]
                if(float(result)>20): #change the 20 if u want.  it's the threshold for what counts as a detection.
                    fire()
            elif(result.find("hare") == True):
                numberIndex = result.find("hare")-8
                result = result[numberIndex:numberIndex+5]
                if(result.find(" ") != -1 or result.find("'") != -1):
                    result = result[:4]
                if(float(result) > 20):  # change the 20 if u want.  it's the threshold for what counts as a detection.
                    fire()
            elif(result.find("wood_rabbit") == True):
                numberIndex = result.find("wood_rabbit")-8
                result = result[numberIndex:numberIndex+5]
                if(result.find(" ") != -1 or result.find("'") != -1):
                    result = result[:4]
                # change the 20 if u want.  it's the threshold for what counts as a detection.
                if(float(result) > 20):
                    fire()
            elif(result.find("hare") == True):
                numberIndex = result.find("hare")-8
                result = result[numberIndex:numberIndex+5]
                if(result.find(" ") != -1 or result.find("'") != -1):
                    result = result[:4]
                # change the 20 if u want.  it's the threshold for what counts as a detection.
                if(float(result) > 20):
                    fire()
            elif(result.find("marmot") == True):
                numberIndex = result.find("hare")-8
                result = result[numberIndex:numberIndex+5]
                if(result.find(" ") != -1 or result.find("'") != -1):
                    result = result[:4]
                # change the 20 if u want.  it's the threshold for what counts as a detection.
                if(float(result) > 20):
                    fire()
            elif(result.find("mink") == True):
                numberIndex = result.find("hare")-8
                result = result[numberIndex:numberIndex+5]
                if(result.find(" ") != -1 or result.find("'") != -1):
                    result = result[:4]
                # change the 20 if u want.  it's the threshold for what counts as a detection.
                if(float(result) > 20):
                    fire()
            elif(result.find("meerkat") == True):
                numberIndex = result.find("hare")-8
                result = result[numberIndex:numberIndex+5]
                if(result.find(" ") != -1 or result.find("'") != -1):
                    result = result[:4]
                # change the 20 if u want.  it's the threshold for what counts as a detection.
                if(float(result) > 20):
                    fire()
            elif(result.find("weasel") == True):
                numberIndex = result.find("hare")-8
                result = result[numberIndex:numberIndex+5]
                if(result.find(" ") != -1 or result.find("'") != -1):
                    result = result[:4]
                # change the 20 if u want.  it's the threshold for what counts as a detection.
                if(float(result) > 20):
                    fire()
            else:
                pass
        elif(squirrel.getresponse().status == 200):
            os.system('./detectHuman.sh')
        else:
            time.sleep(0.1)
