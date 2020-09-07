import os
import http.client
import time
from PIL import Image
import tempfile
import subprocess
import urllib.request
import requests

def req(url):
    r = requests.get(url)
    return r.text

def url_is_alive(url):
    """
    Checks that a given URL is reachable.
    :param url: A URL
    :rtype: bool
    """
    request = urllib.request.Request(url)
    request.get_method = lambda: 'HEAD'

    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False

def fire():
    print("shoot!!!!!!")
    req("http://khaosgun.local/pinon.php")
    time.sleep(0.1)  # this is how long the gun revs to get going before firing.
    req("http://khaosgun.local/firepinon.php")

def stopFire():
    req("http://khaosgun.local/firepinoff.php")
    req("http://khaosgun.local/pinoff.php")


while True:
    os.system("wget http://khaosgun.local:8080/?action=snapshot -O /Users/reiddye/darknet/data/mjpgStream.jpg")
    #Mink
    #Meerkat
    #Weasel
    #fox_squirrel
    #Hare
    #Marmot
    #wood_rabbit
    with tempfile.TemporaryFile() as tempf:
        if(url_is_alive("http://khaosgun.local/squirrel.html")):
            im = Image.open("/Users/reiddye/darknet/data/mjpgStream.jpg")
            width, height = im.size

            # Setting the points for cropped image

            # oall x: 1280
            # oall y: 720
            #2556 × 1440

            top = (850/2556) * width  # x1
            left = (1070/1440) * height  # y1

            right = (1415/2556) * width  # x2
            bottom = (1286/1440) * height  # y2

            # Cropped image of above dimension
            im = im.crop((left, top, right, bottom))

            # Shows the image in image viewer
            im.save("/Users/reiddye/darknet/data/mjpgStream.jpg")
            proc = subprocess.Popen(['./detectSquirrel.sh'], stdout=tempf)
            proc.wait()
            tempf.seek(0)
            print(tempf.read())
            result = tempf.read()
            result = str(result)
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
                stopFire()
        elif(url_is_alive("http://khaosgun.local/human.html")):
            im = Image.open("/Users/reiddye/darknet/data/mjpgStream.jpg")
            width, height = im.size

             # Setting the points for cropped image
            top = (250/2556) * width  # x1
            left = (800/1440) * height  # y1

            right = (1665/2556) * width  # x2
            bottom = (1440/1440) * height  # y2

            # Cropped image of above dimension
            im = im.crop((left, top, right, bottom))

            # saves the cropped image
            im.save("/Users/reiddye/darknet/data/mjpgStream.jpg")


            proc = subprocess.Popen(['./detectHuman.sh'], stdout=tempf)
            proc.wait()
            tempf.seek(0)
            result = tempf.read()
            result = str(result)
            print(result)
            numberIndex = result.find('person')
            if(numberIndex != -1):
                numberIndex += 8
                result = result[numberIndex:numberIndex+2]
                if(result.find("%") != -1):
                    result = result[:1]
                if(float(result)>20): #change the 20 for changing sensetivity
                    fire()
                print(result)
            else:
                stopFire()
                
        else:
            stopFire()
