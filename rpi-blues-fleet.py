#!/usr/bin/env python
## Fleet Management Tool using Edge Impulse Model to detect person and seatbelt
## as well as location and velocity information from the Notecard
## Results are sent via Blues Wireless Notecard

import cv2
import os
import time
import sys, getopt
import numpy as np
import json
import notecard
from periphery import I2C
import time
from edge_impulse_linux.image import ImageImpulseRunner
from DFRobot_Alcohol import *
from LIS3DHTR import LIS3DHTR
#define the accelerometer
lis3dhtr = LIS3DHTR()
num_hard_accel = 0
num_hard_brake = 0

# Blues Wireless Product ID
productUID = "xxxxxx"
#only send results every 2 minutes
TIME_SLEEP = 120
port = I2C("/dev/i2c-1")
card = notecard.OpenI2C(port, 0, 0)
# Configure the notecard.  Sync should be periodic and GPS periodic as well
req = {"req": "hub.set"}
req["product"] = productUID
req["mode"] = "periodic"
req["outbound"] = 15
req["inbound"] = 60
print(json.dumps(req))
rsp = card.Transaction(req)
print(rsp)
#GPS
req = {"req": "card.location.mode"}
req["mode"] = "periodic"
req["seconds"] = 20
print(json.dumps(req))
rsp = card.Transaction(req)
print(rsp)
#heartbeat
req = {"req": "card.location.track"}
req["start"] = True
req["heartbeat"] = True
req["hours"] = 1
print(json.dumps(req))
rsp = card.Transaction(req)
print(rsp)
#increase sensitivity
req = {"req": "card.motion.mode"}
req["start"] = True
req["sensitivity"] = 2
print(json.dumps(req))
rsp = card.Transaction(req)
print(rsp)
#take an average of the seatbelt inference
avg_seatbelt = []
gps_found = False
#set up alcohol sensor
COLLECT_NUMBER   = 1               # collect number, the collection range is 1-100
I2C_MODE         = 0x01            # default use I2C1
#confirmed with i2cdetect that I2C address is 0x75, which is ALCOHOL_ADDRESS_3
alcohol = DFRobot_Alcohol_I2C (I2C_MODE, ALCOHOL_ADDRESS_3)
alcohol.set_mode(MEASURE_MODE_AUTOMATIC)
#similar to seatbelt, keep running average of alcohol detection
avg_alcohol = []

runner = None
# if you don't want to see a video preview, set this to False
#show_camera = True
show_camera = False
if (sys.platform == 'linux' and not os.environ.get('DISPLAY')):
    show_camera = False


def help():
    print('python classify-video.py <path_to_model.eim> <path_to_video.mp4 or /dev/video0>')
    #can also use /dev/video0 for live camera feed

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["--help"])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
            sys.exit()

    if len(args) != 2:
        help()
        sys.exit(2)

    model = args[0]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    print('MODEL: ' + modelfile)

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
            labels = model_info['model_parameters']['labels']

            vidcap = cv2.VideoCapture(args[1])
            sec = 0
            start_time = time.time()

            def getFrame(sec):
                vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
                hasFrames,image = vidcap.read()
                if hasFrames:
                    return image
                else:
                    print('Failed to load frame', args[1])
                    exit(1)


            img = getFrame(sec)
            
            #only run inference when GPS is found
            while img.size != 0:
                # make sure we have GPS
                req = {"req": "card.location"}
                rsp = card.Transaction(req)
                #global gps_found
                #if not gps_found:
                print(rsp)
                if "lat" in rsp:
                    gps_found = True
                else:
                    #don't run inference until GPS is found
                    continue

                # imread returns images in BGR format, so we need to convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # get_features_from_image also takes a crop direction arguments in case you don't have square images
                features, cropped = runner.get_features_from_image(img)

                # the image will be resized and cropped, save a copy of the picture here
                # so you can see what's being passed into the classifier
                cv2.imwrite('debug.jpg', cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

                res = runner.classify(features)

                if "classification" in res["result"].keys():
                    print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
                    for label in labels:
                        score = res['result']['classification'][label]
                        print('%s: %.2f\t' % (label, score), end='')
                    print('', flush=True)

                elif "bounding_boxes" in res["result"].keys():
                    print('Found %d bounding boxes (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
                    face_id = "null"
                    seatbelt = 0
                    conf = 0
                    for bb in res["result"]["bounding_boxes"]:
                        print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))
                        img = cv2.rectangle(cropped, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (255, 0, 0), 1)
                        img = cv2.putText(img, bb['label'], (bb['x'], bb['y']-10), cv2.FONT_HERSHEY_TRIPLEX,1, (255,0,0),2,cv2.LINE_AA)
                        #detections found, send to Notecard
                        
                        #my model has 2 classes: Justin (driver) and seatbelt (if it detects a seatbelt being worn)
                        if bb['label'] == "Justin":
                            face_id = bb['label']
                            #convert from decimal to %
                            conf = round((bb['value'] * 100))
                        elif bb['label'] == "seatbelt":
                            seatbelt = 1
                    #keep a list of seatbelt readings to calculate average
                    global avg_seatbelt
                    avg_seatbelt.append(seatbelt)
                    global alcohol
                    #get the alcohol concentration reading from the DFRobot MQ3 sensor
                    alcohol_concentration = alcohol.get_alcohol_data(COLLECT_NUMBER);
                    if alcohol_concentration == ERROR:
                        print("Please check the connection !")
                    else:
                        print("\n*** alcohol concentration is %.2f PPM.\n"%alcohol_concentration)
                        #add to list
                        global avg_alcohol
                        avg_alcohol.append(alcohol_concentration)
                    #get accel data to check for aggressive driving
                    lis3dhtr.select_datarate()
                    lis3dhtr.select_data_config()
                    accl = lis3dhtr.read_accl()
                    global num_hard_accel
                    if accl['y'] > 10000:
                        num_hard_accel = num_hard_accel + 1
                    global num_hard_brake
                    if accl['y'] < 0:
                        num_hard_brake = num_hard_brake + 1
                    #only send data every 2 minutes
                    # number of seconds elapsed modulo 120 should be < 1
                    if sec % TIME_SLEEP < 1.0:
                        #calculate the average
                        avg = sum(avg_seatbelt) / len(avg_seatbelt)
                        sb = 0
                        #if the average is > 50%, driver is wearing their seatbelt
                        if avg > 0.5:
                            sb = 1
                        # empty list and start over
                        avg_seatbelt.clear()
                        #do the same for alcohol
                        avg = sum(avg_alcohol) / len(avg_alcohol)
                        alc = 0
                        #if alcohol avg is > 0.6 ppm then flag it
                        if avg > 0.6:
                            alc = 1
                        avg_alcohol.clear()
                        req = {"req": "note.add"}
                        req["file"] = "sensors.qo"
                        req["body"] = { "faceID": face_id, "confidence": conf, "seatbelt": sb, 
                                        "alcohol_detected": alc, "num_hard_accel" : num_hard_accel,
                                        "num_hard_brake" : num_hard_brake }
                        rsp = card.Transaction(req)
                        print(rsp)
                        print("Note sent!")

                if (show_camera):
                    cv2.imshow('Fleet Management', cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) == ord('q'):
                        break

                sec = time.time() - start_time
                sec = round(sec, 2)
                #print("Getting frame at: %.2f sec" % sec)
                img = getFrame(sec)
        finally:
            if (runner):
                runner.stop()

if __name__ == "__main__":
   main(sys.argv[1:])