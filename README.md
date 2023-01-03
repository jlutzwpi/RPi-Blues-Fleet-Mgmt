# RPi-Blues-Fleet-Mgmt

Determine driver of vehicle, if they are wearing a seatbelt, and speed and location information.

To run: python3 rpi-blues-fleet.py model_name.eim [/dev/video0 | video-file.mp4]

Make sure you run the code with view of the southern sky so GPS satellite fix can be acquired.  Inference will not run until GPS location is determined.
