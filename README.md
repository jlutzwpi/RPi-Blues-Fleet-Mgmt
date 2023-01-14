# RPi-Blues-Fleet-Mgmt

Determine driver of vehicle, if they are wearing a seatbelt, if they are driving aggressively, if they've been drinking alcohol, and speed and location information.

To run: python3 rpi-blues-fleet.py rpi-blues-yolo-linux-armv7-v3.eim [/dev/video0 | video-file.mp4]

/dev/video0 will run live video from the Raspberry Pi camera.  You can also test with a video.

Make sure you run the code with view of the southern sky so GPS satellite fix can be acquired.  Inference will not run until GPS location is determined.
