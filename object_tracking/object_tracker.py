from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="Path to input video file")
ap.add_argument("-t", "--tracker", type=str, help="Which tracker to use")
args = vars(ap.parse_args())

trackers = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

tracker = trackers[args["tracker"]]()

# if args.get("video", False):
#     print("Loading webcam")
#     vs = VideoStream(src=0).start()
#     time.sleep(1.0)
# else:
#     print("Loading video:", args["video"])
#     vs = cv2.VideoCapture(args["video"])
vs = VideoStream(src=0).start()
time.sleep(1.0)

fps = None

#Initial bounding box
initialBB = None

while True:
    frame = vs.read()

    if frame is None:
        break


    #resize the frame
    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]

    #Do the tracking!
    if initialBB is not None:
        (success, box) = tracker.update(frame)

        if success:
            (x,y,w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

        fps.update()
        fps.stop()

        info = [
            ("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps()))
        ]

        for(i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("s"):
        initialBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)

        tracker.init(frame, initialBB)
        fps = FPS().start()