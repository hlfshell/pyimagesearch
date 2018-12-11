import argparse
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import time

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tracker", type=str)

args = vars(ap.parse_args())

trackerFncs = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

vs = VideoStream(src=0).start()
time.sleep(1.0)

fps = None

trackers = []

fps = FPS().start()

while True:
    frame = vs.read()

    (H, W) = frame.shape[:2]

    if frame is None:
        break

    key = cv2.waitKey(1) & 0xFF 

    if key == ord("q"):
        break

    if key == ord("s"):
        bounding_box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)

        tracker = trackerFncs[args["tracker"]]()
        tracker.init(frame, bounding_box)
        trackers.append(tracker)

    ##Handle tracking
    if len(trackers) > 0:
        for tracker in trackers:
            (success, box) = tracker.update(frame)

            if success:
                (x,y,w,h) = [int(verticie) for verticie in box]
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

            cv2.putText(frame, "FPS: {}".format(fps.fps()), (10, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    
    fps.update()
    fps.stop()