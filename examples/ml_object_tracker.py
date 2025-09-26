import cv2
from ultralytics import YOLO
import numpy as np
import os
from time import sleep, time
import threading
from collections import defaultdict
from multiprocessing import Process, Queue
from siyi_control import SIYIControl
from scipy.spatial import distance as dist

import queue

"""
----------------------------------------------------------------------
                    CONFIGURATION AND INITIALIZATION
----------------------------------------------------------------------
"""

# Check OpenCV version for proper tracker creation
opencv_version = cv2.__version__.split('.')
OPENCV_MAJOR = int(opencv_version[0])
OPENCV_MINOR = int(opencv_version[1])

class Track_Object:
   def __init__(self) -> None:      
      self.xyxy = [0, 0, 0, 0]
      self._xywh = [0, 0, 0, 0]
      self.name = ""
      self.centerPoint = []
      self.class_id = -1
      self.track_id = -1

   def xywh(self):
      w = int(self.xyxy[2] - self.xyxy[0])
      h = int(self.xyxy[3] - self.xyxy[1])
      self._xywh = (int(self.xyxy[0]), int(self.xyxy[1]), w, h)
      return self._xywh
   
#----------------------------------------------------------------------
# VIDEO CAPTURE CLASS
#----------------------------------------------------------------------

class VideoCapture:
    def __init__(self, name):
        self.q = queue.Queue()  # Always create the queue
        self.cap = None
        # Use different backend for camera vs RTSP stream
        if isinstance(name, int):
            # For built-in camera, use default backend
            self.cap = cv2.VideoCapture(name)
            print(f"Initializing built-in camera {name}")
        else:
            # For RTSP streams, use FFmpeg with hardware acceleration
            self.cap = cv2.VideoCapture(name, cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
            print(f"Initializing RTSP stream {name}")

        # Check if camera opened successfully
        if not self.cap or not self.cap.isOpened():
            print(f"Error: Could not open camera/stream {name}")
            self.cap = None  # Mark as not available
            return

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        if not self.cap or not self.cap.isOpened():
            print("Camera not connected")
            return None
        if self.q.empty():
            print("No frames available in queue")
            return None
        return self.q.get()

    def __del__(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

#----------------------------------------------------------------------
# MODEL LOADING AND GLOBAL VARIABLES
#----------------------------------------------------------------------

# Remove model file if it exists and force fresh download
# if os.path.exists('yolov8n.pt'):
#      os.remove('yolov8n.pt')
model = YOLO('yolov8n.pt')

# Tracking variables
ml_results = None
track_history_ml = defaultdict(lambda: [])
track_history = list()

# Mouse interaction and selection variables
is_click = False
selected_point = None
drawing = False  # True while ROI (Region of Intrest) is actively being drawn by mouse
show_drawing = False  # True while ROI is drawn
p1, p2 = (0, 0), (0, 0)
click_point = []
select_obj_point = None

# Region of Interest (ROI) variables
bbox_ROI = (0, 0, 0, 0)
bbox_old = (0, 0, 0, 0)
state = 0
tracker_roi = None

#----------------------------------------------------------------------
# DRAWING AND VISUALIZATION FUNCTIONS
#----------------------------------------------------------------------

def draw_border(img, pt1, pt2, color, thickness, r, d):
    """
    Draw a fancy border with rounded corners around a detection

    Args:
        img: Image to draw on
        pt1: Top-left corner point (x1, y1)
        pt2: Bottom-right corner point (x2, y2)
        color: Border color in BGR format
        thickness: Line thickness
        r: Radius of rounded corners
        d: Length of corner lines
    """
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

def draw_center(frame, track_point, color=(255, 255, 255), _thickness=1):

    """
    Draw crosshair at a specific point in the frame

    Args:
        frame: Image to draw on
        track_point: Point coordinates (x, y)
        color: Line color in BGR format
        _thickness: Line thickness
    """
    h, w, _ = frame.shape
    cv2.line(frame, (track_point[0], 0), (track_point[0], h), color=color, thickness=_thickness)
    cv2.line(frame, (0, track_point[1]), (w, track_point[1]), color=color, thickness=_thickness)

#----------------------------------------------------------------------
# OBJECT DETECTION AND TRACKING FUNCTIONS
#----------------------------------------------------------------------

def return_nearest_obj(results, targetPose, trashhold=100):
    """
    Find the nearest detected object to the target position

    Args:
        results: Detection results from YOLO
        targetPose: Target position (x, y)
        trashhold: Maximum distance threshold

    Returns:
        Track_Object or None if no suitable object found
    """
    global model
    # Get array of all found objects
    detections = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
    classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
    track_ids = np.array(results[0].boxes.id.cpu(), dtype="int")

    # Check if detected something
    if len(detections) > 0:
        # Sort object detection by distance from target position
        distances = [np.sqrt((((x[0] + x[2])/2)-targetPose[0])**2 + (((x[1] + x[3])/2)-targetPose[1])**2) for x in detections]
        sorted_indices = np.argsort(distances)
        detections = detections[sorted_indices]

        # Found nearest object
        nearest_detection = detections[0]
        dist = np.sqrt((targetPose[0] - ((nearest_detection[0] + nearest_detection[2])/2))**2 + (targetPose[1] - ((nearest_detection[1] + nearest_detection[3])/2))**2)
        if dist > trashhold:
            print(" dist > trashhold")
            return None

        # Coords of nearest object
        x1, y1, x2, y2 = nearest_detection[:4]

        # ID and name of nearest object
        class_id = classes[sorted_indices][0]
        track_id = track_ids[sorted_indices][0]
        name_id = model.names[class_id]

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        ml_obj = Track_Object()
        ml_obj.name = name_id
        ml_obj.centerPoint = (center_x, center_y)
        ml_obj.class_id = class_id
        ml_obj.track_id = track_id
        ml_obj.xyxy = nearest_detection[:4]

        return ml_obj
    else:
        print("Object not found.")
        return None

def norm_2d_points(pts):
    """
    Normalize the coordinates of 2D points from an arbitrary quadrilateral

    Args:
        pts: Array of four 2D points

    Returns:
        Normalized points in [tl, br] format (flattened)
    """
    # Sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # Grab the left-most and right-most points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # Return the coordinates in top-left, bottom-right format
    return np.array([tl,br], dtype="int16").flatten().tolist()

def tracking_camera(frame, track_point, control, power=0.03):

    """
    Calculate camera movement to center on tracked object

    Args:
        frame: Current video frame
        track_point: Point to track (x, y)
        control: SIYI camera control object
        power: Movement power coefficient, lower is slower, smoother movement
    """

    h, w, _ = frame.shape
    dy, dx = int(h / 2), int(w / 2)
    # Calculate offset from center
    offset = (dx-track_point[0], dy-track_point[1])
    print("offset", offset)
    control.set_offset(yaw_off=offset[0], pitch_offset=offset[1], power=power)

#----------------------------------------------------------------------
# MOUSE INTERACTION HANDLING
#----------------------------------------------------------------------

def on_mouse(event, x, y, flags, userdata):
    """
    Handle mouse events for selecting objects or regions

    Args:
        event: Mouse event type
        x, y: Mouse coordinates
        flags: Additional flags
        userdata: User data
    """
    global click_point, is_click
    global p1, p2, drawing, show_drawing, bbox_ROI

    if event == cv2.EVENT_LBUTTONDOWN:
        # Left click down (select first point)
        drawing = True
        show_drawing = True
        p1 = x, y
        p2 = x, y
        is_click = True
        click_point = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        # Drag to second point
        if drawing:
            p2 = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        # Left click up (select second point)
        drawing = False
        p2 = x, y

    # Calculate rectangle from the two points
    p3 = (p1[0], p2[1])
    p4 = (p2[0], p1[1])
    
    pts = np.array([p1, p2, p3, p4])
    xyx1y1 = norm_2d_points(pts)
    w = int(xyx1y1[2] - xyx1y1[0])
    h = int(xyx1y1[3] - xyx1y1[1])
    bbox_ROI = (int(xyx1y1[0]), int(xyx1y1[1]), w, h)

#----------------------------------------------------------------------
# MAIN EXECUTION BLOCK
#----------------------------------------------------------------------

if __name__ == "__main__":

    dt = time()
    old_dt = dt
    currnt_time = time()    
    timer = 0
    is_bisy = False
    
    # Use RTSP stream or built-in camera (0)
    RTSP_URL = "rtsp://192.168.144.25:8554/main.264"
    siyi_cap = VideoCapture(RTSP_URL)


    # Create display window and set mouse callback
    cv2.namedWindow("Object detection with telemetry", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Object detection with telemetry", on_mouse)
    siyi_control = SIYIControl()

    while True:

        # Read frame from camera
        if is_click:
            selected_point = click_point
            is_click = False

        frame = siyi_cap.read()

        # Check if frame is valid and display Yolo detections using try-except
        try:
            if frame is None:
                raise ValueError("No frame received from camera")

            # If a ROI is selected, draw it
            if show_drawing:
                # Fix p2 to be always within the frame
                p2 = (
                    0 if p2[0] < 0 else (p2[0] if p2[0] < frame.shape[1] else frame.shape[1]),
                    0 if p2[1] < 0 else (p2[1] if p2[1] < frame.shape[0] else frame.shape[0])
                )
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)

            """
            
            ML detections is based on YOLOv8 with Bot-SORT tracker.
            
            ML parameters:
            
            persist=True - maintain object IDs across frames
            tracker="botsort.yaml" - use Bot-SORT tracker for better performance
            conf=0.1 - minimum confidence threshold for detections
            iou=0.5 - IOU threshold for tracking
            device="mps" - use Apple Silicon GPU if available, otherwise use !auto!
            
            """

            ml_results = model.track(frame, persist=True, tracker="botsort.yaml", conf=0.1, iou=0.5, device="mps")
            if ml_results:
                is_bisy = False

                # Process YOLO detection results
                if ml_results[0].boxes.id is not None:
                    bboxes = np.array(ml_results[0].boxes.xyxy.cpu(), dtype="int")
                    classes = np.array(ml_results[0].boxes.cls.cpu(), dtype="int")
                    track_ids = np.array(ml_results[0].boxes.id.cpu(), dtype="int")

                    # Find the object closest to the selected point
                    if selected_point:
                      print("selected_point", selected_point)
                      select_obj_point = return_nearest_obj(ml_results, selected_point, 100)
                    else:
                       select_obj_point = None

                    """
                    ----------------------------------------------------------------------
                                            VISUALIZE YOLO DETECTIONS
                    ----------------------------------------------------------------------
                    """

                    for cls, bbox, track_id in zip(classes, bboxes, track_ids):
                        (x, y, x2, y2) = bbox

                        # Handle selected object (RED)
                        if select_obj_point and track_id == select_obj_point.track_id:
                          # Draw tracking history path
                          track = track_history_ml[track_id]
                          center_x = int((x + x2) / 2)
                          center_y = int((y + y2) / 2)
                          track.append((int(center_x), int(center_y)))  # x, y center point
                          if len(track) > 10:  # retain 10 tracks
                              track.pop(0)
                          points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                          cv2.polylines(frame, [points], isClosed=False, color=(255, 0, 0), thickness=2)

                          # Draw selected object with RED border
                          draw_border(frame, (x,y), (x2, y2), (0,0,255), 2, 5, 10)  # Red border for selected object
                          cv2.circle(frame, select_obj_point.centerPoint, 5, (0, 0, 255), 5)  # Red center point
                          draw_center(frame, selected_point, color=(255, 0, 0), _thickness=1)
                        else:
                          # Draw regular detection with BLUE border
                          draw_border(frame, (x,y), (x2, y2), (255,0,0), 2, 5, 10)  # Blue border for YOLO detection

                        # Draw object label
                        name = "ID: %s %s " % (track_id, model.names[cls])
                        cv2.putText(frame, name, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 225), 2)

                        # Draw center point
                        center_x = int((x + x2) / 2)
                        center_y = int((y + y2) / 2)
                        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), 2)  # Blue center point for all YOLO detected objects



            """
            ----------------------------------------------------------------------
                                ACTIVE OBJECT TRACKING
            ----------------------------------------------------------------------
            """


            if tracker_roi:
                ret_tracking, bbox = tracker_roi.update(frame)
                if bbox_old != bbox:
                    bbox_old = bbox

                # Draw active tracking visualization (GREEN)
                if ret_tracking:
                    p1_ot = (int(bbox[0]), int(bbox[1]))
                    p2_ot = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    draw_border(frame, p1_ot, p2_ot, (0,255,0), 2, 5, 10)  # Green border for actively tracked object

                    # Calculate and store tracking position
                    (x, y, x2, y2) = (p1_ot[0], p1_ot[1], p2_ot[0], p2_ot[1])
                    center_x = int((x + x2) / 2)
                    center_y = int((y + y2) / 2)
                    track_history.append((int(center_x), int(center_y)))  # x, y center point

                    # Maintain tracking history
                    if len(track_history) > 10:  # retain 10 tracks for 10 frames
                        track_history.pop(0)

                    # Visualize tracking path
                    points = np.hstack(track_history).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 0), thickness=3)  # Green tracking history

                    # Control camera to follow tracked object
                    tracking_camera(frame, track_history[-1], siyi_control)
                    selected_point = track_history[-1]
                else:
                    # Handle tracking failure
                    if select_obj_point:
                        ret = tracker_roi.init(frame, select_obj_point.xywh())
                    else:
                       selected_point = None

                    cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

            # Display output window
            cv2.imshow("Object detection with telemetry", frame)

        except ValueError as e:
            print(f"Error: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error processing frame: {e}")
            continue

        #----------------------------------------------------------------------
        # KEY PRESS HANDLING
        #----------------------------------------------------------------------
        pressed = cv2.waitKey(1)

        # ENTER or SPACE - Start/reset active tracking
        if pressed in [13, 32]:
            # Clear red outline, only green remain
            drawing = False
            show_drawing = False

            # Reset existing tracker if active
            if tracker_roi:
                tracker_roi = None

            # Create appropriate tracker based on OpenCV version
            try:
                if OPENCV_MAJOR == 4 and OPENCV_MINOR >= 5:
                    # For OpenCV 4.5+ (new API)
                    tracker_roi = cv2.TrackerMIL.create()
                elif OPENCV_MAJOR == 4:
                    # For OpenCV 4.x (legacy module)
                    tracker_roi = cv2.legacy.TrackerMIL_create()
                elif OPENCV_MAJOR == 3:
                    # For OpenCV 3.x
                    tracker_roi = cv2.TrackerMIL_create()
                else:
                    # Fallback for very old versions
                    raise Exception("Unsupported OpenCV version for object tracking")
            except Exception as e:
                print(f"Error creating tracker: {str(e)}")
                print("Your OpenCV installation may not support object tracking")
                tracker_roi = None

            # Clear tracking history
            track_history.clear()

            # Initialize tracker with selected object or ROI
            if select_obj_point:
                # Initialize with ML-selected object
                ret = tracker_roi.init(frame, select_obj_point.xywh())
                # Clear selection point to prevent red outline from reappearing
                select_obj_point = None
            else:
                # Initialize with manually drawn ROI
                if bbox_ROI[2] == 0 or bbox_ROI[3] == 0:
                    print("ROI not selected")
                    drawing = False
                    show_drawing = False
                    track_history.clear()
                    selected_point = None
                    if tracker_roi:
                        tracker_roi = None
                else:
                    ret = tracker_roi.init(frame, bbox_ROI)

        # 'C' key - Cancel selection
        elif pressed in [ord('c'), ord('C')]:
            drawing = False
            show_drawing = False
            track_history.clear()
            selected_point = None
            if tracker_roi:
                tracker_roi = None

        # ESC key - Exit program
        elif pressed in [27]:
            drawing = False
            show_drawing = False
            track_history.clear()
            if tracker_roi:
                tracker_roi = None
            break


    # Cleanup on exit
    cv2.destroyAllWindows()
    siyi_cap.cap.release()
