import numpy as np
from ultralytics import YOLO
import supervision as sv

# Initialize Costum Trained YOLOv8 model
model = YOLO("model/costum_trained_yolov8.pt")

# Initialize trackers and annotators
tracker = sv.ByteTrack(track_buffer=160)
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator(trace_length=130, color=sv.Color.RED)

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    # Perform object detection using YOLO model
    results = model(frame)[0]

    # Convert results to Detections object
    detections = sv.Detections.from_ultralytics(results)

    # Filter detections with confidence > 0.5
    detections = detections[detections.confidence > 0.5]

    # Update trackers with detections
    detections = tracker.update_with_detections(detections)

    # Generate labels for each detection
    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    # Annotate frame with bounding boxes
    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)

    # Annotate frame with labels
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)

    # Annotate frame with traces
    return trace_annotator.annotate(annotated_frame, detections=detections)

# Process video with the defined callback function
sv.process_video(
    source_path="./data/Soldier_Tracking_02.mp4",
    target_path="./runs/result_2.mp4",
    callback=callback
)