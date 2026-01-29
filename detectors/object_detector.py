from ultralytics import YOLO

def compute_iou(boxA, boxB):
    """
    boxA, boxB: (x1, y1, x2, y2)
    """

    #This computes the overlapping rectangle between two boxes.
    # If the boxes overlap, these coordinates define the intersection box
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    #Calculate area of intersection box
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    #IoU must be 0 if there is no intersection
    if inter_area == 0:
        return 0.0

    #computes individual box areas
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # IoU formula
    """
    Area(A∩B) / (Area(A)+Area(B)−Area(A∩B))
    """
    return inter_area / float(boxA_area + boxB_area - inter_area)

def merge_person_detections(detections, iou_threshold=0.5):
    """
    Merges overlapping 'person' detections using IoU clustering.
    Returns new detections list.
    """

    #Separate person vs non-person
    person_dets = [d for d in detections if d["class"] == "person"]
    other_dets = [d for d in detections if d["class"] != "person"]

    clusters = []
    """
    Each detection must: Join an existing cluster OR start a new cluster
    clusters = [
        [person_det1, person_det2],
        [person_det3],
    ]
    """

    for det in person_dets:
        added_to_cluster = False

        for cluster in clusters:
            # compare with representative box of cluster
            iou = compute_iou(det["bbox"], cluster[0]["bbox"])
            
            #If this box overlaps enough with an existing cluster, it belongs to the same person
            if iou >= iou_threshold:
                cluster.append(det)
                added_to_cluster = True
                break
        
        #detection does not overlap with any known person → new person
        if not added_to_cluster:
            clusters.append([det])

    # For each cluster, keep the largest box (best representative)
    merged_persons = []
    for cluster in clusters:
        largest = max(
            cluster,
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
        )
        merged_persons.append(largest)

    return other_dets + merged_persons

class ObjectDetector:
    def __init__(self, model_path="yolov8s.pt",confidence_threshold=0.5, book_and_mobile_confidence_threshold=0.3):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.class_thresholds = {
            "book": book_and_mobile_confidence_threshold,
            "cell phone": book_and_mobile_confidence_threshold,
            "default": confidence_threshold
        }
        

        #COCO Classes Data set on which YOLO is trained : Common objects in Context
        self.target_classes = {"cell phone", "person", "book",}

    def detect(self, frame):
        results = self.model(frame, verbose=False)

        detections=[]

        def appendDetections(class_name, confidence, box):                
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "class": class_name,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2)
            })

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                confidence = float(box.conf[0])

                if class_name not in self.target_classes:
                    continue
                
                threshold = self.class_thresholds.get(class_name, self.confidence_threshold)
                if confidence >= threshold:
                    appendDetections(class_name, confidence, box)

        return detections
    
    
