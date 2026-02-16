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


def merge_by_class(detections, classes, iou_threshold=0.5):

    final = []
    used = set()

    # Group detections by class
    grouped = {}

    for i, d in enumerate(detections):
        if d["class"] in classes:
            grouped.setdefault(d["class"], []).append((i, d))
        else:
            final.append(d)


    for cls, items in grouped.items():

        clusters = []

        for idx, det in items:

            if idx in used:
                continue

            used.add(idx)

            cluster = [det]

            for jdx, other in items:

                if jdx in used:
                    continue

                if compute_iou(det["bbox"], other["bbox"]) >= iou_threshold:
                    cluster.append(other)
                    used.add(jdx)

            clusters.append(cluster)

        # Keep largest from each cluster
        for cluster in clusters:

            best = max(
                cluster,
                key=lambda d: (d["bbox"][2] - d["bbox"][0]) *
                              (d["bbox"][3] - d["bbox"][1])
            )

            final.append(best)

    return final


class ObjectDetector:
    def __init__(self, 
                 person_model="yolov8s.pt",
                 cheat_model="YOLO_fineTune_v3.pt",

                 default_conf=0.5, 
                 person_conf=0.5,
                 book_conf=0.4,
                 phone_conf=0.6,
                 audio_conf=0.5,
                 ):

        self.person_model = YOLO(person_model)
        self.cheat_model = YOLO(cheat_model)

        # Thresholds
        self.default_conf = default_conf
        self.person_conf = person_conf
        self.class_thresholds = {
            "cell_phone": phone_conf,
            "book": book_conf,
            "headphone": audio_conf,
            "earbud": audio_conf,
        }

    def _run_model(self, model, frame, allowed_classes, default_conf):
        detections = []

        results = model(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                name = model.names[cls_id]
                conf = float(box.conf[0])

                if name not in allowed_classes:
                    continue

                threshold = self.class_thresholds.get(name, default_conf)

                if conf < threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "class": name,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2)
                })

        return detections

    def detect(self, frame):

        # 1️⃣ Person detection
        person_dets = self._run_model(
            self.person_model,
            frame,
            {"person"},
            self.person_conf
        )

        # 2️⃣ Cheating objects
        cheat_dets = self._run_model(
            self.cheat_model,
            frame,
            {"person", "cell_phone", "book", "headphone", "earbud"},
            self.default_conf
        )

        # Merge
        return person_dets + cheat_dets