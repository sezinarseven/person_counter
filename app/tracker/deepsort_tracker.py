import cv2
from matplotlib import pyplot as plt
import numpy as np
from deep_sort.tracker import Tracker
from tracking_helpers import ImageEncoder, create_box_encoder, read_class_names
from deep_sort.detection import Detection
from deep_sort import nn_matching, preprocessing


class v7_Tracker:

    def __init__(self, reID_model_path="./deep_sort/model_weights/mars-small128.pb", max_cosine_distance:float=0.4, nn_budget:float=None, nms_max_overlap:float=1.0, coco_names_path:str ="./data/coco.yaml",  ):
        self.coco_names_path = coco_names_path
        self.nms_max_overlap = nms_max_overlap
        self.class_names = read_class_names()

        self.encoder = create_box_encoder(reID_model_path, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric) 

    def track(self, preds, count_objects:bool=False, verbose:int = 0, frame=None):
        output=[]
        if preds is None:
            bboxes = []
            scores = []
            classes = []
            num_objects = 0
        else:
            bboxes = preds[:,:4]
            bboxes[:,2] = bboxes[:,2] - bboxes[:,0] 
            bboxes[:,3] = bboxes[:,3] - bboxes[:,1]

            scores = preds[:,4]
            classes = preds[:,-1]
            num_objects = bboxes.shape[0]
        
            names = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = self.class_names[class_indx]
                names.append(class_name)

            names = np.array(names)
            count = len(names)

            if count_objects:
                cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)

            # ---------------------------------- DeepSORT tacker work starts here ------------------------------------------------------------
            features = self.encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            boxs = np.array([d.tlwh for d in detections]) 
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       

            self.tracker.predict()
            self.tracker.update(detections) 

            for track in self.tracker.tracks: 
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()
                
                color = colors[int(track.track_id) % len(colors)] 
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + " : " + str(track.track_id),(int(bbox[0]), int(bbox[1]-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA)    
                out = [bbox[0], bbox[1], bbox[2], bbox[3], track.track_id, class_name]
                output.append(out)
                if verbose == 2:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                    
            # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------
            
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            return output