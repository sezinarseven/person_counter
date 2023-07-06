import random
import torch
import numpy as np

from models.experimental import attempt_load
from utils.torch_utils import TracedModel
from utils.datasets import letterbox
from utils.plots import plot_one_box
from utils.general import check_img_size, non_max_suppression, scale_coords
import cv2

class Yolov7Detector:

    def __init__(self, weights: str = "yolov7x.pt", trace: bool = True, image_size = 640):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.device="cpu"
        self.weights = weights
        self.model = attempt_load(self.weights, map_location=self.device) # Model Load FP32
        self.stride = int(self.model.stride.max()) 
        self.image_size = check_img_size(image_size, self.stride)
        
        if trace:
            self.model = TracedModel(self.model, self.device, self.image_size) #optimized model

        self.half = False

        if self.half:
            self.model.half() # not FP16

        self.names = self.model.module.names if hasattr(self.model , 'module') else self.model.names 

        color_values = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.colors = {i:color_values[i] for i in range(len(self.names))} 
    

    def detect(self, raw_image: np.ndarray, conf_thresh: float = 0.5, iou_thresh: float = 0.35, classes=None):
        
        # Run inference
        if self.device != 'cpu':
            self.model(torch.zeros(1, 3, self.image_size, self.image_size).to(self.device).type_as(next(self.model.parameters())))

        with torch.no_grad():
            image = letterbox(raw_image, self.image_size, stride=self.stride)[0]
            image = image[:, :, ::-1].transpose(2, 0, 1)
            
            image = np.ascontiguousarray(image) 
            
            image = torch.from_numpy(image).to(self.device)
            image = image.half() if self.half else image.float() 
            image /= 255.0
            if image.ndimension() == 3:
                image = image.unsqueeze(0)

            # Inference
            detections = self.model(image, augment=False)[0] 
            
            # Apply NMS
            detections = non_max_suppression(detections, conf_thresh, iou_thresh, classes = classes, agnostic=False)[0]
            
            detections[:, :4] = scale_coords(image.shape[2:], detections[:, :4], raw_image.shape).round()
            return detections.cpu().detach().numpy()
            
    def draw_detections(self, img:np.ndarray, preds): 
        for *bbox, conf, cls in preds:
            label = '%s %.2f' % (self.names[int(cls)], conf)
            plot_one_box(bbox, img, label=label[:-4], color=self.colors[int(cls)], line_thickness=3)


if __name__ == "__main__":
    detector = Yolov7Detector(weights="yolov7.pt")
    cap = cv2.VideoCapture("deneme.mp4")
    while True:
        ret, frame = cap.read()

        preds = detector.detect(frame)
        detector.draw_detections(frame, preds=preds)
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) & 0XFF == ord("q"): #0:image 1:video q:quit
            break