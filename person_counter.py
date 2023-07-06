import cv2
import numpy as np
from app.detector.detection_yolo import Yolov7Detector
from app.tracker.deepsort_tracker import v7_Tracker
from shapely.geometry import Point, Polygon
from table_class import Plotter

def region_box(img, region_coords, color = (0, 255, 0)):
    cv2.rectangle(img, region_coords[0], region_coords[2], color, 3)

def get_pred_coords(predictions):
    mid_x = (predictions[0] + predictions[2]) / 2
    mid_y = (predictions[1] + predictions[3]) / 2
    return mid_x, mid_y

if __name__ == "__main__":
    detector = Yolov7Detector()
    tracker = v7_Tracker()
    plotter = Plotter(title="Person Counter")
    cap = cv2.VideoCapture("1.mp4")
    current_ids = []
    all_ids = []
    reg_tl = (300, 300)
    reg_tr = (1000, 300)
    reg_bl = (300, 800)
    reg_br = (1000, 800)

    region_coords = np.array([reg_tl, reg_bl, reg_br, reg_tr, reg_tl])
    frame_counter = 0
    total_people_in_region = []
    region_counter = []
    while cap.isOpened():
        ret, frame = cap.read()
        frame_counter = frame_counter + 1
        region_box(frame, region_coords, color=(0, 255, 0))

        if not ret:
            break
        
        predictions = detector.detect(frame)
        predictions = tracker.track(preds = predictions, frame = frame)
        i=0
        for pred in predictions:
            if pred[4] not in all_ids:
                all_ids.append(pred[4])
                current_ids.append(pred[4])
            mid_x, mid_y = get_pred_coords(pred)
            mid_point = Point(mid_x, mid_y)

            in_reg = mid_point.within(Polygon(region_coords))
            
            if in_reg and pred[4] not in region_counter and pred[5]=="person":
                region_counter.append(pred[4])
                if pred[4] not in total_people_in_region:
                    total_people_in_region.append(pred[4])
                cv2.putText(frame, "ID " + str(pred[4]) + " -> entered in the box", (70, 1000+i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (204, 153, 255), 2)
                i=i-30
            
            elif not in_reg and pred[4] in region_counter:
                region_counter.remove(pred[4])
                cv2.putText(frame, "ID " + str(pred[4]) + " -> left the box", (70, 1000+i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (204, 153, 255), 2)
                i=i-30

        if frame_counter % 5 == 0:
            for id in all_ids:
                if id not in current_ids:
                    all_ids.remove(id)
                    if id in region_counter:
                        region_counter.remove(id)
                        print("ID " + str(id) + " -> deleted")
                    i=i-30
            current_ids = []

        if region_counter != []:
            region_box(frame, region_coords, color=(0, 0, 255))
        
        data = ["Total People In the Region", "Current People In the Region",str(len(total_people_in_region)), str(len(region_counter))]

        plotter.plot_table(frame, cell_data=data, num_rows=2, num_columns=2)

        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break