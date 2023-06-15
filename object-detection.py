from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "retinanet_resnet50_fpn_coco-eeacb38b.pth"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "car.jpg"), output_image_path=os.path.join(execution_path , "car_detected.jpg"), minimum_percentage_probability=30)
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "street.jpg"), output_image_path=os.path.join(execution_path , "street_detected.jpg"), minimum_percentage_probability=30)

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")
