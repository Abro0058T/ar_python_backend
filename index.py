from flask import Flask ,jsonify
import cv2
import numpy as np
from flask import *

from flask_cors import CORS, cross_origin
from distutils.log import debug
from fileinput import filename
from ultralytics import YOLO

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "http://localhost:5173"}}, methods=["GET", "POST"], allow_headers=["Content-Type"])

# app.config["CORS_HEADERS"] = 'Content-Type'
# app.run(use_reloader=False)
model = YOLO("best.pt")


@app.route("/wall", methods=["POST"])
@cross_origin()
def detect_lines_and_measure(image_path="./test2.png"):
    # response.headers.add("Access-Control-Allow-Origin", "*")
    # if request.method == "POST":
    #     f = request.files["file"]
    #     f.save(f.filename)

    # image = cv2.imread(f.filename)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    # results = model.predict(image_path)
    results= model(image)
    # print(len(results[0].boxes))
    furnitureCoordinate = []

    for furni in results[0].boxes:
        # return f"{furni}"
        print(f"{furni.xyxy} this is furni")
        furnitureCoordinate.append({
            "coordinate" : [int(np.array(furni.xyxy)[0][0]),int(np.array(furni.xyxy)[0][1]),int(np.array(furni.xyxy)[0][2]),int(np.array(furni.xyxy)[0][3])],
            "name": int(np.array(furni.cls)[0])
        })
    furniture = {
        "furnitureCount": len(results[0].boxes),
        "furnitureCoordinate": furnitureCoordinate
    }

    # Get the results
    boxes = results[0].boxes.xyxy  # Bounding boxes in (x1, y1, x2, y2) format
    confidences = results[0].boxes.conf  # Confidence scores
    labels = results[0].boxes.cls  # Class labels

    # Class names (COCO dataset example)
    # class_names = model.names

    # Draw bounding boxes on the image
    for box, confidence, label in zip(boxes, confidences, labels):
        x1, y1, x2, y2 = map(int, box)
        # class_name = class_names[int(label)]
        color = (255, 255, 255)  # Color for bounding box (green)

        # Draw the bounding box
        cv2.rectangle(image, (x1+10, y1+10), (x2-10, y2-10), color, -1)

        # Add label and confidence score
        # text = f"{class_name} {confidence:.2f}"
        # cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    # walls 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 30, 30)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 70, minLineLength=2, maxLineGap=1)

    if lines is None:
        print("No lines detected.")
        return

    line_image = np.copy(image)
    linesCo = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        linesCo.append([int(x1), int(y1), int(x2),int(y2), int(length)])

    finalResult = {"wall": linesCo, "furniture": furnitureCoordinate}
    print(f"{finalResult}")
    return json.dumps(finalResult)


@app.route("/", methods=["POST"])
@cross_origin()
def detect_furniture(image_path="./blueprint4.png"):
    # if request.method == "POST":
    #     f = request.files["file"]
    #     f.save(f.filename)

    # image = cv2.imread(f.filename)
    # if image is None:
    #     raise ValueError("Image not found or unable to load.")
    # print(image)
    # results = model.predict(f.filename)
    results = model.predict(image_path)
    # print(len(results[0].boxes))
    furnitureCoordinate = []

    for furni in results[0].boxes:
        # return f"{furni}"
        print(f"{furni.xyxy} this is furni")
        furnitureCoordinate.append({
            "coordinate" : [int(np.array(furni.xyxy)[0][0]),int(np.array(furni.xyxy)[0][1]),int(np.array(furni.xyxy)[0][2]),int(np.array(furni.xyxy)[0][3])],
            "name": int(np.array(furni.cls)[0])
        })
    furniture = {
        "furnitureCount": len(results[0].boxes),
        "furnitureCoordinate": furnitureCoordinate
    }
    print(f"{furniture}")
    return json.dumps(furniture)


if __name__ == "__main__":
    app.run(debug=True)
