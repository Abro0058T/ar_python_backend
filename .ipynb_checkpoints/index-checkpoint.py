from flask import Flask
import cv2
import numpy as np
from flask import *
from distutils.log import debug 
from fileinput import filename  

app  = Flask(__name__)

@app.route('/', methods=['POST'])
def detect_lines_and_measure(image_path="./blueprint7.png"):
    if request.method == 'POST':
        f=request.files['file']
        f.save(f.filename)

    image = cv2.imread(f.filename)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 30, 30)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180 , 70, minLineLength=2, maxLineGap=1)

    if lines is None:
        print("No lines detected.")
        return


    line_image = np.copy(image)
    linesCo = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        linesCo.append([x1, y1, x2, y2, length])
    return (f"{linesCo}")

if __name__ == '__main__':
    app.run(debug=True)

