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
    print("here")
    # Load the image
    image = cv2.imread(f.filename)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)


    # Detect edges using Canny edge detector
    edges = cv2.Canny(blurred, 30, 30)


    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 170 , 110, minLineLength=2, maxLineGap=1)

    if lines is None:
        print("No lines detected.")
        return


    line_image = np.copy(image)
    # print(lines)
    linesCo = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Compute line length
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        print(f"Line from ({x1-400}, {y1-500}) to ({x2-400}, {y2-500}) with length {length:.2f}")
        linesCo.append([x1-400, y1-500, x2-400, y2-500, length])
    return (f"{linesCo}")

if __name__ == '__main__':
    app.run(debug=True)


# load image using cv2....and do processing.

