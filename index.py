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


def visualize_merged_lines(original_lines, merged_lines, merge_map=None, filename='wall_merging_visual.jpg'):
    """
    Create a visualization of original vs merged lines with detailed information.
    
    Args:
        original_lines: List of original lines [x1, y1, x2, y2, length]
        merged_lines: List of merged lines [x1, y1, x2, y2, length]
        merge_map: Optional dict mapping merged line indices to original line indices
        filename: Output filename for visualization
    """
    # Create a white canvas
    height, width = 1500, 1500
    visual = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw original lines in light blue
    for i, line in enumerate(original_lines):
        x1, y1, x2, y2, _ = line
        cv2.line(visual, (x1, y1), (x2, y2), (255, 200, 100), 1)  # Light orange
        cv2.putText(visual, str(i), (x1 - 15, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    # Draw merged lines
    for i, line in enumerate(merged_lines):
        x1, y1, x2, y2, _ = line
        # Draw the line in red
        cv2.line(visual, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red
        
        # Draw endpoints as circles
        cv2.circle(visual, (x1, y1), 4, (0, 0, 255), -1)
        cv2.circle(visual, (x2, y2), 4, (0, 0, 255), -1)
        
        # Add merged line number
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(visual, f"M{i}", (mid_x + 10, mid_y + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # If merge map is provided, show which original lines were merged
        if merge_map and i in merge_map:
            origin_indices = merge_map[i]
            if len(origin_indices) > 1:  # Only add label if actually merged
                label = f"Merged: {','.join(map(str, origin_indices))}"
                cv2.putText(visual, label, (mid_x + 10, mid_y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    
    # Add legend
    cv2.putText(visual, "Orange: Original lines", (20, height - 60), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 150, 255), 2)
    cv2.putText(visual, "Red: Merged lines", (20, height - 30), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add statistics
    cv2.putText(visual, f"Original lines: {len(original_lines)}", (20, 30), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(visual, f"Merged lines: {len(merged_lines)}", (20, 60), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Save the visualization
    cv2.imwrite(filename, visual)
    print(f"Visualization saved to {filename}")
    
    return visual


# parameters - (degrees , pixels)
def merge_similar_walls(lines, angle_threshold=10, distance_threshold=40):
    """
    Merge similar wall lines that likely represent the same wall.
    
    Args:
        lines: List of lines [x1, y1, x2, y2, length]
        angle_threshold: Maximum angle difference in degrees to consider lines parallel
        distance_threshold: Maximum distance to consider lines for merging
    
    Returns:
        List of merged lines
    """
    if not lines or len(lines) < 2:
        return lines
    
    # Create a copy of original lines (unsorted) for visualization
    original_lines = lines.copy()
    
    # Sort lines by length (descending)
    lines = sorted(lines, key=lambda x: x[4], reverse=True)
    
    # Map from original indices to sorted indices
    original_to_sorted = {i: idx for idx, i in enumerate(sorted(range(len(lines)), 
                                                               key=lambda i: lines[i][4], 
                                                               reverse=True))}
    
    # Track which lines have been merged
    merged = [False] * len(lines)
    result = []
    merge_map = {}  # Maps result index to list of original indices
    
    def calculate_angle(line):
        """Calculate the angle of a line in degrees"""
        x1, y1, x2, y2, _ = line
        if x2 - x1 == 0:  # Vertical line
            return 90
        return abs(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))
    
    def lines_are_close_and_parallel(line1, line2):
        """Check if two lines are close and parallel enough to merge"""
        x1, y1, x2, y2, _ = line1
        x3, y3, x4, y4, _ = line2
        
        # Check angles first (cheaper calculation)
        angle1 = calculate_angle(line1)
        angle2 = calculate_angle(line2)
        angle_diff = min(abs(angle1 - angle2), 180 - abs(angle1 - angle2))
        
        if angle_diff > angle_threshold:
            return False
        
        # For horizontal lines (similar y-coordinates)
        if abs(y1 - y2) < 15 and abs(y3 - y4) < 15:
            # Check if y-values are close
            y_diff = abs((y1 + y2) / 2 - (y3 + y4) / 2)
            if y_diff > distance_threshold / 2:  # Stricter for parallel walls
                return False
                
            # Check if x-ranges overlap or are close
            x_min1, x_max1 = min(x1, x2), max(x1, x2)
            x_min2, x_max2 = min(x3, x4), max(x3, x4)
            
            # Ranges overlap or are within threshold distance
            return (x_min2 <= x_max1 + distance_threshold and 
                    x_min1 <= x_max2 + distance_threshold)
        
        # For vertical lines (similar x-coordinates)
        if abs(x1 - x2) < 15 and abs(x3 - x4) < 15:
            # Check if x-values are close
            x_diff = abs((x1 + x2) / 2 - (x3 + x4) / 2)
            if x_diff > distance_threshold / 2:  # Stricter for parallel walls
                return False
                
            # Check if y-ranges overlap or are close
            y_min1, y_max1 = min(y1, y2), max(y1, y2)
            y_min2, y_max2 = min(y3, y4), max(y3, y4)
            
            # Ranges overlap or are within threshold distance
            return (y_min2 <= y_max1 + distance_threshold and 
                    y_min1 <= y_max2 + distance_threshold)
        
        # For diagonal lines, use midpoint distance
        mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
        mid2 = ((x3 + x4) / 2, (y3 + y4) / 2)
        distance = np.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)
        return distance < distance_threshold
    
    def merge_lines_with_consistent_dimensions(line1, line2):
        """
        Merge two lines while maintaining consistent dimensions based on orientation
        """
        x1, y1, x2, y2, _ = line1
        x3, y3, x4, y4, _ = line2
        
        # For horizontal lines, preserve y-coordinate
        if abs(y1 - y2) < 15 and abs(y3 - y4) < 15:
            # Use y-coordinate from longer line for consistency
            use_y = y1 if line1[4] >= line2[4] else y3
            
            # Find min and max x values
            x_values = [x1, x2, x3, x4]
            min_x = min(x_values)
            max_x = max(x_values)
            
            # Create new consistent horizontal line
            length = max_x - min_x
            return [int(min_x), int(use_y), int(max_x), int(use_y), int(length)]
            
        # For vertical lines, preserve x-coordinate
        elif abs(x1 - x2) < 15 and abs(x3 - x4) < 15:
            # Use x-coordinate from longer line for consistency
            use_x = x1 if line1[4] >= line2[4] else x3
            
            # Find min and max y values
            y_values = [y1, y2, y3, y4]
            min_y = min(y_values)
            max_y = max(y_values)
            
            # Create new consistent vertical line
            length = max_y - min_y
            return [int(use_x), int(min_y), int(use_x), int(max_y), int(length)]
            
        # For diagonal or other lines, find the two farthest points
        else:
            points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            max_dist = 0
            max_points = None
            
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    dist = np.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                    if dist > max_dist:
                        max_dist = dist
                        max_points = (points[i], points[j])
            
            if max_points:
                new_line = [int(max_points[0][0]), int(max_points[0][1]), 
                            int(max_points[1][0]), int(max_points[1][1]),
                            int(max_dist)]
                return new_line
                
            return line1  # Fallback
    
    # Process all lines
    for i in range(len(lines)):
        if merged[i]:
            continue
            
        current_line = lines[i]
        merged_line_indices = [i]  # Keep track of which lines were merged
        
        # Find all similar lines to merge with current line
        for j in range(i+1, len(lines)):
            if merged[j]:
                continue
                
            if lines_are_close_and_parallel(current_line, lines[j]):
                # Print merging information for debugging
                print(f"Merging line {i}: {current_line} with line {j}: {lines[j]}")
                current_line = merge_lines_with_consistent_dimensions(current_line, lines[j])
                merged_line_indices.append(j)
                merged[j] = True
        
        merged[i] = True
        result.append(current_line)
        
        # Map this result to the original indices
        result_idx = len(result) - 1
        merge_map[result_idx] = merged_line_indices
    
    # Create a visualization of the merging process
    visualize_merged_lines(original_lines, result, merge_map, 'wall_merging_visual.jpg')
    
    print(f"Original lines: {len(lines)}, Merged lines: {len(result)}")
    print(f"Merge mapping: {merge_map}")
    
    return result


@app.route("/wall", methods=["POST"])
@cross_origin()
def detect_lines_and_measure  (image_path="./test3.png"):
    #(image_path="./test.png"):
    #(image_path="./test3.png"):


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
        return jsonify({"error": "No lines detected", "furniture": furnitureCoordinate})  
    
    print("=== Hough Lines Transform Output ===")
    print(f"Detected {len(lines)} lines")

    line_image = np.copy(image)
    linesCo = []
    
    for i, line in enumerate(lines):
        print(f"Line {i}: {line[0]}")
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        linesCo.append([int(x1), int(y1), int(x2), int(y2), int(length)])

    # Apply the wall merging algorithm (degrees, pixels)
    merged_walls = merge_similar_walls(linesCo, angle_threshold=10, distance_threshold=30)
    
    # Create visualization of original vs merged walls
    merged_image = np.copy(image)
        # Create a detailed visualization showing the merging process
    merging_visual = np.copy(image)
    
    # Draw original lines with numbers
    for i, line in enumerate(linesCo):
        x1, y1, x2, y2, _ = line
        cv2.line(merging_visual, (x1, y1), (x2, y2), (255, 200, 0), 1)  # Orange
        cv2.putText(merging_visual, str(i), (x1 - 10, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Draw merged lines with thicker red lines
    for i, wall in enumerate(merged_walls):
        x1, y1, x2, y2, _ = wall
        cv2.line(merging_visual, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red
        # Label with M for merged
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        cv2.putText(merging_visual, f"M{i}", (mid_x + 5, mid_y + 5), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Save this detailed visualization
    cv2.imwrite('wall_merging_details.jpg', merging_visual)
    
    finalResult = {"wall": merged_walls, "furniture": furnitureCoordinate}
    print(f"Original walls: {len(linesCo)}, Merged walls: {len(merged_walls)}")
    return json.dumps(finalResult)


@app.route("/", methods=["POST"])
@cross_origin()
def detect_furniture(image_path="./blueprint4.png"):
    if request.method == "POST":
        f = request.files["file"]
        f.save(f.filename)

    image = cv2.imread(f.filename)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    print(image)
    results = model.predict(f.filename)
    # results = model.predict(image_path)
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


