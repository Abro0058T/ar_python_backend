import cv2
import numpy as np
import json
import os
from datetime import datetime

def detect_walls(image_path, params, output_dir="test_results"):
    """
    Detect walls with configurable edge detection parameters
    
    Args:
        image_path: Path to the input image
        params: Dictionary of parameters for edge detection
        output_dir: Directory to save output visualizations
    
    Returns:
        Dictionary with detected lines and visualization paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to load: {image_path}")
    
    # Extract parameters with defaults
    blur_kernel = params.get("blur_kernel", (5, 5))
    canny_low = params.get("canny_low", 30)
    canny_high = params.get("canny_high", 30)
    hough_threshold = params.get("hough_threshold", 70)
    min_line_length = params.get("min_line_length", 2)
    max_line_gap = params.get("max_line_gap", 1)
    use_adaptive_thresholds = params.get("use_adaptive_thresholds", False)
    
    # Edge detection pipeline
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    
    # Use adaptive thresholds if specified
    if use_adaptive_thresholds:
        median = np.median(blurred)
        sigma = 0.33
        canny_low = int(max(0, (1.0 - sigma) * median))
        canny_high = int(min(255, (1.0 + sigma) * median))
        
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, canny_low, canny_high)
    
    # Optional: enhance edges
    if params.get("dilate_edges", False):
        kernel = np.ones((2,2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Apply Hough transform to detect lines
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=hough_threshold, 
        minLineLength=min_line_length, 
        maxLineGap=max_line_gap
    )
    
    if lines is None:
        return {"error": "No lines detected", "lines": []}
    
    # Convert lines to a more manageable format
    line_list = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        line_list.append([int(x1), int(y1), int(x2), int(y2), int(length)])
    
    # Create visualization
    result_image = np.copy(image)
    for i, line in enumerate(line_list):
        x1, y1, x2, y2, _ = line
        cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Save edge detection visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_str = f"C{canny_low}_{canny_high}_H{hough_threshold}_L{min_line_length}_G{max_line_gap}"
    edges_path = os.path.join(output_dir, f"edges_{param_str}_{timestamp}.jpg")
    result_path = os.path.join(output_dir, f"lines_{param_str}_{timestamp}.jpg")
    
    cv2.imwrite(edges_path, edges)
    cv2.imwrite(result_path, result_image)
    
    # Save parameters to JSON for reference
    params_path = os.path.join(output_dir, f"params_{param_str}_{timestamp}.json")
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
    
    return {
        "lines": line_list,
        "count": len(line_list),
        "edges_path": edges_path,
        "result_path": result_path,
        "params_path": params_path,
        "params": params
    }

def test_parameter_combinations(image_path, test_cases):
    """
    Test multiple parameter combinations and compare results
    
    Args:
        image_path: Path to the blueprint image
        test_cases: Dictionary of named parameter sets to test
        
    Returns:
        Dictionary with results for each test case
    """
    results = {}
    
    print("=" * 60)
    print("WALL DETECTION PARAMETER COMPARISON")
    print("=" * 60)
    
    # Table header
    print(f"{'Test Case':<20} {'Detected Lines':<15} {'Parameters'}")
    print("-" * 60)
    
    for name, params in test_cases.items():
        print(f"Testing {name}...")
        result = detect_walls(image_path, params)
        results[name] = result
        
        # Format parameters for display
        canny_low = params.get("canny_low", "adaptive" if params.get("use_adaptive_thresholds") else 30)
        canny_high = params.get("canny_high", "adaptive" if params.get("use_adaptive_thresholds") else 30)
        hough_threshold = params.get("hough_threshold", 70)
        min_line_length = params.get("min_line_length", 2)
        max_line_gap = params.get("max_line_gap", 1)
        
        param_str = f"C:{canny_low},{canny_high} H:{hough_threshold} L:{min_line_length} G:{max_line_gap}"
        
        # Print in table format
        print(f"{name:<20} {result.get('count', 0):<15} {param_str}")
        print(f"  → Result saved to: {os.path.basename(result.get('result_path', ''))}")
    
    print("=" * 60)
    
    # Create a comparison summary image
    create_comparison_image(image_path, results)
    
    return results

def create_comparison_image(image_path, results):
    """Create a side-by-side comparison of all test results"""
    if not results:
        return
    
    # Determine the number of test cases and configure the grid
    n_tests = len(results)
    cols = min(3, n_tests)  # Max 3 columns
    rows = (n_tests + cols - 1) // cols  # Ceiling division
    
    # Read original image for reference
    orig_img = cv2.imread(image_path)
    h, w = orig_img.shape[:2]
    
    # Create a canvas for the comparison
    thumb_w = 500  # Width of each thumbnail
    thumb_h = int(h * thumb_w / w)  # Keep aspect ratio
    
    # Create canvas with space for titles and stats
    canvas = np.ones((rows * (thumb_h + 80), cols * (thumb_w + 20), 3), dtype=np.uint8) * 255
    
    # Place each result on the canvas
    i = 0
    for name, result in results.items():
        row = i // cols
        col = i % cols
        
        # Load the result image
        if "result_path" in result and os.path.exists(result["result_path"]):
            result_img = cv2.imread(result["result_path"])
            if result_img is not None:
                # Resize to thumbnail
                thumb = cv2.resize(result_img, (thumb_w, thumb_h))
                
                # Calculate position
                x = col * (thumb_w + 20)
                y = row * (thumb_h + 80)
                
                # Place thumbnail
                canvas[y:y+thumb_h, x:x+thumb_w] = thumb
                
                # Add title and stats
                title = name
                stats = f"Lines: {result.get('count', 0)}"
                
                cv2.putText(canvas, title, (x+10, y+thumb_h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(canvas, stats, (x+10, y+thumb_h+55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
        
        i += 1
    
    # Save the comparison image
    comparison_path = os.path.join("test_results", f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    cv2.imwrite(comparison_path, canvas)
    print(f"\nComparison image saved to: {comparison_path}")

# Example usage
if __name__ == "__main__":
    # Test image path - update this to your test blueprint
    test_image = "./test3.png"  # or use a path from command line args
    
    # Define test cases with different parameters
    test_cases = {
        "current": {  # Current parameters in index.py
            "blur_kernel": (5, 5),
            "canny_low": 30,
            "canny_high": 30,
            "hough_threshold": 70,
            "min_line_length": 2,
            "max_line_gap": 1
        },
        "adaptive_thresholds": {  # Using adaptive thresholds
            "blur_kernel": (5, 5),
            "use_adaptive_thresholds": True,
            "hough_threshold": 70,
            "min_line_length": 2,
            "max_line_gap": 1
        },
        "optimized_hough": {  # Better Hough parameters for architectural drawings
            "blur_kernel": (5, 5),
            "canny_low": 30,
            "canny_high": 90,
            "hough_threshold": 50,
            "min_line_length": 20,
            "max_line_gap": 10
        },
        "full_optimization": {  # Combination of adaptive thresholds and better Hough parameters
            "blur_kernel": (5, 5),
            "use_adaptive_thresholds": True,
            "hough_threshold": 50,
            "min_line_length": 20,
            "max_line_gap": 10,
            "dilate_edges": True
        }
    }
    
    # Run tests
    results = test_parameter_combinations(test_image, test_cases)