import cv2 # imports OpenCV library for reading videos, edge detection and drawing lines
import numpy as np # imports numpy library for numerical operations
import csv # imports the csv library for writing detected lines to csv for later analysis
from math import atan2, degrees, radians, sqrt # necessary for angle anf distance computations

# The Following Functions Below are Helper Functions

def distance(point1, point2):

    """
    Calculates the Euclidean distance between two points.
    Each point is represented as a tuple (x, y).

    """    
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def are_close(line1, line2, threshold=15):
    
    """
    Checks if two lines are close to each other based on a distance threshold.
    This function is used to filter out duplicate or overlapping line detections.
    """    
    for point1 in line1:
        for point2 in line2:
            if distance(point1, point2) < threshold:
                return True
    return False

def save_to_csv(data):

    """
    Saves detected lines to a CSV file named 'filtered_lines.csv'.
    - If only 2 lines are detected, a placeholder middle line (NaN values) is added.
    - If 3 lines are detected, the lines are ordered by their x-coordinates (left to right)
      and labeled as line1, middleline, and line2 accordingly.
    """

    with open('filtered_lines.csv', 'a', newline='') as csvfile:
        # writer.writerow(['LINE1', 'LINE2', 'MIDDLELINE'])        
        writer = csv.writer(csvfile)

        # This block of code handles the case when there are exactly two detected line
        if len(data) == 2:
            if data[0][0][0] < data[1][0][0]:
                line1 = data[0]
                line2 = data[1]
                middleline = [(np.nan,np.nan),(np.nan,np.nan)]
                writer.writerow([line1, line2,middleline])
            else:
                line1 = data[1]
                line2 = data[0]
                middleline = [(np.nan,np.nan),(np.nan,np.nan)]
                writer.writerow([line1, line2,middleline])
        
        # This block of code handles the case when there are exactly three detected line
        if len(data) == 3:

            # sorts the lines by their x-coordinate (i.e. leftmost to rightmost)
            if data[0][0][0] < data[1][0][0] and data[0][0][0] < data[2][0][0]:
                if data[1][0][0] < data[2][0][0]:
                    line1 = data[0]
                    line2 = data[2]
                    middleline = data[1] 
                    writer.writerow([line1, line2,middleline])
                else:
                    line1 = data[0]
                    line2 = data[1]
                    middleline = data[2] 
                    writer.writerow([line1, line2,middleline])
            
            elif data[1][0][0] < data[0][0][0] and data[1][0][0] < data[2][0][0]:
                if data[0][0][0] < data[2][0][0]:
                    line1 = data[1]
                    line2 = data[2]
                    middleline = data[0] 
                    writer.writerow([line1, line2,middleline])
                else:
                    line1 = data[1]
                    line2 = data[0]
                    middleline = data[2] 
                    writer.writerow([line1, line2,middleline])
            
            if data[2][0][0] < data[0][0][0] and data[2][0][0] < data[1][0][0]:
                if data[0][0][0] < data[1][0][0]:
                    line1 = data[2]
                    line2 = data[1]
                    middleline = data[0] 
                    writer.writerow([line1, line2,middleline])
                else:
                    line1 = data[2]
                    line2 = data[0]
                    middleline = data[1] 
                    writer.writerow([line1, line2,middleline])
                
def filter(lines):

    """
    Filters out duplicate or very close lines using the are_close() function.
    Returns a list of unique, distinct lines.
    """

    filtered_lines = []
    for i, line1 in enumerate(lines):
        keep_line = True
        for j, line2 in enumerate(lines):
            if i != j and are_close(line1, line2):
                keep_line = False
                break
        if keep_line:
            filtered_lines.append(line1)
    return filtered_lines

def region_of_interest(img):

    """
    Applies a mask to keep only the region of interest (ROI).
    This helps focus line detection on the road/lane area.
    The ROI is defined as a polygonal area within the image.
    """    

    height = img.shape[0]
    polygons = np.array([
    [(0,300),(0,600),(600,600),(600,300)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image
    
def func_canny(img):

    """
    Applies the Canny edge detection algorithm to the input image.
    Steps:
      1. Converts image to grayscale.
      2. Applies Gaussian blur to reduce noise.
      3. Detects edges using Canny.
    """    

    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),1)
    canny = cv2.Canny(blur, 100, 150)
    return canny

def get_angle(point_1, point_2):
    
    """
    Compute the angle (in degrees) between two points.
    The angle helps determine the line orientation (vertical, horizontal, etc.).
    """

    angle = atan2(point_1[1] - point_2[1], point_1[0] - point_2[0])
    #Optional
    angle = degrees(angle)
    # OR
    #angle = radians(angle)
    return angle

def draw_the_lines(img,lines): 
    
    """
    Draw lines on the image based on angle thresholds and filter overlapping ones.
    Also saves valid lines to CSV if at least 2 are detected.
    """

    imge=np.copy(img)
    xs = []     # this line store the detected lines' coordinates
    for line in lines:  
        for x1,y1,x2,y2 in line:
            point_1 = (x1,y1)
            point_2 = (x2,y2)            
            angle = get_angle(point_1, point_2)

            # keeps lines within a specific angular range 
            if (angle > 95 and angle < 150) or (angle > -150 and angle < -95):
            # if (angle > 90 and angle < 170) or (angle > -170 and angle < -90):
                xs.append([point_1,point_2])
                # print(angle)                                    
                cv2.line(imge,(x1,y1),(x2,y2),(0,255,0),thickness=5)
    
    # filters out similar/duplicate lines
    filtered_lines = filter(xs)

    # saves filteres lines to a csv file if at least 2 lines are detected
    if len(filtered_lines) >= 2:
        if len(filtered_lines) == 2 or len(filtered_lines) == 3:
            save_to_csv(filtered_lines)
            print(filtered_lines)
    return imge


# The MAIN processing pipeline

cap = cv2.VideoCapture('test.mp4') # loads the input video

while cap.isOpened():
    success, frame = cap.read()  # reads frame by frame

    if success:
        # frame = cv2.imread("sls_road.jpg")

        # resizes frame for consistent processing
        small_img = cv2.resize(frame,(800,600))
        lane_image = np.copy(small_img)

        # The First Step: Detects Edges
        canny = func_canny(lane_image)

        # The Second Step: Crops Images to a region of interest (ROI)
        cropped_image = region_of_interest(canny)

        try:

            # The Third Step: Detects lines using Probabilistic Hough Transform
            lines_hough = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength = 150,maxLineGap = 200)

            # The Fourth Step: Draws and filters detected lines
            lane_image = draw_the_lines(lane_image,lines_hough)         
            
            #averaged_lines = generate_average_lines(lane_image,lines_hough)
            #line_image = display_lines(lane_image,averaged_lines)
            #combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
            
        except Exception as e:
            print(e)
        
        # The Fifth Step: Displays cropped and final processed images
        cv2.imshow("cropped image", cropped_image)
        cv2.imshow("image", lane_image)

        # Alternative option to press "q" on keyboard to quit the program
        if cv2.waitKey(1) == ord('q'):
            break
        
# Releases resources
cap.release()
cv2.destroyAllWindows()  