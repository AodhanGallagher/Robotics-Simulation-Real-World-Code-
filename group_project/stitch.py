import math
import os
import numpy as np
import cv2
import cv2 as cv

def find_planet(img, stitched_img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    dilated = cv2.dilate(edges, None)

    circles = cv2.HoughCircles(dilated, cv2.HOUGH_GRADIENT, 1.2, 100, param1=50, param2=30, minRadius=0, maxRadius=150)
    if circles is None:
        return
    
    circles = np.uint16(np.around(circles))

    midpoint = None
    for i in circles[0,:]:
        cv2.circle(stitched_img,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(stitched_img,(i[0],i[1]),2,(0,0,255),3)
        midpoint = (i[0], i[1])
    return midpoint

# This function removes the border and fixes the skew. 
def isolate_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find Corners 
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    dst = cv.dilate(dst,None)

    corners = np.argwhere(dst > 0.1 * dst.max())

    # Find the corners of the border, we will use this to rotate our image and remove the borders.
    top_left_corner = corners[corners.sum(axis=1).argmin()]
    top_right_corner = corners[(corners[:,1]-corners[:,0]).argmax()]
    bottom_left_corner = corners[(corners[:,0]-corners[:,1]).argmax()]
    bottom_right_corner = corners[corners.sum(axis=1).argmax()]

    # Calculate the level of skew. (The gradient between the top left corner and the top right corner should be 0)
    gradient = (top_left_corner[0] - top_right_corner[0]) / (top_left_corner[1] - top_right_corner[1])
    angle = np.arctan(gradient * 180 / np.pi)

    # Rotate from the center.
    center = (img.shape[1]//2, img.shape[0]//2)
    M = cv.getRotationMatrix2D(center, angle, 1)
    img = cv.warpAffine(img, M, (img.shape[1], img.shape[0]))

    # We also want to apply the rotation to the corner points so we can use them to get rid of the border.
    top_left_corner = np.dot(M, np.array([top_left_corner[0], top_left_corner[1], 1])).astype(int)
    top_right_corner = np.dot(M, np.array([top_right_corner[0], top_right_corner[1], 1])).astype(int)
    bottom_left_corner = np.dot(M, np.array([bottom_left_corner[0], bottom_left_corner[1], 1])).astype(int)
    bottom_right_corner = np.dot(M, np.array([bottom_right_corner[0], bottom_right_corner[1], 1])).astype(int)

    # Remove the border.
    min_y = min(top_left_corner[0], top_right_corner[0]) + 10
    max_y = max(bottom_left_corner[0], bottom_right_corner[0]) - 10
    min_x = min(top_left_corner[1], bottom_left_corner[1]) + 5
    max_x = max(top_right_corner[1], bottom_right_corner[1]) - 5

    img = img[min_y:max_y, min_x:max_x]

    return img


# This function uses BFMatcher to match where the stars align. We then calculate the average distance between them all
# to know where to stitch the images.
def stitch(img1, img2):
    # Initial resize to ensure both images start with the same dimensions
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Remove skew and borders.
    img1 = isolate_image(img1)
    img2 = isolate_image(img2)

    # Resize again if dimensions differ after isolation
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Sift feature detection and description
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default settings
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    avg_diff_x, avg_diff_y = 0, 0

    # Filtering good matches based on the ratio test
    for m, n in matches:
        if m.distance < 0.30 * n.distance:
            good.append(m)
            img1_idx = m.queryIdx
            img2_idx = m.trainIdx
            x1, y1 = kp1[img1_idx].pt
            x2, y2 = kp2[img2_idx].pt
            avg_diff_x += x2 - x1
            avg_diff_y += y2 - y1

    if len(good) == 0:
        return None

    avg_diff_x /= len(good)
    avg_diff_y /= len(good)

    # Adjust image dimensions based on average differences calculated from matches
    new_width = max(img1.shape[1] + abs(int(avg_diff_x)), img2.shape[1] + int(avg_diff_x))
    new_height = max(img1.shape[0] + abs(int(avg_diff_y)), img2.shape[0] + int(avg_diff_y))

    # Create new image with appropriate dimensions
    new_image = np.zeros((new_height, new_width, 3), np.uint8)

    # Copy img1 to the new_image
    new_image[:img1.shape[0], :img1.shape[1]] = img1

    # Calculate start points for img2 based on average offsets
    start_y = max(int(avg_diff_y), 0)
    end_y = start_y + img2.shape[0]
    start_x = max(-int(avg_diff_x), 0)
    end_x = start_x + img2.shape[1]

    # Ensure we don't go out of bounds
    end_y = min(end_y, new_image.shape[0])
    end_x = min(end_x, new_image.shape[1])

    # Place img2 in the new image considering boundaries
    new_image[start_y:end_y, start_x:end_x] = img2[:end_y-start_y, :end_x-start_x]

    return new_image

def find_diameter(mask):
     # Ensure mask is of type uint8
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Find contours directly
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0  # No contours found

    # Select the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Initialize minimum and maximum coordinates on x and y
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    # Iterate through all points in the largest contour
    for point in largest_contour:
        x, y = point[0][0], point[0][1]
        # Update min and max x and y
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y

    # Calculate the maximum distances in the horizontal and vertical directions
    horizontal_diameter = max_x - min_x
    vertical_diameter = max_y - min_y

    return max(horizontal_diameter, vertical_diameter)


def calculate_distance(actual_size, apparent_size):
    distance_km = actual_size / (2 * math.tan(math.radians(apparent_size / 2)))
    return distance_km


def calculate_angular_size(pixels, image_width, fov_degrees):
    """
    Calculate the angular size in radians from pixels using the field of view.
    """
    fov_radians = math.radians(fov_degrees)
    angular_size = (pixels / image_width) * fov_radians
    return angular_size

def calculate_distance_from_angular_size(diameter_km, angular_size_radians):
    """
    Calculate the distance to an object based on its real diameter and its angular size in radians.
    """
    distance_km = diameter_km / (2 * math.tan(angular_size_radians / 2))
    return distance_km


def main_stitch(img1_Path, img2_Path, debug=False):
    # Set images
    img1 = cv2.imread(img1_Path)
    img2 = cv2.imread(img2_Path)

    # Get the stitched image.
    stitched_img = stitch(img1, img2)

    # Save the stitched image
    filepath = os.path.join("group2", "panorama.png")
    cv2.imwrite(filepath, stitched_img)
    
    hsv = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2HSV)

    # Define the HSV range for the Earth
    lower_earth = np.array([1, 18, 10])
    upper_earth = np.array([179, 255, 255])

    # Define the HSV range for the Moon
    lower_moon = np.array([0, 0, 10])
    upper_moon = np.array([165, 23, 148])

    # Define the HSV range for stars.
    lower_star = np.array([0, 0, 175])
    upper_star = np.array([180, 255, 255])

    # Create the masks for the Earth and the Moon
    mask_earth = cv2.inRange(hsv, lower_earth, upper_earth)
    mask_moon = cv2.inRange(hsv, lower_moon, upper_moon)
    mask_star = cv2.inRange(hsv, lower_star, upper_star)

    # Apply the masks to the image
    result_earth = cv2.bitwise_and(stitched_img, stitched_img, mask=mask_earth)
    result_moon = cv2.bitwise_and(stitched_img, stitched_img, mask=mask_moon)
    result_moon = cv2.bitwise_and(result_moon, result_moon, mask=cv2.bitwise_not(mask_star))

    # Since the moon and stars are similar colours, we need a little more to get the moon isolated.
    gray = cv2.cvtColor(result_moon, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    moon_contour = max(contours, key=cv2.contourArea)
    result_moon = np.zeros_like(stitched_img)
    cv2.drawContours(result_moon, [moon_contour], -1, (255), thickness=cv2.FILLED)

    earth_midpoint = find_planet(result_earth, stitched_img)
    moon_midpoint = find_planet(result_moon, stitched_img)
    
    # Get diameters in pixels
    earth_diameter_pixels = find_diameter(result_earth)
    moon_diameter_pixels = find_diameter(result_moon)
    
    # Convert to NumPy float64 if not already using NumPy
    earth_midpoint = np.array(earth_midpoint, dtype=np.float64)
    moon_midpoint = np.array(moon_midpoint, dtype=np.float64)

    # Calculate radii in pixels
    earth_radius_pixels = earth_diameter_pixels / 2
    moon_radius_pixels = moon_diameter_pixels / 2
    
    # Calculate the distance between the midpoints of Earth and Moon in pixels
    pixel_distance_earth_to_moon = np.sqrt(np.sum((earth_midpoint - moon_midpoint) ** 2))

    # Subtract the sum of radii from the midpoint distance to get surface-to-surface distance
    surface_to_surface_pixel_distance = pixel_distance_earth_to_moon - (earth_radius_pixels + moon_radius_pixels)

    # Earth's known diameter in kilometers and the camera scaling factor
    earth_diameter_km = 12742  # in kilometers
    moon_diameter_km = 3474.8
    camera_scaling_factor = 3

    # Convert pixel distance to kilometers using conversion factor from Earth diameter
    conversion_factor = (earth_diameter_km / earth_diameter_pixels) * camera_scaling_factor
    surface_to_surface_distance_km = surface_to_surface_pixel_distance * conversion_factor

    # Calculate the conversion factor from pixels to kilometers
    conversion_factor = (earth_diameter_km / earth_diameter_pixels) * camera_scaling_factor

    fov_degrees = 60
    image_width_pixels = 960

    earth_angular_size = calculate_angular_size(earth_diameter_pixels, image_width_pixels, fov_degrees)
    moon_angular_size = calculate_angular_size(moon_diameter_pixels, image_width_pixels, fov_degrees)

    # Calculate distances to Earth and Moon from spacecraft
    distance_earth_spacecraft_km = calculate_distance_from_angular_size(earth_diameter_km, earth_angular_size)
    distance_moon_spacecraft_km = calculate_distance_from_angular_size(moon_diameter_km, moon_angular_size)

    # Write distances to a file
    with open('group2/measurements.txt', 'w') as file:
        file.write(f"Earth: {round(distance_earth_spacecraft_km)} km\n")
        file.write(f"Moon: {round(distance_moon_spacecraft_km)} km\n")
        file.write(f"Distance: {round(surface_to_surface_distance_km)} km\n")
            
    if( debug ):
        # The text
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255) # white color
        thickness = 1
        
        # Draw line between earth and moon
        cv2.line(stitched_img, tuple(map(int, earth_midpoint)), tuple(map(int, moon_midpoint)), (255,0,0), 5)
        cv2.putText(stitched_img, str(round(surface_to_surface_pixel_distance, 2)), 
                    (int((earth_midpoint[0] + moon_midpoint[0])/2), int((earth_midpoint[1] + moon_midpoint[1])/2)), 
                    font, 0.8, color, thickness)

        # save
        filepath = os.path.join("group2", "distances.png")
        cv2.imwrite(filepath, stitched_img)