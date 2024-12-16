import cv2
import numpy as np
from cv_bridge import CvBridgeError
import os

def create_mask(hsv_image, color_ranges):
    mask = np.zeros(hsv_image.shape[:2], dtype="uint8")  # Initialize mask
    for lower, upper in color_ranges:
        lower_np = np.array(lower, dtype="uint8")
        upper_np = np.array(upper, dtype="uint8")
        color_mask = cv2.inRange(hsv_image, lower_np, upper_np)
        mask = cv2.bitwise_or(mask, color_mask)
    return mask

def detect_color(hsv_image, color_name, ranges):
    mask = np.zeros_like(hsv_image[:, :, 0])
    for lower, upper in ranges:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_image, lower, upper))
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) >= 500:
            return True
    return False

def crop_image_border(image, border_size=40):
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Define the ROI coordinates
    x1 = border_size
    y1 = border_size
    x2 = width - border_size
    y2 = height - border_size
    
    # Crop the image using the ROI coordinates
    cropped_image = image[y1:y2, x1:x2]

    return cropped_image


def scan_for_planets(self, cropped_image, window):
    # Crop
    contours_cropped_image = crop_image_border(cropped_image)
    
    # Convert image to HSV color space
    try:
        hsv_image = cv2.cvtColor(contours_cropped_image, cv2.COLOR_BGR2HSV)
    except CvBridgeError as e:
        self.get_logger().error(f'Error converting image: {e}')
        return
    
    # Create masks for 'earth' and 'moon'
    earth_mask = create_mask(hsv_image, self.colors['earth'])
    moon_mask = create_mask(hsv_image, self.colors['moon'])

    # Combine earth and moon masks into a single mask
    combined_mask = cv2.bitwise_or(earth_mask, moon_mask)
    masked_image = cv2.bitwise_and(hsv_image, hsv_image, mask=combined_mask)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        
    # Find contours
    contours, _ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Calculate area and circularity
        area = cv2.contourArea(contour)
        
        # Check if countour is large enough to be a planet
        if area > 350:
            # Circle to the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Ensure the indices are within bounds to avoid cropping errors
            x1, y1 = max(center[0] - radius, 0), max(center[1] - radius, 0)
            x2, y2 = min(center[0] + radius, contours_cropped_image.shape[1]-1), min(center[1] + radius, cropped_image.shape[0]-1)
            
            if x1 < x2 and y1 < y2:
                # Crop the circular area from the original image
                circular_region = contours_cropped_image[y1:y2, x1:x2]

            # Determine if it's the Earth or the Moon
            planet_name = identify_planet(self, circular_region)

            # Chcek that the planet is already detected
            if planet_name == 'Earth':
                if not self.earth_detected:
                    self.earth_detected = True
                    
                    # Save the planet image
                    save_path = os.path.join("group2", f'view{planet_name}.png')
                    cv2.imwrite(save_path, cropped_image)
                    
                    # Save the window and increment count
                    filepath = os.path.join("group2", f'window{self.window_count}.png')
                    cv2.imwrite(filepath, window)
                    self.window_count += 1
                return planet_name
            
            elif planet_name == 'Moon':
                if not self.moon_detected:
                    self.moon_detected = True
                    
                    # Save the planet image
                    save_path = os.path.join("group2", f'view{planet_name}.png')
                    cv2.imwrite(save_path, cropped_image)
                    
                    # Save the window and increment count
                    filepath = os.path.join("group2", f'window{self.window_count}.png')
                    cv2.imwrite(filepath, window)
                    self.window_count += 1
                return planet_name
    return None

def identify_planet(self, region):   
    # Convert image to HSV color space
    try:
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    except CvBridgeError as e:
        self.get_logger().error(f'Error converting image: {e}')
        return
    
        # Check for Moon by detecting gray tones
    if self.detect_color(hsv_region, "moon", self.colors["moon"]):
        return 'Moon'
    
    # Check for Earth by detecting blue tones
    if self.detect_color(hsv_region, "earth", self.colors["earth"]):
        return 'Earth'
    
    return 'None'

def is_good_image(image):    
    # Define the border width
    border_width = 10
    threshold = 250

    # Extract the border areas
    top_border = image[:border_width, :]
    bottom_border = image[-border_width:, :]
    left_border = image[:, :border_width]
    right_border = image[:, -border_width:]

    # Compute the average grey level in the border areas
    average_grey_top = np.mean(top_border)
    average_grey_bottom = np.mean(bottom_border)
    average_grey_left = np.mean(left_border)
    average_grey_right = np.mean(right_border)

    # Compute overall average grey level for all borders
    overall_average_grey = (average_grey_top + average_grey_bottom + average_grey_left + average_grey_right) / 4

    # Return True if the overall average grey level is less than the threshold
    return overall_average_grey < threshold