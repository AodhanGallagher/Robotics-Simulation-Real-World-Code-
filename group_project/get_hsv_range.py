import cv2
import numpy as np
from typing import Tuple, List

# takes an HSV image and return the HSV range
# will use it to get accurate ranges for objects and planets 
def get_hsv_range(image: np.ndarray) -> Tuple[List[int], List[int]]:

    # Mask out black pixels
    mask = cv2.inRange(image, np.array([0, 0, 10]), np.array([180, 255, 255]))

    # Find the min and max HSV values
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    min_hsv = np.min(masked_image[mask == 255], axis=0)
    max_hsv = np.max(masked_image[mask == 255], axis=0)
    
    return min_hsv.tolist(), max_hsv.tolist()

# Load image
image = cv2.imread('')

# Convert it to HSV 
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Pass the HSV image to the function
lower_hsv, upper_hsv = get_hsv_range(hsv_image)

print(lower_hsv, upper_hsv)