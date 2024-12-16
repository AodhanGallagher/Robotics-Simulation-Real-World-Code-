import threading
from group_project import coordinates, planet_detection, stitch
import rclpy
import signal
import time
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
import os
from group_project import coordinates
from geometry_msgs.msg import Twist
from math import inf, pi, sin, cos
from nav_msgs.msg import Odometry
from math import atan2, degrees

# image processing
import cv2
import numpy as np

#  image processing
import cv2
import numpy as np
from sensor_msgs.msg import Image,LaserScan
from cv_bridge import CvBridge, CvBridgeError


class RoboNaut(Node):
    def __init__(self):
        super().__init__('robotnaut')
        self.initialize_parameters()
        self.initialize_image_processing()
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Movement
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.rate = self.create_rate(10)  # 10 Hz
        self.destination_arrive_flag = False
        
        # Odometry / Robot Pose Publisher
        self.odometry_publisher = self.create_publisher(Odometry, '/odometry', 10)
        
        # Subscriber for Odometry
        self.odometry_subscriber = self.create_subscription(Odometry, '/odom', self.odometry_callback, 10)
        
        # Colour Flags
        self.green_detected = False
        self.red_detected = False
        self.white_detected = False
        
        # Entrance sign Flags
        self.green_sign_detected = False
        self.red_sign_detected = False
        
        # Window sighted flag
        self.window_sighted = False
        self.lookingForWindow = False
        self.lookingForPlanets = False
        self.image = None
        self.lookingForSafeRoom = False
        
        self.robot_rotation = 0.0  # Initialize robot rotation variable
        self.window_rotation_values = [] #list to store rotation values when a window is detected
        
        # Lidar
        self.lidar_subscriber = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.desired_distance_to_window = 1.0  # Define the desired distance to the window
        self.moving_forward = False
        self.closest_wall_angle = 0.0

        # Stitching
        self.readyToStitch = False
                    
            
    def initialize_parameters(self):
        self.declare_parameter('coordinates_file_path', '')
        coordinates_file_path = self.get_parameter('coordinates_file_path').get_parameter_value().string_value
        self.coordinates = coordinates.get_module_coordinates(coordinates_file_path)

    def initialize_image_processing(self):
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, 'camera/image_raw', self.image_callback, 10)
        self.sensitivity = 15
                
        # Define the range of colors in HSV as a class attribute
        self.colors = {
            "red": [([0, 100, 100], [10, 255, 255]), ([170, 100, 100], [180, 255, 255])], # CHECK THIS, MIGHT BE ABLE TO ELIMINATE FIRE HYDRANT
            "green": [([60 - self.sensitivity, 100, 100], [60 + self.sensitivity, 255, 255])],
            "white": [([0, 0, 255 - self.sensitivity * 2], [255, self.sensitivity, 255])],
            "earth": [([0, 0, 10], [178, 237, 245])],
            "moon": [([0, 0, 10], [165, 30, 160])]
        }
        
        # Initialize window attributes
        self.window_count = 0
        self.detected_windows = [] # keep track of detected windows
        
        # Planet detection
        self.earth_detected = False
        self.moon_detected = False

    def detect_color_sign(self, hsv_image, color_name, ranges):
       mask = np.zeros_like(hsv_image[:, :, 0])
       for lower, upper in ranges:
           lower = np.array(lower, dtype="uint8")
           upper = np.array(upper, dtype="uint8")
           mask = cv2.bitwise_or(mask, cv2.inRange(hsv_image, lower, upper))
       contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

       if contours:
           largest_contour = max(contours, key=cv2.contourArea)
               
           area = cv2.contourArea(largest_contour)
           perimeter = cv2.arcLength(largest_contour, True) + 0.0001 # 0.00001 to remove 0 cases.
           ratio = 4 * np.pi * (area / (perimeter * perimeter))
            
           if cv2.contourArea(largest_contour) >= 2000:
               setattr(self, f'{color_name}_detected', True)
               if color_name == "green" or color_name == "red":
                   if ratio < 0.65:
                       return False
                   
               return True
       return False
    
    def detect_green_sign(self, image):
        # Convert image to HSV color space
        try:
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        except CvBridgeError as e:
            self.get_logger().error(f'Error converting image: {e}')
            return
    
        # Reset color detection flags
        self.green_detected = False
        self.white_detected = False
        
        # Detect green and white
        self.detect_color_sign(hsv_image, "green", self.colors["green"])
        self.detect_color_sign(hsv_image, "white", self.colors["white"])
        
        # Check if both green and white colors are detected
        if self.green_detected and self.white_detected:
            self.get_logger().info('Green sign detected!')
            self.green_sign_detected = True
            return True
        return False
    
    def detect_red_sign(self, image):
        # Convert image to HSV color space
        try:
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        except CvBridgeError as e:
            self.get_logger().error(f'Error converting image: {e}')
            return
        
        # Reset color detection flags
        self.red_detected = False
        self.white_detected = False
        
        # Detect red and white
        self.detect_color_sign(hsv_image, "red", self.colors["red"])
        self.detect_color_sign(hsv_image, "white", self.colors["white"])
        
        # Check if both red and white colors are detected
        if self.red_detected and self.white_detected:
            self.get_logger().info('Red sign detected!')
            self.red_sign_detected = True
            return True
        return False
    
    def detect_color(self, hsv_image, color_name, ranges):
        # --from planet_detection file--
        return planet_detection.detect_color(hsv_image, color_name, ranges)

    def is_new_window(self, current_rotation, current_time):
        threshold_angle = 10  # Degrees
        threshold_time = 1.5  # Seconds
        
        # Debug: print 
        for window in self.detected_windows:
            # Convert rotation for comparison and calculate differences
            current_rotation_degrees = degrees(current_rotation)
            rotation_difference = abs(current_rotation_degrees - window['rotation'])
            time_difference = current_time - window['time']
            
            if rotation_difference < threshold_angle or time_difference < threshold_time:
                return False  # It's likely the same window
        
        return True 

    def find_windows(self, image):
        self.window_sighted = False
        # If both planets detected, skip window
        if self.earth_detected and self.moon_detected:
            self.readyToStitch = True
            return
        
        # Convert to greyscale to simplify processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
        # Use binary thresholding so that white represents the potential window borders
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Set save dir
        save_directory = "group2"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
                
        for c in contours:
            # Approximate the contour, using approxPolyDP
            approx = cv2.approxPolyDP(c, 0.05 * cv2.arcLength(c, True), True)
            
            # Calculate the bounding rectangle of this contour
            x, y, w, h = cv2.boundingRect(approx)
            
            # Define minimum size requirements for a window
            min_width, min_height = 50, 50
            
            # Check if the approximated contour has 4 vertices
            # Check its not previously detected
            # Check minimum size requirements
            if len(approx) == 4 and w >= min_width and h >= min_height:
                # Save the rotation value when a centered window is detected
                rotation_value = self.get_latest_rotation()
                current_time = time.time()  # Get current time in seconds                        

                if self.is_new_window(degrees(rotation_value), current_time):
                    # Check if the window is centered
                    if self.check_window_centering(x, w, image.shape[1]):
                        self.window_sighted = True
                        rotation_value = self.get_latest_rotation()
                        self.detected_windows.append({'rotation': rotation_value, 'time': current_time})
                        self.window_rotation_values.append(rotation_value)
                        self.get_logger().info("New window seen...")
                        break

    def detect_windows(self, image):
        self.window_sighted = False
        
        if self.earth_detected and self.moon_detected:
            self.readyToStitch = True
            return
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            approx = cv2.approxPolyDP(c, 0.05 * cv2.arcLength(c, True), True)
            x, y, w, h = cv2.boundingRect(approx)
            
            if len(approx) == 4 and w >= 50 and h >= 50:
                cropped_image = image[y:y+h, x:x+w]

                if planet_detection.is_good_image(cropped_image):                     
                    # Scan for planets
                    planet_name = planet_detection.scan_for_planets(self, cropped_image, image)
                    if planet_name is not None:
                        if planet_name == 'Earth' and not self.earth_detected:
                            self.earth_detected = True
                        elif planet_name == 'Moon' and not self.moon_detected:
                            self.moon_detected = True

                        # Stop looking for planet since one was found
                        self.lookingForPlanets = False
                        self.get_logger().info(f"planet:{planet_name}")
                    else:
                        self.get_logger().info("no planets here.")
                else:
                    self.get_logger().info("bad image.")
            else:
                self.get_logger().info("small window.")
                
    def check_window_centering(self, window_x, window_width, image_width, threshold_distance=20):
        # Calculate the center of the window
        window_center_x = window_x + window_width // 2
        
        # Calculate the center of the image
        image_center_x = image_width // 2
        
        # Calculate the distance between the window center and the image center
        distance_from_center = abs(window_center_x - image_center_x)
        
        # Check if the window is centered based on the given threshold
        return distance_from_center < threshold_distance
    
    def image_callback(self, data):        
        # Convert the received image into an OpenCV image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            self.image = cv_image
        except CvBridgeError as e:
            self.get_logger().error(f'Error converting image: {e}')
            return
        
        # Check for signs
        if ( self.lookingForSafeRoom ):
            self.detect_green_sign(cv_image)
            self.detect_red_sign(cv_image)
        
        # Window detection
        if ( self.lookingForWindow ):
            self.find_windows(cv_image)
            
        if ( self.lookingForPlanets ):
            self.get_logger().info(f"Searching for planet...")
            self.detect_windows(cv_image)
            
        # Show camera feed - no mask
        cv2.imshow('Camera Feed', cv_image)
        cv2.resizeWindow('Camera Feed', 420, 340)
        cv2.waitKey(1)
        return
    
    def is_too_close_to_wall(self, lidar_data):
        # Extract the range data from the lidar message
        ranges = lidar_data.ranges[0]

        # Define the threshold distance for being too close to the wall
        threshold_distance = 0.5 # Adjust this threshold as needed
        
        # Check if the closest distance is less than the threshold
        if ranges < threshold_distance:
            return True  # Robot is too close to the wall
        return False  # Robot is not too close to the wall
    
    def is_crashed_into_wall(self, lidar_data):
        # Extract the range data from the lidar message
        ranges = lidar_data.ranges[0]

        # Define the threshold distance for being too close to the wall
        threshold_distance = 0.3 # Adjust this threshold as needed
        
        # Check if the closest distance is less than the threshold
        if ranges < threshold_distance:
            return True  # Robot is too close to the wall
        return False  # Robot is not too close to the wall
    
    def is_too_far_to_wall(self, lidar_data):
        # Extract the range data from the lidar message
        ranges = lidar_data.ranges[0]

        # Define the threshold distance for being too close to the wall
        threshold_distance = 1.8 # Adjust this threshold as needed
        
        # Check if the closest distance is less than the threshold
        if ranges > threshold_distance:
            return True  # Robot is too far to the wall
        return False  # Robot is not too far to the wall

    
    def move_to_wall(self):
        # Start moving forward
        self.walk_forward()

        try:
            while True:
                # Check if the robot is too close to the wall
                if self.is_too_close_to_wall(self.latest_lidar_data):
                    # If too close, stop moving forward
                    self.stop()
                    break
        except KeyboardInterrupt:
            # Stop the robot if the program is interrupted
            self.stop()
            
    def crash_in_wall(self):
        # Start moving forward
        self.walk_forward_slow()

        try:
            while True:
                # Check if the robot is too close to the wall
                if self.is_crashed_into_wall(self.latest_lidar_data):
                    # If too close, stop moving forward
                    time.sleep(0.5)
                    self.stop()
                    break
        except KeyboardInterrupt:
            # Stop the robot if the program is interrupted
            self.stop()
            
    def move_from_wall(self):
        # Start moving back
        self.walk_backward()

        try:
            while True:
                # Check if the robot is too far to the wall
                if self.is_too_far_to_wall(self.latest_lidar_data):
                    # If too far, stop moving back
                    self.stop()
                    break
        except KeyboardInterrupt:
            # Stop the robot if the program is interrupted
            self.stop()

    def take_picture(self):
        image = self.image
        save_directory = "group2"
        filepath = os.path.join(save_directory, f'window{self.window_count}.png')
        cv2.imwrite(filepath, image)
        self.window_count += 1
        self.get_logger().info("Window saved.")
    
    def rotate_to_perpendicular(self, threshold=0.0):
        if self.closest_wall_angle is not None:
            # Rotate until range[0] is shortest
            shortest_distance = min(self.latest_lidar_data.ranges) + threshold
            angle_increment = self.latest_lidar_data.angle_increment
            num_ranges = len(self.latest_lidar_data.ranges)
            
            # Find the index of the range directly ahead
            forward_index = 0
            
            # Initialize variables to keep track of rotation direction
            clockwise_rotation = False
            counterclockwise_rotation = False

            # Check if the range directly ahead is the shortest
            if self.latest_lidar_data.ranges[forward_index] == shortest_distance:
                # Robot is already facing the wall directly ahead
                return
            elif self.latest_lidar_data.ranges[forward_index + 10] < self.latest_lidar_data.ranges[forward_index + 350]:
                # Rotate counterclockwise
                counterclockwise_rotation = True
            else:
                # Rotate clockwise
                clockwise_rotation = True
            
            # Rotate the robot
            desired_velocity = Twist()
            angular_velocity = 0.25  # Adjust the angular velocity as needed
            
            while True:
                # Check if the range directly ahead becomes the shortest
                if self.latest_lidar_data.ranges[forward_index] <= shortest_distance:
                    break  # Exit the loop if directly ahead is shortest
                
                if counterclockwise_rotation:
                    desired_velocity.angular.z = angular_velocity  # Rotate counterclockwise
                elif clockwise_rotation:
                    desired_velocity.angular.z = -angular_velocity  # Rotate clockwise
                
                # Publish the desired velocity
                self.publisher.publish(desired_velocity)
                self.rate.sleep()
                
            # Stop the robot once directly ahead is shortest
            self.stop()
            self.get_logger().info("I am perpendicular !.")
        else:
            # Log or handle the case where no closest wall angle has been found
            self.get_logger().info("No closest wall detected. Cannot rotate.")

            
    def lidar_callback(self, lidar_data):
        self.latest_lidar_data = lidar_data

            
    # -------------------------------------------------------------
    # ROBOT ODOMETRY
    
    def odometry_callback(self, msg):
        # Extract position and orientation from the received odometry message
        position_x = msg.pose.pose.position.x
        position_y = msg.pose.pose.position.y
        orientation_z = msg.pose.pose.orientation.z
        orientation_w = msg.pose.pose.orientation.w
        
        self.x, self.y = position_x, position_y
        # Calculate the rotation angle (yaw) using quaternion to Euler angle conversion
        # This calculation depends on your specific convention and frame of reference
        # Here, we assume a simple 2D orientation where yaw is obtained from the quaternion
        robot_rotation = 2 * atan2(orientation_z, orientation_w)  # Update robot rotation

        # Update the class variable
        self.robot_rotation = robot_rotation

        # Do something with the position and orientation information
        # For example, print it
        #self.get_logger().info(f'Robot Rotation (Degrees): {robot_rotation_degrees}')

    def publish_odometry(self, position, orientation):
        odometry_msg = Odometry()
        odometry_msg.pose.pose.position.x = position[0]
        odometry_msg.pose.pose.position.y = position[1]
        odometry_msg.pose.pose.orientation.z = orientation[0]
        odometry_msg.pose.pose.orientation.w = orientation[1]

        # Publish the odometry message
        self.odometry_publisher.publish(odometry_msg)
    
    def get_latest_rotation(self):
        return self.robot_rotation


    # -------------------------------------------------------------
    # ROBOT MOVEMENT FUNCTIONS
    def walk_forward(self):
        # Move the robot forward
        desired_velocity = Twist()
        desired_velocity.linear.x = 0.2  # Adjust the forward speed as needed
        self.publisher.publish(desired_velocity)
        
    def walk_forward_slow(self):
        # Move the robot forward
        desired_velocity = Twist()
        desired_velocity.linear.x = 0.05  # Adjust the forward speed as needed
        self.publisher.publish(desired_velocity)

    
    def walk_backward(self):
        # Move the robot backwards
        desired_velocity = Twist()
        desired_velocity.linear.x = -0.2  # Adjust the backwards speed as needed
        self.publisher.publish(desired_velocity)


    def turn360(self):
        desired_velocity = Twist()
        # Turning the robot by 360 degrees in 15 seconds anti-clockwise
        desired_velocity.angular.z = -3.1416 / 6
        
        for _ in range(150):
            self.publisher.publish(desired_velocity)
            self.rate.sleep()
        self.stop()

    def stop(self):
        # Stop the robot
        desired_velocity = Twist()
        self.publisher.publish(desired_velocity)

    def point_in_rotation(self, rotation_value):
        # Normalize rotation values to be in the range from -pi to pi
        rotation_value = rotation_value % (2 * pi)
        if rotation_value > pi:
            rotation_value -= 2 * pi

        current_rotation = self.robot_rotation
        
        # Calculate the difference between the current rotation and the desired rotation
        rotation_difference = rotation_value - current_rotation
        if rotation_difference > pi:
            rotation_difference -= 2 * pi
        elif rotation_difference < -pi:
            rotation_difference += 2 * pi
        
        # Adjust the robot's orientation by turning to face the desired rotation
        desired_velocity = Twist()
        if rotation_difference > 0:
            desired_velocity.angular.z = 0.5  # Turn counter-clockwise
        else:
            desired_velocity.angular.z = -0.5  # Turn clockwise

        # Log the initial rotation difference in degrees
        self.get_logger().info(f"Rotating to target: {degrees(rotation_difference):.2f} degrees")

        # Turn until the desired rotation is reached
        while abs(rotation_difference) > 0.1:  # Assuming a tolerance of 0.1 radians
            self.publisher.publish(desired_velocity)
            self.rate.sleep()
            
            # Update the current rotation and ensure it is normalized
            current_rotation = self.robot_rotation % (2 * pi)
            if current_rotation > pi:
                current_rotation -= 2 * pi

            # Recalculate the rotation difference
            rotation_difference = rotation_value - current_rotation
            if rotation_difference > pi:
                rotation_difference -= 2 * pi
            elif rotation_difference < -pi:
                rotation_difference += 2 * pi
        
        # Stop the robot once the desired rotation is reached
        self.stop()

        
    # --------------------------------------------------------------
    # SEND CO-ORDS TO ROBOT
    def send_goal(self, x, y, yaw):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Position
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y

        # Orientation
        goal_msg.pose.pose.orientation.z = sin(yaw / 2)
        goal_msg.pose.pose.orientation.w = cos(yaw / 2)

        self.action_client.wait_for_server()
        self.send_goal_future = self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self.send_goal_future.add_done_callback(self.goal_response_callback)


    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)


    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Navigation result: {result}')
        self.destination_arrive_flag = True


    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback


    # --------------------------------------------------------------
    # STITCH
    def stitch_images(robonaut, img1="group2/viewEarth.png", img2="group2/viewMoon.png"):
        robonaut.get_logger().info("Ready for stitching...")
        # Start the stitch
        try:
            # Check if the image files exist
            if not os.path.exists(img1):
                robonaut.get_logger().error(f"Error: File not found - {img1}")
                return
            if not os.path.exists(img2):
                robonaut.get_logger().error(f"Error: File not found - {img2}")
                return
            
            robonaut.get_logger().info("Files found...")
            
            # Stitch here
            robonaut.get_logger().info("Starting stitching...")
            stitch.main_stitch(img1, img2)
        except Exception as e:
            robonaut.get_logger().error(f"An error occurred during stitching: {str(e)}")
    
    
    # ------------------------------------------------------------------------------------
    # Inch forward function
    def inch_forward(self, safe_room):
        self.window_rotation_values.clear()
            
        self.send_goal(safe_room.x, safe_room.y, 4.78)
        while not self.destination_arrive_flag:
            time.sleep(0.1)
        self.destination_arrive_flag = False
        
        self.lookingForWindow = True
        self.turn360()
        self.lookingForWindow = False
            
        # Iterate through the window rotation values and print them
        for rotation_value in self.window_rotation_values:
                        
            # Point in the window rotation
            self.get_logger().info(f"Pointing to window...")
            self.point_in_rotation(rotation_value)
            
            time.sleep(1)
            
            # Move to window
            self.get_logger().info(f"Moving to window...")
            self.move_to_wall()
            
            # Rotate to be perpendicular to the wall
            self.get_logger().info(f"Getting perpendicular...")
            self.rotate_to_perpendicular()
            
            # Move backwards to adjust view
            self.get_logger().info(f"Moving back...")
            self.move_from_wall()
            
            # Look for planets
            self.lookingForPlanets = True
            time.sleep(5)
            self.lookingForPlanets = False
            
            # Else, continue looking for windows
            # Move back to center
            self.get_logger().info(f"Moving back to center...")
            self.send_goal(safe_room.x, safe_room.y, 4.78)
            
            while not self.destination_arrive_flag:
                time.sleep(0.1)
            self.destination_arrive_flag = False
            
            # Wait for some time to let the robot stabilize
            time.sleep(1)
            
            # Check if both planets are detected and then stitch
            if self.earth_detected and self.moon_detected:
                self.stitch_images()
                self.get_logger().info(f"Stitching done...")
                self.readyToStitch = True
                break
        
        
        
def main():
    def signal_handler(sig, frame):
        robonaut.stop()
        rclpy.shutdown()

    rclpy.init(args=None)
    
    robonaut = RoboNaut()
    
    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(robonaut,), daemon=True)
    thread.start()
    
    try:
        # Create directory
        save_directory = "group2"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            
            
        # =========== Corridor NAV ==================
        safe_room = None
        
        # Move to the first entrance.
        robonaut.send_goal(robonaut.coordinates.module_1.entrance.x, robonaut.coordinates.module_1.entrance.y, 4.78)
        while not robonaut.destination_arrive_flag:
            time.sleep(0.1)
        robonaut.destination_arrive_flag = False
        
        # We're at the correct coordinates, 360 until we see the dot.
        robonaut.lookingForSafeRoom = True 
        while not (robonaut.red_sign_detected or robonaut.green_sign_detected):
            robonaut.turn360()

        robonaut.lookingForSafeRoom = False 
        
        if robonaut.green_sign_detected:
            safe_room = robonaut.coordinates.module_1.center 
        else:
            safe_room = robonaut.coordinates.module_2.center 
            
        robonaut.get_logger().info(f"Found Safe Room: {safe_room}")
        
        
        # =========== ROOM NAV ==================
        robonaut.send_goal(safe_room.x, safe_room.y, 4.78)
        while not robonaut.destination_arrive_flag:
            time.sleep(0.1)
        robonaut.destination_arrive_flag = False
        
        robonaut.lookingForWindow = True
        robonaut.turn360()
        robonaut.lookingForWindow = False
            
        # Iterate through the window rotation values and print them
        for rotation_value in robonaut.window_rotation_values:
                        
            # Point in the window rotation
            robonaut.get_logger().info(f"Pointing to window...")
            robonaut.point_in_rotation(rotation_value)
            
            time.sleep(1)
            
            # Move to window
            robonaut.get_logger().info(f"Moving to window...")
            robonaut.move_to_wall()
            
            # Rotate to be perpendicular to the wall
            robonaut.get_logger().info(f"Getting perpendicular...")
            robonaut.rotate_to_perpendicular()
            
            # Move backwards to adjust view
            robonaut.get_logger().info(f"Moving back...")
            robonaut.move_from_wall()
            
            # Take picture of the window
            robonaut.get_logger().info(f"Taking picture...")
            robonaut.take_picture()

            # Look for planets     
            robonaut.lookingForPlanets = True       
            time.sleep(5)
            robonaut.lookingForPlanets = False
            
            # Check if both planets are detected and then stitch
            if robonaut.earth_detected and robonaut.moon_detected:
                robonaut.stitch_images()
                robonaut.get_logger().info(f"Stitching done...")
                robonaut.readyToStitch = True
                break
        
            # Else, continue looking for windows
            # Move back to center
            robonaut.get_logger().info(f"Moving back to center...")
            robonaut.send_goal(safe_room.x, safe_room.y, 4.78)
            
            while not robonaut.destination_arrive_flag:
                time.sleep(0.1)
            robonaut.destination_arrive_flag = False
            
            # Wait for some time to let the robot stabilize
            time.sleep(1)
            
            
        # INCH FORWARD 1
        if (robonaut.readyToStitch == False):
            safe_room.y = safe_room.y + 3.5
            robonaut.inch_forward(safe_room)
            
        # INCH FORWARD 2
        if (robonaut.readyToStitch == False):
            safe_room.y = safe_room.y - 5.0
            robonaut.inch_forward(safe_room)

        # end
        robonaut.stop()
        robonaut.get_logger().info(f"-END-")

    except Exception as e:
        robonaut.get_logger().error(f"An error occurred during main run: {str(e)}")
        
    finally:
        # Ensure proper shutdown
        robonaut.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()