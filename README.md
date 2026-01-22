# WAREHOUSE TREASURE HUNT & OBJECT RECOGNITION

## <span style="background-color: #FFFF00">INTRODUCTION</span>
This project provides a ROS 2 framework for the **NXP B3RB Warehouse Challenge**.
- The `b3rb_ros_warehouse.py` script serves as a foundational ROS 2 node.
    * Participants will extend this script to implement the full challenge logic.
- Participants will perform the following tasks to score points:
    * Autonomous navigation
    * Locating the shelves using map or camera
    * Utilizing heuristic for faster completion
    * Identifying objects using YOLO-derived data
    * Decoding QR codes via the front camera
    * Strategically revealing hidden shelves

### <span style="background-color: #CBC3E3">HARDWARE</span>
This software is designed to run on the B3RB and can be tested in compatible Gazebo simulations.
1.  [NXP MR-B3RB](https://nxp.gitbook.io/mr-b3rb): The target hardware rover.
    * Requires a forward-facing camera for QR code detection and potentially shelf/object recognition.
    * Relies on sensors (LIDAR, encoders, IMU) for localization & mapping (SLAM), and navigation (Nav2).
2.  [Gazebo Simulator](https://gazebosim.org/home): For development and testing in a simulated warehouse environment.
    * The simulation provides a B3RB model with sensors and necessary packages such as SLAM & NAV2.

### <span style="background-color: #CBC3E3">SOFTWARE</span>
This project is based on the autopilot project - [CogniPilot](https://cognipilot.org/) (AIRY Release for B3RB).
<br>
Refer the [CogniPilot AIRY Dev Guide](https://airy.cognipilot.org/) for information about it's various components.
<br>
- **ROS 2:** Targeted for Humble Hawksbill.
- **Navigation:** Relies on a fully functional Nav2 stack.
    * Configuration for SLAM and Nav2 can be adjusted in `cranium/src/b3rb/b3rb_nav2/config`.
- **Object Recognition:** An external YOLO model is provided by default to publish the detected objects.
    * The objects are publish on `/shelf_objects` (`synapse_msgs/WarehouseShelf`) topic.
- **Python Libraries:**
    * `rclpy`: ROS 2 client library for Python.
    * `numpy`: For numerical operations, particularly with map data.
    * `opencv`: For image processing, crucial for QR code decoding.
    * `scipy`: For image analysis and spatial distance calculations.
    * `tkinter`: For the optional progress table GUI.
- **[Cranium](https://airy.cognipilot.org/cranium/about/)**: A ROS workspace that performs higher level computation for CogniPilot.
    * On the hardware B3RB, it runs on [NavQPlus](https://nxp.gitbook.io/navqplus/) IMX8MPLUS (Mission Computer).
    * On the Gazebo Simulator, it runs on the Ubuntu Linux machine.
    * Relevant packages (detailed later):
        1. **b3rb_ros_aim_india**
        2. **synapse_msgs**
        3. **dream_world**
        4. **b3rb**
    * Interaction with the `cerebri` via `/cerebri/out/status` and `/cerebri/in/joy` topics.
- This project includes a ROS2 Python package that integrates into Cranium.

---
## <span style="background-color: #FFFF00">WAREHOUSE CHALLENGE DESCRIPTION</span>

The primary goal is to maximize points by correctly identifying objects and decoding QR codes.

### <span style="background-color: #CBC3E3; font-weight:bold">SIMULATION WORLD</span>
- **Warehouse:** Contains N distinct shelves with various obstacles that the participants must avoid.
    * Navigation is completely automated; alternatively participants may utilize manual mode at their discretion.
- **QR Codes:** A QR code is placed on both sides of the shelves. It's a 30 character string that contains the following:
    1. **Shelf ID:** A natural number from 1 to n.
    2. **Heuristic:** The **angle (0-360°)** from the x-axis to the line from current to next shelf.
        * The x-axis and y-axis are given by the red and green arrow in the foxglove map respectively.
        * The angle is embedded as a 5-character string starting from the 2nd index, as a float value.
        * First heuristic (angle from x-axis to the line from robot to first shelf) will be given beforehand.
        * NOTE: Angles are measured using the center of mass of shelves and the robot in the 2-D space.
    3. **Random string**: 20 character secret code unique for each shelf used for evaluation.
    * Format: For example QR string `2_116.6_HKq3wvCg8DGyflz3oNIj8d`, shelf ID = `2` and angle = `116.6`.
- **Shelves & Curtains:** Each shelf has two rows, with 3 objects per row (6 total), visible from both front and back.
    * **All shelves, except the first one in the sequence, are completely covered by a curtain.**
    * A shelf's curtain is unveiled only when the QR of **all previous shelves** are published to `/shelf_data`.
        * For example, the curtain of shelf 3 is unveiled when QR of shelf 1 and 2 are published correctly.
    * Hence, it's most efficient to utilize the heuristic and traverse the shelves in the sequence order.

### <span style="background-color: #CBC3E3; font-weight:bold">TASK WORKFLOW</span>
Implementing the logic for the following treasure hunt sequence:

1.  **Locate First Shelf:**
    * **Task:** Identify the center of mass and orientation of the first shelf.
        * By recognizing the footprint on the SLAM-generated map (`/map`, `/global_costmap`).
        * Alternatively, image processing on the feed from the buggy's front camera can be used.
        * The choice of method is at the participants' discretion.
    * ⚠️ NOTE: Exploration to unknown spaces may be required before a sufficient map is created.
2.  **Navigate to Target Shelf:**
    * **Task:** Utilize the Nav2 action client (detailed later) to move the B3RB to the shelf's vicinity.
3.  **Navigate to and Decode QR Code:**
    * **Task:** Maneuver the robot to the side of the shelf to view the QR code on the current shelf.
    * **Action:** Capture an image and implement QR decoding to extract its string.
4.  **Align for Object Recognition:**
    * **Task:** Position the buggy so its front camera captures a clear image of the objects in the shelf.
        * This facilitates accurate YOLO model recognition.
    * **Action:** Subscribe to `/shelf_objects` to receive the recognized objects from the YOLO model.
5.  **Publish Shelf Data:**
    * **Task:** Create and populate a `synapse_msgs/WarehouseShelf` message with:
        * Identified objects and count.
        * The decoded QR code string.
    * **Action:** Publish this message to `/shelf_data`.
6.  **Curtain Reveal Mechanism:**
    * **Rule:** Correctly publishing the QR for all preceding shelves in the sequence unveils the **next shelf**.
    * **Example:** Publishing the QR of shelf 1 and 2 will reveal the curtain of the 3rd shelf.
7.  **Find and Navigate to Next Shelf:**
    * **Task:** Use the decoded heuristic (angle) to determine the location of the next "treasure" spot (shelf).
        * Alternatively, participants may explore randomly to discover the remaining shelves.
    * **Action:** Explore and locate the newly revealed shelf and repeat the process from step 3.
### <span style="background-color: #CBC3E3; font-weight:bold">Sample world videos....For Qualifying Round</span>


https://github.com/user-attachments/assets/e34e23d5-e84a-4305-92b6-41ae19084e48



https://github.com/user-attachments/assets/794a09a8-5597-43aa-b73a-689f9e2ded23



https://github.com/user-attachments/assets/9c50e240-f37d-4eb9-b49f-84ff4bc4002a



https://github.com/user-attachments/assets/9ebd51ee-9a95-4fe7-8bed-b6a38defa95b


