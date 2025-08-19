# CRAFT-E: A Neuro-Symbolic Framework for Embodied Affordance Grounding

The `Cornell_Dataset_Annotation` folder contains paired RGB-D images (`rgbd/`) and corresponding JSON annotations (`annotation/`), where each annotation specifies `verb` (a verb representing the functional request), `object` (the single object that affords the action), and `shapes` (bounding boxes and labels for all visible objects). Additional real-world annotated data are in `RealWorld_Dataset_Annotation`, while `Evaluation Code` provides scripts for affordance detection and grasp evaluation.

The `RealWorld_Dataset_Annotation` folder contains annotations from three independent annotators (`annotator1.json`, `annotator2.json`, `annotator3.json`) over a real-world dataset of 20 verbs and 39 physical objects. For each verb, annotators selected the object that they judged to most affordably satisfy the verb query, providing diverse perspectives on functional grounding in real-world scenarios.

The `Samples` folder contains experimental videos of real-world grasping tasks performed using either **GraspNet** or **GraspKpNet**. Each video filename encodes both the verb query and the ground truth object that affords the verb, e.g., `Rotate_Screwdriver.mov` represents the verb *rotate* with the object *screwdriver*.

## Evaluation Code

The `Evaluation Code` folder contains our implementation for real-world grasping tasks. Due to the large gap between the environments required by **SAMv2** and **GraspNet**, the system is implemented in a **client–server** mode.  

### 📂 GraspNet Pipeline
- **camera.py** – wrapper for Intel RealSense camera functions.  
- **dobot_cr3.py** – wrapper for controlling the DoBot CR3 robot arm.  
- **graspnet_pipeline.py** – main pipeline code for the GraspNet backbone model.  
- **hand_eye_calibration.py** – hand–eye calibration utilities.  
- **util.py** – helper utility functions.  
- **main.py** – client script for detecting grasp points and sending requests.  
- **T_cam2gripper.npy** – calibration matrix for camera-to-gripper transformation.  
- **cornell_evaluation.py** – evaluation scripts on the Cornell dataset.  

### 📄 Other Files
- **sam2_clip_server.py** – server that receives grasp points and RGBs from the client, and returns grasp points for the object that best affords the queried verb. 
- **util.py** – helper utility functions.  
