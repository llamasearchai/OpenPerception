import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, Imu, NavSatFix
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import numpy as np
import logging
from typing import Callable, Dict, Any, Optional
# Adjusted import path for sensor_fusion.fusion
from openperception.sensor_fusion.fusion import CameraReading, LidarReading, ImuReading, GpsReading
import threading
import time

logger = logging.getLogger(__name__)

class ROS2Bridge(Node):
    """Bridge between ROS2 and OpenPerception"""
    
    def __init__(self, node_name="open_perception_node"):
        """Initialize ROS2 bridge
        
        Args:
            node_name: Name of the ROS2 node
        """
        # Initialize ROS2
        if not rclpy.ok(): # Check if rclpy has been initialized
            rclpy.init()
        super().__init__(node_name)
        
        # CV bridge for image conversion
        self.cv_bridge = CvBridge()
        
        # Subscribers
        self.camera_subscribers = {}
        self.lidar_subscribers = {}
        self.imu_subscribers = {}
        self.gps_subscribers = {}
        
        # Publishers
        self.pose_publisher = self.create_publisher(Odometry, 'open_perception/pose', 10)
        
        # Callbacks
        self.camera_callbacks = {}
        self.lidar_callbacks = {}
        self.imu_callbacks = {}
        self.gps_callbacks = {}
        
        # Extrinsics & Intrinsics storage
        self.camera_intrinsics = {} # Store camera_matrix and dist_coeffs
        self.camera_extrinsics = {}
        self.lidar_extrinsics = {}
        self.imu_extrinsics = {}
        self.gps_extrinsics = {}
        
        # Spin thread
        self.spin_thread = threading.Thread(target=self._spin, daemon=True)
        self.running = False
        
    def start(self):
        """Start ROS2 bridge"""
        if not self.running:
            self.running = True
            if not self.spin_thread.is_alive():
                 self.spin_thread.start()
            logger.info(f"ROS2 bridge started for node {self.get_name()}")
        
    def stop(self):
        """Stop ROS2 bridge"""
        if self.running:
            self.running = False
            if self.spin_thread.is_alive():
                self.spin_thread.join(timeout=1.0) # Add timeout to join
            # self.destroy_node() # This should be called by the user of the bridge if they manage its lifecycle
            # rclpy.shutdown() # Avoid shutting down globally if other nodes are running
            logger.info(f"ROS2 bridge stopped for node {self.get_name()}")
        
    def _spin(self):
        """Thread function for ROS2 spinning"""
        while self.running and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            # time.sleep(0.001) # spin_once with timeout already blocks
            
    def add_camera_subscriber(self, topic: str, sensor_id: str, 
                              callback: Optional[Callable[[CameraReading], None]] = None,
                              extrinsics: Optional[np.ndarray] = None):
        """Add a subscriber for camera images and info
        
        Args:
            topic: ROS2 topic for camera images (e.g., /camera/image_raw)
            sensor_id: Unique ID for this camera
            callback: Optional callback function for camera readings
            extrinsics: Optional 4x4 extrinsic matrix
        """
        self.camera_callbacks[sensor_id] = callback
        if extrinsics is not None:
            self.camera_extrinsics[sensor_id] = extrinsics
        self.camera_intrinsics[sensor_id] = {'camera_matrix': np.eye(3), 'dist_coeffs': np.zeros(5)} # Default
            
        image_topic = topic
        info_topic = topic.rsplit('/', 1)[0] + '/camera_info' if '/' in topic else 'camera_info'
        if topic.endswith('image_rect_color') or topic.endswith('image_rect'): # Common for rectified images
             info_topic = topic.replace('image_rect_color', 'camera_info').replace('image_rect', 'camera_info')
        elif topic.endswith('image_raw'):
             info_topic = topic.replace('image_raw', 'camera_info')
        else: # Fallback if not standard naming
            logger.warning(f"Cannot infer camera_info topic from {topic}, please ensure it's published correctly or manually subscribe.")

        image_sub = self.create_subscription(
            Image,
            image_topic,
            lambda msg: self._camera_callback(msg, sensor_id),
            10
        )
        
        info_sub = self.create_subscription(
            CameraInfo,
            info_topic,
            lambda msg: self._camera_info_callback(msg, sensor_id),
            10
        )
        
        self.camera_subscribers[sensor_id] = {
            'image': image_sub,
            'info': info_sub
        }
        
        logger.info(f"Added camera subscriber for image topic {image_topic} and info topic {info_topic} with ID {sensor_id}")
        
    def _camera_callback(self, msg: Image, sensor_id: str):
        """Callback for camera images"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') # Use bgr8 generally
            
            intrinsics = self.camera_intrinsics.get(sensor_id, {'camera_matrix': np.eye(3), 'dist_coeffs': np.zeros(5)})
            
            reading = CameraReading(
                timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                sensor_id=sensor_id,
                image=cv_image,
                camera_matrix=intrinsics['camera_matrix'],
                dist_coeffs=intrinsics['dist_coeffs']
            )
            
            if sensor_id in self.camera_callbacks and self.camera_callbacks[sensor_id] is not None:
                self.camera_callbacks[sensor_id](reading)
                
        except Exception as e:
            logger.error(f"Error in camera callback for {sensor_id} on topic {msg.header.frame_id}: {e}", exc_info=True)
            
    def _camera_info_callback(self, msg: CameraInfo, sensor_id: str):
        """Callback for camera info"""
        try:
            self.camera_intrinsics[sensor_id] = {
                'camera_matrix': np.array(msg.k).reshape((3, 3)),
                'dist_coeffs': np.array(msg.d)
            }
            # logger.debug(f"Received camera info for {sensor_id}: K={self.camera_intrinsics[sensor_id]['camera_matrix']}")
        except Exception as e:
            logger.error(f"Error in camera_info_callback for {sensor_id}: {e}", exc_info=True)
        
    def add_lidar_subscriber(self, topic: str, sensor_id: str,
                             callback: Optional[Callable[[LidarReading], None]] = None,
                             extrinsics: Optional[np.ndarray] = None):
        """Add a subscriber for LiDAR point clouds"""
        self.lidar_callbacks[sensor_id] = callback
        if extrinsics is not None:
            self.lidar_extrinsics[sensor_id] = extrinsics
            
        sub = self.create_subscription(
            PointCloud2,
            topic,
            lambda msg: self._lidar_callback(msg, sensor_id),
            10
        )
        self.lidar_subscribers[sensor_id] = sub
        logger.info(f"Added LiDAR subscriber for topic {topic} with ID {sensor_id}")
        
    def _lidar_callback(self, msg: PointCloud2, sensor_id: str):
        """Callback for LiDAR point clouds.
           Simplified: assumes x, y, z fields. Add intensity if available.
        """
        try:
            # Basic parsing of PointCloud2. For robust parsing, consider ros_numpy or point_cloud2.read_points
            from sensor_msgs_py import point_cloud2
            points_list = []
            intensities_list = []
            has_intensity = any(field.name == 'intensity' for field in msg.fields)

            for point_data in point_cloud2.read_points(msg, field_names=("x", "y", "z", "intensity") if has_intensity else ("x", "y", "z"), skip_nans=True):
                points_list.append([point_data[0], point_data[1], point_data[2]])
                if has_intensity:
                    intensities_list.append(point_data[3])
            
            points_np = np.array(points_list, dtype=np.float32)
            intensities_np = np.array(intensities_list, dtype=np.float32) if has_intensity and intensities_list else None
            
            reading = LidarReading(
                timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                sensor_id=sensor_id,
                points=points_np,
                intensities=intensities_np
            )
            
            if sensor_id in self.lidar_callbacks and self.lidar_callbacks[sensor_id] is not None:
                self.lidar_callbacks[sensor_id](reading)
                
        except Exception as e:
            logger.error(f"Error in LiDAR callback for {sensor_id}: {e}", exc_info=True)
            
    def add_imu_subscriber(self, topic: str, sensor_id: str,
                             callback: Optional[Callable[[ImuReading], None]] = None,
                             extrinsics: Optional[np.ndarray] = None):
        """Add a subscriber for IMU data"""
        self.imu_callbacks[sensor_id] = callback
        if extrinsics is not None:
            self.imu_extrinsics[sensor_id] = extrinsics
            
        sub = self.create_subscription(
            Imu, topic, lambda msg: self._imu_callback(msg, sensor_id), 10)
        self.imu_subscribers[sensor_id] = sub
        logger.info(f"Added IMU subscriber for topic {topic} with ID {sensor_id}")
        
    def _imu_callback(self, msg: Imu, sensor_id: str):
        """Callback for IMU data"""
        try:
            accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
            ang_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
            orientation = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
            
            reading = ImuReading(
                timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                sensor_id=sensor_id,
                acceleration=accel,
                angular_velocity=ang_vel,
                orientation=orientation
            )
            
            if sensor_id in self.imu_callbacks and self.imu_callbacks[sensor_id] is not None:
                self.imu_callbacks[sensor_id](reading)
                
        except Exception as e:
            logger.error(f"Error in IMU callback for {sensor_id}: {e}", exc_info=True)
            
    def add_gps_subscriber(self, topic: str, sensor_id: str,
                             callback: Optional[Callable[[GpsReading], None]] = None,
                             extrinsics: Optional[np.ndarray] = None):
        """Add a subscriber for GPS data"""
        self.gps_callbacks[sensor_id] = callback
        if extrinsics is not None:
            self.gps_extrinsics[sensor_id] = extrinsics
            
        sub = self.create_subscription(
            NavSatFix, topic, lambda msg: self._gps_callback(msg, sensor_id), 10)
        self.gps_subscribers[sensor_id] = sub
        logger.info(f"Added GPS subscriber for topic {topic} with ID {sensor_id}")
        
    def _gps_callback(self, msg: NavSatFix, sensor_id: str):
        """Callback for GPS data"""
        try:
            reading = GpsReading(
                timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                sensor_id=sensor_id,
                latitude=msg.latitude,
                longitude=msg.longitude,
                altitude=msg.altitude,
                horizontal_accuracy=msg.position_covariance[0], # Typically variance, sqrt for std dev
                vertical_accuracy=msg.position_covariance[8]  # Typically variance, sqrt for std dev
            )
            
            if sensor_id in self.gps_callbacks and self.gps_callbacks[sensor_id] is not None:
                self.gps_callbacks[sensor_id](reading)
                
        except Exception as e:
            logger.error(f"Error in GPS callback for {sensor_id}: {e}", exc_info=True)
            
    def publish_pose(self, pose: np.ndarray, timestamp: float, frame_id: str = "world", child_frame_id: str = "base_link"):
        """Publish pose as Odometry message
        
        Args:
            pose: 4x4 transformation matrix (world to child_frame_id, e.g. world to base_link)
            timestamp: Timestamp in seconds
            frame_id: The coordinate frame ID for the pose.
            child_frame_id: The coordinate frame ID of the object whose pose is being reported.
        """
        msg = Odometry()
        current_time = self.get_clock().now().to_msg()
        msg.header.stamp = current_time # Use current ROS time for message header for consistency
        msg.header.frame_id = frame_id
        msg.child_frame_id = child_frame_id
        
        msg.pose.pose.position.x = pose[0, 3]
        msg.pose.pose.position.y = pose[1, 3]
        msg.pose.pose.position.z = pose[2, 3]
        
        try:
            from scipy.spatial.transform import Rotation
            rotation = Rotation.from_matrix(pose[:3, :3])
            quaternion = rotation.as_quat()  # [x, y, z, w]
            msg.pose.pose.orientation.x = quaternion[0]
            msg.pose.pose.orientation.y = quaternion[1]
            msg.pose.pose.orientation.z = quaternion[2]
            msg.pose.pose.orientation.w = quaternion[3]
        except ImportError:
            logger.warning("scipy not installed, cannot convert rotation matrix to quaternion for publishing pose.")
            # Set to default identity quaternion if scipy is not available
            msg.pose.pose.orientation.x = 0.0
            msg.pose.pose.orientation.y = 0.0
            msg.pose.pose.orientation.z = 0.0
            msg.pose.pose.orientation.w = 1.0
        
        # Covariance is not set in this basic example, but should be for a proper Odometry message.
        # msg.pose.covariance = [...] 
        self.pose_publisher.publish(msg)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop() 