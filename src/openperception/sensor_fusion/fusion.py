import numpy as np
import cv2
from dataclasses import dataclass, field # Added field
from typing import List, Dict, Tuple, Optional, Any
import logging
import time

# Import from the unified utils package
from openperception.utils.transformations import euler_to_rotation_matrix

# Import configuration
from openperception.config.config import load_config

# Load configuration
config = load_config()

logger = logging.getLogger(__name__)

@dataclass
class SensorReading:
    """Base class for sensor readings"""
    timestamp: float
    sensor_id: str
    
@dataclass
class CameraReading(SensorReading):
    """Camera image reading"""
    image: np.ndarray
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    # Optionally, add pose of the camera if known at the time of reading
    pose: Optional[np.ndarray] = None # 4x4 transformation matrix (world_to_camera or camera_to_world)
    
@dataclass
class LidarReading(SensorReading):
    """LiDAR point cloud reading"""
    points: np.ndarray  # Nx3 array of points in sensor frame
    intensities: Optional[np.ndarray] = None  # N intensity values
    # Optionally, add pose of the LiDAR if known
    pose: Optional[np.ndarray] = None # 4x4 transformation matrix
    
@dataclass
class ImuReading(SensorReading):
    """IMU reading with acceleration and angular velocity"""
    acceleration: np.ndarray  # 3D acceleration vector (m/s^2)
    angular_velocity: np.ndarray  # 3D angular velocity vector (rad/s)
    orientation: Optional[np.ndarray] = None  # Orientation as quaternion [x,y,z,w] or 3x3 rotation matrix
    orientation_covariance: Optional[np.ndarray] = None # Covariance for orientation
    angular_velocity_covariance: Optional[np.ndarray] = None # Covariance for angular velocity
    linear_acceleration_covariance: Optional[np.ndarray] = None # Covariance for linear acceleration

@dataclass
class GpsReading(SensorReading):
    """GPS reading with position and accuracy"""
    latitude: float  # Degrees
    longitude: float # Degrees
    altitude: float  # Meters
    covariance: Optional[np.ndarray] = None # 3x3 covariance matrix for position (ENU - East, North, Up)
    covariance_type: int = 0 # Type of covariance matrix (0: unknown, 1: diagonal stddev, 2: full matrix)
    # Deprecating horizontal_accuracy and vertical_accuracy in favor of covariance matrix
    # horizontal_accuracy: float # Meters
    # vertical_accuracy: float # Meters
    
@dataclass
class SensorExtrinsics:
    """Extrinsic parameters of a sensor relative to a base_link or common reference frame."""
    sensor_id: str
    # 4x4 transformation matrix from sensor frame to the base_link frame (T_base_sensor)
    transform_to_base: np.ndarray = field(default_factory=lambda: np.eye(4))
    # Optionally, store the reverse transform for convenience
    transform_from_base: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.transform_from_base is None and self.transform_to_base is not None:
            try:
                self.transform_from_base = np.linalg.inv(self.transform_to_base)
            except np.linalg.LinAlgError:
                logger.error(f"Failed to invert transform_to_base for sensor {self.sensor_id}. Ensure it is a valid SE(3) matrix.")
                self.transform_from_base = np.eye(4) # Fallback

class SensorFusion:
    """Sensor fusion system for combining data from multiple sensors."""
    
    def __init__(self, max_buffer_size: Optional[int] = None):
        """Initialize sensor fusion system
        Args:
            max_buffer_size: Maximum number of readings to store per sensor. 
                           If None, uses value from config.
        """
        self.extrinsics: Dict[str, SensorExtrinsics] = {}
        
        self.camera_buffer: Dict[str, List[CameraReading]] = {}
        self.lidar_buffer: Dict[str, List[LidarReading]] = {}
        self.imu_buffer: Dict[str, List[ImuReading]] = {}
        self.gps_buffer: Dict[str, List[GpsReading]] = {}
        
        # Get buffer size from config if not provided
        self.max_buffer_size = max_buffer_size or config['sensor_fusion'].get('max_buffer_size', 100)
        
        # Latest fused state (example: EKF state)
        self.latest_state = {
            'timestamp': 0.0,
            'position': np.zeros(3),  # In a global/map frame (e.g., ENU)
            'orientation': np.array([0, 0, 0, 1]),  # Quaternion [x,y,z,w] in global/map frame
            'velocity': np.zeros(3),  # In global/map frame
            'covariance': np.eye(9) # Example: 3 pos, 3 vel, 3 orient (e.g. error angles)
        }
        self._last_imu_reading: Optional[ImuReading] = None

    def register_sensor(self, sensor_id: str, sensor_type: str, transform_to_base: Optional[np.ndarray] = None):
        """Register a sensor with the fusion system.
        
        Args:
            sensor_id: Unique ID for the sensor.
            sensor_type: Type of sensor (camera, lidar, imu, gps).
            transform_to_base: 4x4 transformation matrix from sensor frame to the base_link/robot frame.
                               If None, an identity matrix is assumed (sensor is at base_link origin).
        """
        if transform_to_base is None:
            logger.warning(f"No transform_to_base provided for sensor {sensor_id}. Assuming identity.")
            transform_to_base = np.eye(4)
        
        self.extrinsics[sensor_id] = SensorExtrinsics(sensor_id, transform_to_base)
        
        # Initialize buffer for this sensor
        if sensor_type == 'camera':
            self.camera_buffer.setdefault(sensor_id, [])
        elif sensor_type == 'lidar':
            self.lidar_buffer.setdefault(sensor_id, [])
        elif sensor_type == 'imu':
            self.imu_buffer.setdefault(sensor_id, [])
        elif sensor_type == 'gps':
            self.gps_buffer.setdefault(sensor_id, [])
        else:
            logger.warning(f"Unknown sensor type '{sensor_type}' for sensor_id '{sensor_id}'. Not creating a dedicated buffer.")

    def _add_reading_to_buffer(self, buffer: List[Any], reading: SensorReading):
        """Helper to add reading to a buffer and trim it."""
        buffer.append(reading)
        if len(buffer) > self.max_buffer_size:
            buffer.pop(0)
            
    def add_camera_reading(self, reading: CameraReading):
        if reading.sensor_id not in self.camera_buffer:
            logger.warning(f"Camera {reading.sensor_id} not registered. Registering with identity transform.")
            self.register_sensor(reading.sensor_id, 'camera')
        self._add_reading_to_buffer(self.camera_buffer[reading.sensor_id], reading)
            
    def add_lidar_reading(self, reading: LidarReading):
        if reading.sensor_id not in self.lidar_buffer:
            logger.warning(f"LiDAR {reading.sensor_id} not registered. Registering with identity transform.")
            self.register_sensor(reading.sensor_id, 'lidar')
        self._add_reading_to_buffer(self.lidar_buffer[reading.sensor_id], reading)
            
    def add_imu_reading(self, reading: ImuReading):
        if reading.sensor_id not in self.imu_buffer:
            logger.warning(f"IMU {reading.sensor_id} not registered. Registering with identity transform.")
            self.register_sensor(reading.sensor_id, 'imu')
        self._add_reading_to_buffer(self.imu_buffer[reading.sensor_id], reading)
        self._last_imu_reading = reading # Keep track of last IMU for prediction
        # Potentially trigger a prediction step if using an EKF
        # self._predict_state_with_imu(reading)

    def add_gps_reading(self, reading: GpsReading):
        if reading.sensor_id not in self.gps_buffer:
            logger.warning(f"GPS {reading.sensor_id} not registered. Registering with identity transform.")
            self.register_sensor(reading.sensor_id, 'gps')
        self._add_reading_to_buffer(self.gps_buffer[reading.sensor_id], reading)
        # Potentially trigger an update step if using an EKF
        # self._update_state_with_gps(reading)
            
    def update_fusion(self, timestamp: float) -> Dict[str, Any]:
        """Update fusion at the specified timestamp.
        This is a placeholder for a more sophisticated fusion algorithm (e.g., EKF trigger).
        Currently, it just gathers the latest available data near the timestamp.
        """
        # In a real system, this method might trigger EKF predict/update cycles
        # or process data based on sensor availability and timestamps.
        
        # For now, let's assume this is called when a new state estimate is desired.
        # The _fusion_algorithm would be the core of an EKF or similar filter.
        
        # Find closest readings from each sensor (example, not necessarily how an EKF would use it)
        # camera_data = {sid: self._find_closest_reading(buf, timestamp) for sid, buf in self.camera_buffer.items()}
        # lidar_data = {sid: self._find_closest_reading(buf, timestamp) for sid, buf in self.lidar_buffer.items()}
        # imu_data = {sid: self._find_closest_reading(buf, timestamp) for sid, buf in self.imu_buffer.items()}
        # gps_data = {sid: self._find_closest_reading(buf, timestamp) for sid, buf in self.gps_buffer.items()}

        # The actual fusion logic (EKF predict, EKF update) would be called based on sensor arrival times.
        # This simplified version just notes the call.
        logger.debug(f"Fusion update called for timestamp {timestamp}")

        # The _fusion_algorithm is more of a conceptual placeholder here
        # self._fusion_algorithm(timestamp, camera_data, lidar_data, imu_data, gps_data)
        
        return self.get_fused_state()
                
    def _find_closest_reading(self, buffer: List[SensorReading], timestamp: float, max_delta: float = 0.05) -> Optional[SensorReading]:
        """Find the reading closest to the specified timestamp within max_delta."""
        if not buffer:
            return None
        
        closest_reading = min(buffer, key=lambda r: abs(r.timestamp - timestamp))
        
        if abs(closest_reading.timestamp - timestamp) > max_delta:
            return None # Too far away in time
            
        return closest_reading
        
    def _fusion_algorithm(self, timestamp: float, 
                          camera_readings: Optional[Dict[str, CameraReading]] = None,
                          lidar_readings: Optional[Dict[str, LidarReading]] = None,
                          imu_readings: Optional[Dict[str, ImuReading]] = None,
                          gps_readings: Optional[Dict[str, GpsReading]] = None):
        """Placeholder for the actual sensor fusion algorithm (e.g., EKF).
        
        This method would be the core of state estimation.
        It would involve: 
        1. Prediction step (e.g., using IMU data).
        2. Update step (e.g., using GPS, camera, or LiDAR features).
        """
        # This is a highly simplified version. A real EKF is much more complex.
        logger.info(f"Generic fusion algorithm called at timestamp {timestamp}.")

        # --- IMU based prediction (Conceptual) --- 
        if self._last_imu_reading:
            dt = timestamp - self.latest_state['timestamp']
            if dt > 0:
                # Simplified IMU integration (very basic, not a proper EKF predict)
                # This should use proper EKF state transition equations
                accel = self._last_imu_reading.acceleration
                ang_vel = self._last_imu_reading.angular_velocity
                
                # Transform accel to world frame (requires current orientation)
                # Simplified: assume orientation is world frame for now
                # R = quaternion_to_rotation_matrix(self.latest_state['orientation'])
                # world_accel = R @ accel - np.array([0,0,9.81]) # Example: remove gravity
                world_accel = accel - np.array([0,0,9.81]) # If IMU gives world frame accel (unlikely) or gravity compensated

                self.latest_state['position'] += self.latest_state['velocity'] * dt + 0.5 * world_accel * dt**2
                self.latest_state['velocity'] += world_accel * dt
                
                # Orientation update (simplified, needs quaternion kinematics)
                # For example: q_new = q_old * delta_q_from_angular_velocity(ang_vel, dt)
                # Pass for now
                logger.debug(f"IMU-based prediction performed for dt={dt}")

        # --- GPS based update (Conceptual) ---
        if gps_readings:
            gps = next(iter(gps_readings.values())) # Use first available GPS
            if gps:
                # This is where an EKF update step would use GPS measurement and its covariance
                # For simplicity, directly set position (if GPS is trusted highly or for initialization)
                # This needs conversion from Lat/Lon/Alt to a local metric frame (e.g., ENU)
                # Placeholder for lat/lon to ENU conversion:
                # current_pos_enu = self.convert_gps_to_enu(gps.latitude, gps.longitude, gps.altitude)
                # self.latest_state['position'] = current_pos_enu
                logger.info(f"GPS update: Lat={gps.latitude}, Lon={gps.longitude}, Alt={gps.altitude}")
                # In EKF, you'd calculate innovation, Kalman gain, and update state & covariance.

        # --- Camera/LiDAR based update (Conceptual) ---
        # Visual/LiDAR odometry or SLAM results would update pose and map.
        # This is where Visual Odometry, Loop Closure, etc. would feed into the EKF.

        self.latest_state['timestamp'] = timestamp
        # Covariance update would also happen here in a real filter.

    def project_lidar_to_camera(self, lidar_sensor_id: str, camera_sensor_id: str, 
                                lidar_points: np.ndarray, 
                                image_shape: Tuple[int, int],
                                camera_matrix: Optional[np.ndarray] = None, 
                                dist_coeffs: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Project LiDAR points from a specific LiDAR sensor onto a specific camera image plane.
        
        Args:
            lidar_sensor_id: ID of the LiDAR sensor.
            camera_sensor_id: ID of the Camera sensor.
            lidar_points: Nx3 array of LiDAR points in the LiDAR sensor's frame.
            image_shape: Tuple (height, width) of the camera image.
            camera_matrix: 3x3 camera intrinsic matrix. If None, uses default from config.
            dist_coeffs: Distortion coefficients for the camera. If None, uses default from config.
            
        Returns:
            Nx2 array of projected 2D points in the image plane, or None if error.
        """
        # Use default intrinsics from config if not provided
        if camera_matrix is None:
            camera_matrix = np.array(config['sensor_fusion']['default_camera_intrinsics'])
        if dist_coeffs is None:
            dist_coeffs = np.array(config['sensor_fusion']['default_distortion_coeffs'])
            
        if lidar_sensor_id not in self.extrinsics:
            logger.warning(f"LiDAR sensor {lidar_sensor_id} extrinsics not registered. Using default from config.")
            T_base_lidar = np.array(config['sensor_fusion']['default_lidar_extrinsics'])
        else:
            T_base_lidar = self.extrinsics[lidar_sensor_id].transform_to_base
            
        if camera_sensor_id not in self.extrinsics:
            logger.warning(f"Camera sensor {camera_sensor_id} extrinsics not registered. Using default from config.")
            T_base_camera = np.array(config['sensor_fusion']['default_camera_extrinsics'])
        else:
            T_base_camera = self.extrinsics[camera_sensor_id].transform_to_base
        
        # Transform from camera frame to base: T_base_camera
        # Transform from base to camera frame: T_camera_base = inv(T_base_camera)
        try:
            T_camera_base = np.linalg.inv(T_base_camera)
        except np.linalg.LinAlgError:
            logger.error(f"Failed to invert camera transform for {camera_sensor_id}")
            return None

        # Transformation from LiDAR frame to Camera frame:
        # T_camera_lidar = T_camera_base @ T_base_lidar
        T_camera_lidar = T_camera_base @ T_base_lidar
        
        # LiDAR points are (N,3). Add homogeneous coordinate.
        points_lidar_hom = np.hstack([lidar_points, np.ones((lidar_points.shape[0], 1))])
        
        # Transform points to camera frame: P_camera = T_camera_lidar @ P_lidar_hom.T
        points_camera_hom = (T_camera_lidar @ points_lidar_hom.T).T
        
        # Keep only points in front of the camera (Z > 0 in camera frame)
        points_in_front = points_camera_hom[points_camera_hom[:, 2] > 0]
        
        if points_in_front.shape[0] == 0:
            return np.array([]) # No points to project
            
        # Project points to image plane using OpenCV
        # cv2.projectPoints expects objectPoints (Nx3), rvec, tvec, cameraMatrix, distCoeffs
        # Here, points_in_front[:, :3] are already in camera coordinates.
        # So, rvec and tvec (camera extrinsics relative to world) are zero vectors if points are already in camera frame.
        rvec = np.zeros(3, dtype=np.float32)
        tvec = np.zeros(3, dtype=np.float32)
        
        projected_points, _ = cv2.projectPoints(
            points_in_front[:, :3].reshape(-1, 3), # Ensure it's (NumPoints, 3)
            rvec, tvec, 
            camera_matrix, dist_coeffs
        )
        
        projected_points = projected_points.squeeze() # Shape from (N,1,2) to (N,2)
        if projected_points.ndim == 1 and projected_points.size == 2: # Single point case
            projected_points = projected_points.reshape(1,2)

        # Filter points that are outside the image dimensions
        h, w = image_shape
        if projected_points.size > 0:
            mask = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < w) & \
                   (projected_points[:, 1] >= 0) & (projected_points[:, 1] < h)
            projected_points_on_image = projected_points[mask]
            return projected_points_on_image
        else:
            return np.array([])

    def get_fused_state(self) -> Dict[str, Any]:
        """Get latest fused state, making a copy to prevent direct modification."""
        # Ensure deep copy for mutable types like numpy arrays if necessary
        state_copy = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k,v in self.latest_state.items()}
        return state_copy
        
    def clear_buffers(self):
        """Clear all sensor data buffers."""
        for sensor_id in self.camera_buffer: self.camera_buffer[sensor_id] = []
        for sensor_id in self.lidar_buffer: self.lidar_buffer[sensor_id] = []
        for sensor_id in self.imu_buffer: self.imu_buffer[sensor_id] = []
        for sensor_id in self.gps_buffer: self.gps_buffer[sensor_id] = []
        self._last_imu_reading = None
        logger.info("All sensor buffers cleared.")

    # --- Placeholder for coordinate transformations (should be in utils) ---
    # def convert_gps_to_enu(self, lat, lon, alt):
    #     # This would involve a local ENU origin (e.g., first GPS reading)
    #     # and a WGS84 to ECEF to ENU transformation.
    #     # For simplicity, returning placeholder values.
    #     logger.warning("GPS to ENU conversion is a placeholder.")
    #     return np.array([lon * 1e5, lat * 1e5, alt]) # Highly inaccurate, just for structure

# Example Usage (for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    fusion_system = SensorFusion(max_buffer_size=10)

    # Register sensors
    cam_extrinsics = np.eye(4) # Assume camera is at base_link
    lidar_extrinsics = np.eye(4) # Assume lidar is at base_link
    imu_extrinsics = np.eye(4)
    gps_extrinsics = np.eye(4)

    fusion_system.register_sensor("camera_front", "camera", cam_extrinsics)
    fusion_system.register_sensor("lidar_top", "lidar", lidar_extrinsics)
    fusion_system.register_sensor("imu_main", "imu", imu_extrinsics)
    fusion_system.register_sensor("gps_main", "gps", gps_extrinsics)

    # Simulate adding data
    ts = time.time()
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cam_matrix = np.array([[500,0,320],[0,500,240],[0,0,1]], dtype=np.float32)
    dist_coeffs = np.zeros(5, dtype=np.float32)

    fusion_system.add_camera_reading(CameraReading(ts, "camera_front", dummy_image, cam_matrix, dist_coeffs))
    fusion_system.add_lidar_reading(LidarReading(ts + 0.01, "lidar_top", np.random.rand(100,3) * 10))
    fusion_system.add_imu_reading(ImuReading(ts + 0.005, "imu_main", np.random.rand(3), np.random.rand(3)))
    fusion_system.add_gps_reading(GpsReading(ts + 0.02, "gps_main", 34.05, -118.24, 100.0, np.eye(3)*0.1, 1))

    print(f"Initial fused state: {fusion_system.get_fused_state()}")

    # Simulate a fusion update
    fusion_system.update_fusion(ts + 0.1)
    print(f"Fused state after update: {fusion_system.get_fused_state()}")

    # Test LiDAR to Camera projection
    lidar_points_test = np.array([[1.0, 0.0, 5.0], [2.0, 0.5, 10.0], [-1.0, -0.5, 8.0]]) # X, Y, Z in lidar frame
    projected = fusion_system.project_lidar_to_camera(
        "lidar_top", "camera_front", 
        lidar_points_test, (480,640), cam_matrix, dist_coeffs
    )
    if projected is not None:
        print(f"Projected LiDAR points ({projected.shape[0]} points):
 {projected}")
        # You could draw these on dummy_image
        img_with_projections = dummy_image.copy()
        for pt in projected.astype(int):
            cv2.circle(img_with_projections, tuple(pt), 3, (0,255,0), -1)
        # cv2.imshow("Lidar Projections", img_with_projections)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("LiDAR projection failed or no points projected.")

    fusion_system.clear_buffers() 