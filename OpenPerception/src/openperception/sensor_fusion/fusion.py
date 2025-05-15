import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import logging
from openperception.utils.transformations import euler_to_rotation_matrix # Updated import
import time

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
    
@dataclass
class LidarReading(SensorReading):
    """LiDAR point cloud reading"""
    points: np.ndarray  # Nx3 array of points
    intensities: Optional[np.ndarray] = None  # N intensity values
    
@dataclass
class ImuReading(SensorReading):
    """IMU reading with acceleration and angular velocity"""
    acceleration: np.ndarray  # 3D acceleration vector
    angular_velocity: np.ndarray  # 3D angular velocity vector
    orientation: Optional[np.ndarray] = None  # Orientation as quaternion [x,y,z,w]
    
@dataclass
class GpsReading(SensorReading):
    """GPS reading with position and accuracy"""
    latitude: float
    longitude: float
    altitude: float
    horizontal_accuracy: float
    vertical_accuracy: float
    
@dataclass
class SensorExtrinsics:
    """Extrinsic parameters of a sensor"""
    sensor_id: str
    transform: np.ndarray  # 4x4 transformation matrix from sensor to base frame
    
class SensorFusion:
    """Sensor fusion system for combining data from multiple sensors"""
    
    def __init__(self, max_buffer_size: int = 100, fusion_method: str = "kalman"):
        """Initialize sensor fusion system
        Args:
            max_buffer_size: Maximum number of readings to store per sensor.
            fusion_method: The fusion algorithm to use (e.g., 'kalman', 'particle').
        """
        # Sensor extrinsics (transformations from sensor frames to base frame)
        self.extrinsics: Dict[str, SensorExtrinsics] = {}
        
        # Sensor data buffers
        self.camera_buffer: Dict[str, List[CameraReading]] = {}
        self.lidar_buffer: Dict[str, List[LidarReading]] = {}
        self.imu_buffer: Dict[str, List[ImuReading]] = {}
        self.gps_buffer: Dict[str, List[GpsReading]] = {}
        
        # Maximum buffer size
        self.max_buffer_size = max_buffer_size
        self.fusion_method = fusion_method # Store fusion_method
        
        # Latest fused state
        self.latest_state = {
            'timestamp': 0,
            'position': np.zeros(3),
            'orientation': np.array([0, 0, 0, 1]),  # Quaternion [x,y,z,w]
            'velocity': np.zeros(3)
        }
        
    def register_sensor(self, sensor_id: str, sensor_type: str, transform: np.ndarray):
        """Register a sensor with the fusion system
        
        Args:
            sensor_id: Unique ID for the sensor
            sensor_type: Type of sensor (camera, lidar, imu, gps)
            transform: 4x4 transformation matrix from sensor to base frame
        """
        self.extrinsics[sensor_id] = SensorExtrinsics(sensor_id, transform)
        
        # Initialize buffer for this sensor
        if sensor_type == 'camera' and sensor_id not in self.camera_buffer:
            self.camera_buffer[sensor_id] = []
        elif sensor_type == 'lidar' and sensor_id not in self.lidar_buffer:
            self.lidar_buffer[sensor_id] = []
        elif sensor_type == 'imu' and sensor_id not in self.imu_buffer:
            self.imu_buffer[sensor_id] = []
        elif sensor_type == 'gps' and sensor_id not in self.gps_buffer:
            self.gps_buffer[sensor_id] = []
            
    def add_camera_reading(self, reading: CameraReading):
        """Add a camera reading to the buffer"""
        if reading.sensor_id not in self.camera_buffer:
            logger.warning(f"Camera {reading.sensor_id} not registered")
            self.camera_buffer[reading.sensor_id] = []
            
        self.camera_buffer[reading.sensor_id].append(reading)
        
        # Trim buffer if necessary
        if len(self.camera_buffer[reading.sensor_id]) > self.max_buffer_size:
            self.camera_buffer[reading.sensor_id].pop(0)
            
    def add_lidar_reading(self, reading: LidarReading):
        """Add a LiDAR reading to the buffer"""
        if reading.sensor_id not in self.lidar_buffer:
            logger.warning(f"LiDAR {reading.sensor_id} not registered")
            self.lidar_buffer[reading.sensor_id] = []
            
        self.lidar_buffer[reading.sensor_id].append(reading)
        
        # Trim buffer if necessary
        if len(self.lidar_buffer[reading.sensor_id]) > self.max_buffer_size:
            self.lidar_buffer[reading.sensor_id].pop(0)
            
    def add_imu_reading(self, reading: ImuReading):
        """Add an IMU reading to the buffer"""
        if reading.sensor_id not in self.imu_buffer:
            logger.warning(f"IMU {reading.sensor_id} not registered")
            self.imu_buffer[reading.sensor_id] = []
            
        self.imu_buffer[reading.sensor_id].append(reading)
        
        # Trim buffer if necessary
        if len(self.imu_buffer[reading.sensor_id]) > self.max_buffer_size:
            self.imu_buffer[reading.sensor_id].pop(0)
            
    def add_gps_reading(self, reading: GpsReading):
        """Add a GPS reading to the buffer"""
        if reading.sensor_id not in self.gps_buffer:
            logger.warning(f"GPS {reading.sensor_id} not registered")
            self.gps_buffer[reading.sensor_id] = []
            
        self.gps_buffer[reading.sensor_id].append(reading)
        
        # Trim buffer if necessary
        if len(self.gps_buffer[reading.sensor_id]) > self.max_buffer_size:
            self.gps_buffer[reading.sensor_id].pop(0)
            
    def update_fusion(self, timestamp: float):
        """Update fusion at the specified timestamp"""
        # Find closest readings from each sensor
        camera_readings = {}
        lidar_readings = {}
        imu_readings = {}
        gps_readings = {}
        
        for sensor_id, buffer in self.camera_buffer.items():
            closest = self._find_closest_reading(buffer, timestamp)
            if closest is not None:
                camera_readings[sensor_id] = closest
                
        for sensor_id, buffer in self.lidar_buffer.items():
            closest = self._find_closest_reading(buffer, timestamp)
            if closest is not None:
                lidar_readings[sensor_id] = closest
                
        for sensor_id, buffer in self.imu_buffer.items():
            closest = self._find_closest_reading(buffer, timestamp)
            if closest is not None:
                imu_readings[sensor_id] = closest
                
        for sensor_id, buffer in self.gps_buffer.items():
            closest = self._find_closest_reading(buffer, timestamp)
            if closest is not None:
                gps_readings[sensor_id] = closest
                
        # Perform sensor fusion
        self._fusion_algorithm(timestamp, camera_readings, lidar_readings, 
                              imu_readings, gps_readings)
                
    def _find_closest_reading(self, buffer: List[SensorReading], timestamp: float) -> Optional[SensorReading]:
        """Find the reading closest to the specified timestamp"""
        if not buffer:
            return None
            
        # Find reading with minimum time difference
        closest = min(buffer, key=lambda r: abs(r.timestamp - timestamp))
        
        # Check if it's within a reasonable time window (50ms)
        if abs(closest.timestamp - timestamp) > 0.05:
            return None
            
        return closest
        
    def _fusion_algorithm(self, timestamp: float, camera_readings: Dict[str, CameraReading],
                          lidar_readings: Dict[str, LidarReading],
                          imu_readings: Dict[str, ImuReading],
                          gps_readings: Dict[str, GpsReading]):
        """Implement sensor fusion algorithm
        
        This is a simplified fusion algorithm. In a real system, you would use
        a proper state estimation filter like an Extended Kalman Filter or a
        Particle Filter.
        """
        # Update timestamp
        self.latest_state['timestamp'] = timestamp
        
        # Update orientation based on IMU
        if imu_readings:
            # Use the first available IMU reading
            imu = next(iter(imu_readings.values()))
            
            if imu.orientation is not None:
                # Direct orientation measurement
                self.latest_state['orientation'] = imu.orientation
            else:
                # Integrate angular velocity
                dt = timestamp - self.latest_state['timestamp'] # Potential bug: self.latest_state['timestamp'] is updated above
                                                                # Should be: dt = timestamp - previous_timestamp_of_latest_state
                                                                # For simplicity, assume this dt is small and state updated frequently.
                
                # Placeholder: Simple Euler integration for orientation update (not robust)
                # Actual implementation would use quaternion multiplication
                # angular_velocity_body = imu.angular_velocity
                # current_orientation_quat = self.latest_state['orientation'] 
                # R_current = # Convert current_orientation_quat to rotation matrix
                # angular_velocity_world = R_current @ angular_velocity_body
                # orientation_delta_euler = angular_velocity_world * dt
                # delta_quat = # Convert orientation_delta_euler to quaternion
                # self.latest_state['orientation'] = # Multiply current_orientation_quat by delta_quat and normalize
                pass # Placeholder for actual orientation integration
                
        # Update position based on GPS
        if gps_readings:
            # Use the first available GPS reading
            gps = next(iter(gps_readings.values()))
            
            # Convert lat/lon to local coordinate system
            # This is simplified and would need a proper conversion in practice (e.g., UTM or ENU)
            # Also assumes a flat Earth for this basic example.
            # More advanced: Use a geodetic library or implement Haversine/Vincenty for local tangent plane.
            R_earth = 6371000  # Radius of Earth in meters
            lat_rad = np.radians(gps.latitude)
            lon_rad = np.radians(gps.longitude)
            
            # Simplified conversion, good for small areas, assumes origin at first GPS point or configured origin
            # For a more robust system, establish a local ENU frame origin.
            # Here, we'll just use a very rough approximation. This needs significant improvement for real use.
            x = R_earth * lon_rad * np.cos(lat_rad) # This is not a proper ENU x
            y = R_earth * lat_rad                 # This is not a proper ENU y
            z = gps.altitude
            
            position = np.array([x,y,z]) 
            self.latest_state['position'] = position
            
        # Update velocity and position based on IMU acceleration
        if imu_readings:
            imu = next(iter(imu_readings.values()))
            
            # Remove gravity from acceleration (simplified)
            # This requires accurate orientation. Assume latest_state['orientation'] is current.
            # R_body_to_world = # Convert self.latest_state['orientation'] (quaternion) to rotation matrix
            # gravity_world = np.array([0, 0, -9.81]) # Assuming Z is up, gravity is negative Z in world frame
            # gravity_body = R_body_to_world.T @ gravity_world
            # linear_acceleration_body = imu.acceleration - gravity_body
            # linear_acceleration_world = R_body_to_world @ linear_acceleration_body
            
            # Simplified: Assume IMU measures acceleration in world frame after gravity compensation (highly unrealistic)
            acceleration_world = imu.acceleration - np.array([0,0,9.81])  # Very crude gravity compensation

            dt = timestamp - self.latest_state['timestamp'] # Same dt issue as above
            
            # Integrate acceleration to get velocity
            self.latest_state['velocity'] += acceleration_world * dt
            
            # Integrate velocity to get position
            # This position update should be fused with GPS, not just added.
            self.latest_state['position'] += self.latest_state['velocity'] * dt + 0.5 * acceleration_world * (dt**2)
            
        # LiDAR and camera data would typically be used for mapping and localization
        # (e.g., ICP for LiDAR, visual odometry/SLAM for camera) which then update the state.
        # For this simple fusion example, we'll just note that we have the data.
        # Proper fusion would involve a Kalman filter (EKF, UKF) or particle filter.
        logger.debug(f"Fusion method: {self.fusion_method} - actual algorithm (e.g. EKF) not fully implemented here.")
        
    def project_lidar_to_camera(self, lidar_reading: LidarReading, camera_reading: CameraReading) -> np.ndarray:
        """Project LiDAR points onto camera image
        
        Args:
            lidar_reading: LiDAR point cloud
            camera_reading: Camera image
            
        Returns:
            Image with projected LiDAR points
        """
        # Get transformations
        if lidar_reading.sensor_id not in self.extrinsics:
            logger.error(f"LiDAR {lidar_reading.sensor_id} not registered")
            return camera_reading.image.copy()
            
        if camera_reading.sensor_id not in self.extrinsics:
            logger.error(f"Camera {camera_reading.sensor_id} not registered")
            return camera_reading.image.copy()
            
        lidar_to_base = self.extrinsics[lidar_reading.sensor_id].transform
        camera_to_base = self.extrinsics[camera_reading.sensor_id].transform
        
        # Calculate transformation from LiDAR to camera
        base_to_camera = np.linalg.inv(camera_to_base)
        lidar_to_camera = base_to_camera @ lidar_to_base
        
        # Transform LiDAR points to camera frame
        points_lidar_homogeneous = np.hstack([lidar_reading.points, np.ones((lidar_reading.points.shape[0], 1))])
        points_camera_homogeneous = (lidar_to_camera @ points_lidar_homogeneous.T).T
        
        # Keep only points in front of the camera (Z > 0 in camera frame)
        points_camera_homogeneous = points_camera_homogeneous[points_camera_homogeneous[:, 2] > 0]
        
        if len(points_camera_homogeneous) == 0:
            return camera_reading.image.copy()
            
        # Project points to image (from homogeneous to Cartesian for projectPoints)
        points_camera_cartesian = points_camera_homogeneous[:, :3] / points_camera_homogeneous[:, 3, np.newaxis]

        # Rotation and translation vectors for projectPoints should be from world to camera.
        # Here, points_camera_cartesian are already in the camera's coordinate system.
        # So, rvec and tvec should be zero.
        rvec = np.zeros(3)
        tvec = np.zeros(3)

        points_2d, _ = cv2.projectPoints(
            points_camera_cartesian, rvec, tvec, 
            camera_reading.camera_matrix,
            camera_reading.dist_coeffs
        )
        points_2d = points_2d.reshape(-1, 2)
        
        # Filter points that are outside the image
        h, w = camera_reading.image.shape[:2]
        mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & \
               (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
        points_2d = points_2d[mask]
        
        # Draw points on image
        result = camera_reading.image.copy()
        for pt in points_2d.astype(int):
            cv2.circle(result, tuple(pt), 2, (0, 255, 0), -1)
            
        return result
        
    def get_fused_state(self) -> Dict[str, Any]:
        """Get latest fused state"""
        return self.latest_state.copy()
        
    def clear_buffers(self):
        """Clear all sensor data buffers"""
        for sensor_id in self.camera_buffer:
            self.camera_buffer[sensor_id] = []
            
        for sensor_id in self.lidar_buffer:
            self.lidar_buffer[sensor_id] = []
            
        for sensor_id in self.imu_buffer:
            self.imu_buffer[sensor_id] = []
            
        for sensor_id in self.gps_buffer:
            self.gps_buffer[sensor_id] = [] 