"""
Transformation utilities for OpenPerception.

This module provides various transformation functions for 3D transformations, 
rotations, and coordinate system conversions.
"""

import numpy as np
from typing import Tuple, Optional, Union, List

def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles to rotation matrix (ZYX convention).
    
    Args:
        roll: Roll angle in radians (rotation around X-axis)
        pitch: Pitch angle in radians (rotation around Y-axis)
        yaw: Yaw angle in radians (rotation around Z-axis)
        
    Returns:
        3x3 rotation matrix
    """
    # Precompute sines and cosines
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    
    # Construct rotation matrices
    R_x = np.array([
        [1, 0, 0],
        [0, cos_r, -sin_r],
        [0, sin_r, cos_r]
    ])
    
    R_y = np.array([
        [cos_p, 0, sin_p],
        [0, 1, 0],
        [-sin_p, 0, cos_p]
    ])
    
    R_z = np.array([
        [cos_y, -sin_y, 0],
        [sin_y, cos_y, 0],
        [0, 0, 1]
    ])
    
    # Combine rotations: R = R_z * R_y * R_x
    R = R_z @ R_y @ R_x
    
    return R

def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """Convert rotation matrix to Euler angles (ZYX convention).
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    # Check for gimbal lock (pitch close to +/- pi/2)
    if abs(abs(R[2, 0]) - 1.0) < 1e-10:
        # Gimbal lock
        if R[2, 0] < 0:
            # Pitch = pi/2
            yaw = np.arctan2(R[0, 1], R[0, 2])
            pitch = np.pi/2
            roll = 0
        else:
            # Pitch = -pi/2
            yaw = np.arctan2(-R[0, 1], -R[0, 2])
            pitch = -np.pi/2
            roll = 0
    else:
        pitch = -np.arcsin(R[2, 0])
        roll = np.arctan2(R[2, 1]/np.cos(pitch), R[2, 2]/np.cos(pitch))
        yaw = np.arctan2(R[1, 0]/np.cos(pitch), R[0, 0]/np.cos(pitch))
    
    return roll, pitch, yaw

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix.
    
    Args:
        q: Quaternion as [x, y, z, w] (scalar last)
        
    Returns:
        3x3 rotation matrix
    """
    # Normalize quaternion
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    
    # Construct rotation matrix
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    
    return R

def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Quaternion as [x, y, z, w] (scalar last)
    """
    # Compute trace
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    
    return np.array([x, y, z, w])

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles to quaternion (ZYX convention).
    
    Args:
        roll: Roll angle in radians (rotation around X-axis)
        pitch: Pitch angle in radians (rotation around Y-axis)
        yaw: Yaw angle in radians (rotation around Z-axis)
        
    Returns:
        Quaternion as [x, y, z, w] (scalar last)
    """
    # Convert to rotation matrix then to quaternion for numerical stability
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    return rotation_matrix_to_quaternion(R)

def quaternion_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """Convert quaternion to Euler angles (ZYX convention).
    
    Args:
        q: Quaternion as [x, y, z, w] (scalar last)
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    # Convert to rotation matrix then to Euler for numerical stability
    R = quaternion_to_rotation_matrix(q)
    return rotation_matrix_to_euler(R)

def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transform 3D points using a transformation matrix.
    
    Args:
        points: Nx3 array of points
        transform: 4x4 transformation matrix
        
    Returns:
        Nx3 array of transformed points
    """
    # Add homogeneous coordinate (w=1)
    points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
    
    # Transform points
    points_transformed_hom = (transform @ points_hom.T).T
    
    # Back to 3D coordinates (divide by w)
    points_transformed = points_transformed_hom[:, :3] / points_transformed_hom[:, 3:4]
    
    return points_transformed

def create_transformation_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Create a 4x4 transformation matrix from rotation matrix and translation vector.
    
    Args:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        
    Returns:
        4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

def invert_transformation(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 transformation matrix.
    
    Args:
        T: 4x4 transformation matrix
        
    Returns:
        Inverted 4x4 transformation matrix
    """
    R = T[:3, :3]
    t = T[:3, 3]
    
    T_inv = np.eye(4)
    R_inv = R.T
    t_inv = -R_inv @ t
    
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    
    return T_inv

def gps_to_enu(lat: float, lon: float, alt: float, ref_lat: float, ref_lon: float, ref_alt: float) -> np.ndarray:
    """Convert GPS coordinates (WGS84) to local ENU (East-North-Up) coordinates.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude in meters (above reference ellipsoid)
        ref_lat: Reference latitude in degrees
        ref_lon: Reference longitude in degrees
        ref_alt: Reference altitude in meters
        
    Returns:
        ENU coordinates [east, north, up] in meters
    """
    # Constants for WGS84 ellipsoid
    a = 6378137.0  # semi-major axis (m)
    f = 1/298.257223563  # flattening
    b = a*(1-f)  # semi-minor axis (m)
    e_sq = 1 - (b*b)/(a*a)  # eccentricity squared
    
    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)
    
    # Compute ECEF coordinates
    # Function to convert lat/lon/alt to ECEF
    def llh_to_ecef(lat_rad: float, lon_rad: float, alt: float) -> np.ndarray:
        N = a / np.sqrt(1 - e_sq * np.sin(lat_rad)**2)
        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - e_sq) + alt) * np.sin(lat_rad)
        return np.array([x, y, z])
    
    # Convert points to ECEF
    ecef = llh_to_ecef(lat_rad, lon_rad, alt)
    ref_ecef = llh_to_ecef(ref_lat_rad, ref_lon_rad, ref_alt)
    
    # Compute ENU transformation matrix
    sin_lat = np.sin(ref_lat_rad)
    cos_lat = np.cos(ref_lat_rad)
    sin_lon = np.sin(ref_lon_rad)
    cos_lon = np.cos(ref_lon_rad)
    
    # Rotation matrix from ECEF to ENU
    R = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat],
        [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]
    ])
    
    # Compute ENU coordinates
    enu = R @ (ecef - ref_ecef)
    
    return enu

def enu_to_gps(east: float, north: float, up: float, ref_lat: float, ref_lon: float, ref_alt: float) -> Tuple[float, float, float]:
    """Convert local ENU (East-North-Up) coordinates to GPS coordinates (WGS84).
    
    Args:
        east: East coordinate in meters
        north: North coordinate in meters
        up: Up coordinate in meters
        ref_lat: Reference latitude in degrees
        ref_lon: Reference longitude in degrees
        ref_alt: Reference altitude in meters
        
    Returns:
        Tuple of (latitude, longitude, altitude) in (degrees, degrees, meters)
    """
    # Constants for WGS84 ellipsoid
    a = 6378137.0  # semi-major axis (m)
    f = 1/298.257223563  # flattening
    b = a*(1-f)  # semi-minor axis (m)
    e_sq = 1 - (b*b)/(a*a)  # eccentricity squared
    
    # Convert to radians
    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)
    
    # Function to convert lat/lon/alt to ECEF
    def llh_to_ecef(lat_rad: float, lon_rad: float, alt: float) -> np.ndarray:
        N = a / np.sqrt(1 - e_sq * np.sin(lat_rad)**2)
        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - e_sq) + alt) * np.sin(lat_rad)
        return np.array([x, y, z])
    
    # Rotation matrix from ENU to ECEF
    sin_lat = np.sin(ref_lat_rad)
    cos_lat = np.cos(ref_lat_rad)
    sin_lon = np.sin(ref_lon_rad)
    cos_lon = np.cos(ref_lon_rad)
    
    R = np.array([
        [-sin_lon, -sin_lat*cos_lon, cos_lat*cos_lon],
        [cos_lon, -sin_lat*sin_lon, cos_lat*sin_lon],
        [0, cos_lat, sin_lat]
    ])
    
    # Compute ECEF coordinates
    ref_ecef = llh_to_ecef(ref_lat_rad, ref_lon_rad, ref_alt)
    ecef = ref_ecef + R @ np.array([east, north, up])
    
    # Convert ECEF to lat/lon/alt
    # This is an iterative process, but we'll use a simplified direct formula
    # for demonstration. For higher precision, use an iterative method.
    
    # Constants
    e_prime_sq = (a*a - b*b) / (b*b)  # second eccentricity squared
    
    # Compute lat/lon/alt
    x, y, z = ecef
    p = np.sqrt(x*x + y*y)
    theta = np.arctan2(z*a, p*b)
    
    lon = np.arctan2(y, x)
    lat = np.arctan2(
        z + e_prime_sq * b * np.sin(theta)**3,
        p - e_sq * a * np.cos(theta)**3
    )
    
    N = a / np.sqrt(1 - e_sq * np.sin(lat)**2)
    alt = p / np.cos(lat) - N
    
    # Convert to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)
    
    return lat, lon, alt 