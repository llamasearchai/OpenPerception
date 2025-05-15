import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import g2o # type: ignore # Add type ignore if g2o stubs are not available
import concurrent.futures

# Updated imports for OpenPerception structure
from openperception.utils.transformations import transformation_matrix_to_pose # Assuming this exists
from openperception.config import SLAMConfig # Assuming SLAMConfig is part of the new config system

logger = logging.getLogger(__name__)

@dataclass
class KeyFrame:
    """Keyframe for visual SLAM"""
    id: int
    timestamp: float
    image: np.ndarray
    pose: np.ndarray  # 4x4 transformation matrix
    keypoints: List[cv2.KeyPoint] # Changed from np.ndarray for consistency with OpenCV
    descriptors: np.ndarray
    depth: Optional[np.ndarray] = None
    
@dataclass
class MapPoint:
    """3D map point for visual SLAM"""
    id: int
    position: np.ndarray  # 3D world coordinates
    descriptor: Optional[np.ndarray] = None # Can be None initially or averaged
    observations: Dict[int, int] = dataclasses.field(default_factory=dict)  # keyframe_id -> keypoint_idx
    color: Optional[np.ndarray] = None # RGB color
    
class Frame:
    """Camera frame for tracking"""
    def __init__(self, 
                 image: np.ndarray, 
                 timestamp: float,
                 detector: cv2.Feature2D, # Pass detector for consistency
                 depth: Optional[np.ndarray] = None):
        self.image = image
        self.timestamp = timestamp
        self.depth = depth
        self.detector = detector
        self.keypoints: Optional[List[cv2.KeyPoint]] = None
        self.descriptors: Optional[np.ndarray] = None
        self.pose = np.eye(4) # Initial pose: world_to_camera
        self.extract_features()
        
    def extract_features(self):
        """Extract features from image using the provided detector."""
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) > 2 and self.image.shape[2] == 3 else self.image
        self.keypoints, self.descriptors = self.detector.detectAndCompute(gray_image, None)
        return self.keypoints, self.descriptors

class VisualSLAM:
    """Visual SLAM system with monocular or RGB-D support."""
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, config: SLAMConfig):
        """Initialize SLAM system.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix.
            dist_coeffs: Distortion coefficients.
            config: SLAM configuration dataclass instance.
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.config = config
        
        logger.info(f"Initializing VisualSLAM with config: {self.config}")

        # Initialize feature detector
        if self.config.feature_detector.upper() == "ORB":
            self.detector = cv2.ORB_create(nfeatures=self.config.max_features)
        elif self.config.feature_detector.upper() == "SIFT":
            try:
                self.detector = cv2.SIFT_create(nfeatures=self.config.max_features)
            except AttributeError: # SIFT might be in xfeatures2d for older OpenCV
                try: 
                    self.detector = cv2.xfeatures2d.SIFT_create(nfeatures=self.config.max_features)
                except AttributeError:
                    logger.error("SIFT detector not available. Please ensure OpenCV contrib modules are installed if using SIFT.")
                    raise
        # Add other detectors like SURF if needed, ensuring they are available
        else:
            logger.warning(f"Unsupported feature detector: {self.config.feature_detector}. Defaulting to ORB.")
            self.detector = cv2.ORB_create(nfeatures=self.config.max_features)
            
        # Feature matcher
        # Use NORM_HAMMING for ORB, NORM_L2 for SIFT/SURF
        norm_type = cv2.NORM_HAMMING if self.config.feature_detector.upper() == "ORB" else cv2.NORM_L2
        self.matcher = cv2.BFMatcher(norm_type, crossCheck=self.config.matcher_cross_check) # Add crossCheck to config
        
        self.keyframes: Dict[int, KeyFrame] = {}
        self.map_points: Dict[int, MapPoint] = {}
        self.current_frame: Optional[Frame] = None
        self.last_frame: Optional[Frame] = None
        self.reference_keyframe: Optional[KeyFrame] = None # For tracking
        
        self.keyframe_counter = 0
        self.map_point_counter = 0
        
        self.tracking_initialized = False
        self.frame_count = 0
        self.last_pose = np.eye(4) # World to current camera pose
        
        # TODO: Implement g2o optimizer setup if bundle adjustment is added
        # self.optimizer = g2o.SparseOptimizer()
        # ... setup optimizer ...
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_mapping_threads) # Configurable threads
        self.mapping_task: Optional[concurrent.futures.Future] = None
        
    def process_frame(self, image: np.ndarray, timestamp: float, 
                      depth: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Process a new camera frame.
        
        Args:
            image: Input camera image (BGR).
            timestamp: Frame timestamp (seconds).
            depth: Optional depth image (for RGB-D SLAM, aligned with image).
            
        Returns:
            4x4 camera pose matrix (world_to_camera) if tracking, else None.
        """
        self.current_frame = Frame(image, timestamp, self.detector, depth)
        
        if not self.current_frame.keypoints or self.current_frame.descriptors is None:
            logger.warning("No features detected in current frame.")
            return self.last_pose # Return last known pose

        if not self.tracking_initialized:
            if self._initialize_tracking():
                self.last_frame = self.current_frame
                self.last_pose = self.current_frame.pose
                logger.info(f"SLAM initialized. First keyframe ID: {self.reference_keyframe.id if self.reference_keyframe else 'None'}")
                return self.current_frame.pose
            else:
                logger.warning("SLAM initialization failed.")
                return None # Cannot initialize
            
        # Track motion relative to the last frame or reference keyframe
        tracked_pose = self._track_motion()
        
        if tracked_pose is None:
            logger.warning("Tracking lost!")
            # TODO: Implement relocalization logic here
            return self.last_pose # Return last known good pose
        
        self.current_frame.pose = tracked_pose
        self.last_pose = tracked_pose
        
        # TODO: Implement local map tracking if using keyframes extensively
        # if self.reference_keyframe:
        #     self._track_local_map()
            
        if self._need_new_keyframe():
            self._create_keyframe_from_current()
            
            # Optional: Trigger mapping in background (if local BA or new map point creation is intensive)
            # if self.config.enable_mapping_thread and (self.frame_count % self.config.mapping_rate == 0):
            #     if self.mapping_task is None or self.mapping_task.done():
            #         # self.mapping_task = self.executor.submit(self._local_bundle_adjustment) # or self._create_new_map_points
            #         pass # Placeholder for actual mapping task
        
        self.last_frame = self.current_frame
        self.frame_count += 1
        
        return self.current_frame.pose

    def _initialize_tracking(self) -> bool:
        """Initialize tracking with the first frame, creating the first keyframe."""
        if self.current_frame and self.current_frame.keypoints and len(self.current_frame.keypoints) > self.config.min_features_for_initialization:
            self.tracking_initialized = True
            # First keyframe is at the origin of the world coordinate system
            self.current_frame.pose = np.eye(4) 
            self._create_keyframe_from_current()
            return True
        logger.warning(f"Not enough features for initialization: {len(self.current_frame.keypoints if self.current_frame and self.current_frame.keypoints else [])}")
        return False
        
    def _track_motion(self) -> Optional[np.ndarray]:
        """Track camera motion from reference_keyframe to current_frame, or last_frame to current_frame."""
        if not self.current_frame or self.current_frame.descriptors is None:
            return None

        # Determine reference for tracking
        ref_keypoints: Optional[List[cv2.KeyPoint]] = None
        ref_descriptors: Optional[np.ndarray] = None
        ref_pose: np.ndarray = np.eye(4)

        if self.reference_keyframe and self.reference_keyframe.descriptors is not None:
            ref_keypoints = self.reference_keyframe.keypoints
            ref_descriptors = self.reference_keyframe.descriptors
            ref_pose = self.reference_keyframe.pose
        elif self.last_frame and self.last_frame.descriptors is not None: # Fallback to last frame if no ref KF
            ref_keypoints = self.last_frame.keypoints
            ref_descriptors = self.last_frame.descriptors
            ref_pose = self.last_frame.pose # This pose is world_to_last_frame
        else:
            return None # Cannot track

        if ref_descriptors is None or ref_keypoints is None:
             return None

        matches = self.matcher.match(ref_descriptors, self.current_frame.descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < self.config.max_descriptor_distance] # Use a config param
        
        if len(good_matches) < self.config.min_matches_for_pose_estimation: # Config param
            logger.debug(f"Not enough good matches for pose estimation: {len(good_matches)}")
            return None

        # Get matched 2D points
        ref_pts = np.float32([ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        curr_pts = np.float32([self.current_frame.keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate Essential matrix if no depth, or use PnP if 3D points are available from reference
        # For simplicity, this example shows a 2D-2D estimation (e.g., for initialization or monocular)
        # In a full system, you'd use PnP with 3D map points observed by the reference_keyframe.

        E, mask = cv2.findEssentialMat(curr_pts, ref_pts, self.camera_matrix, 
                                       method=cv2.RANSAC, prob=0.999, threshold=self.config.ransac_threshold_pixels)
        if E is None or mask is None:
            logger.debug("findEssentialMat failed.")
            return None

        _, R_rel, t_rel, _ = cv2.recoverPose(E, curr_pts, ref_pts, self.camera_matrix, mask=mask)
        
        if R_rel is None or t_rel is None:
            logger.debug("recoverPose failed.")
            return None

        # Transformation from reference frame to current frame (T_current_ref)
        T_current_ref = np.eye(4)
        T_current_ref[:3, :3] = R_rel
        T_current_ref[:3, 3] = t_rel.flatten()
        
        # New pose: T_world_current = T_world_ref @ T_ref_current
        # We have T_current_ref (current relative to ref). We need T_ref_current.
        # T_ref_current is inv(T_current_ref)
        T_ref_current = np.linalg.inv(T_current_ref) 
        new_pose_world_to_current = ref_pose @ T_ref_current
        
        return new_pose_world_to_current

    def _need_new_keyframe(self) -> bool:
        """Determine if a new keyframe should be created."""
        if not self.tracking_initialized or not self.reference_keyframe or not self.current_frame:
            return False # Not ready or no reference

        # Basic criteria: number of frames passed, distance, or feature match quality
        frames_since_last_kf = self.frame_count - (getattr(self.reference_keyframe, 'frame_number', 0) if self.reference_keyframe else 0)

        if frames_since_last_kf > self.config.max_frames_between_keyframes:
            return True
        
        # Distance based criteria (simplified)
        if self.current_frame.pose is not None and self.reference_keyframe.pose is not None:
            pose_diff = np.linalg.inv(self.reference_keyframe.pose) @ self.current_frame.pose
            translation_dist = np.linalg.norm(pose_diff[:3, 3])
            if translation_dist > self.config.keyframe_distance_threshold:
                return True
        
        # TODO: Add more sophisticated criteria (e.g., feature overlap, parallax)
        return False

    def _create_keyframe_from_current(self):
        """Creates a new KeyFrame object from the current_frame state."""
        if not self.current_frame or not self.current_frame.keypoints:
            logger.warning("Attempted to create keyframe from invalid current_frame.")
            return

        kf_id = self.keyframe_counter
        new_keyframe = KeyFrame(
            id=kf_id,
            timestamp=self.current_frame.timestamp,
            image=self.current_frame.image.copy(), # Copy image for storage
            pose=self.current_frame.pose.copy(),
            keypoints=list(self.current_frame.keypoints), # Store as list of KeyPoint objects
            descriptors=self.current_frame.descriptors.copy() if self.current_frame.descriptors is not None else np.array([]),
            depth=self.current_frame.depth.copy() if self.current_frame.depth is not None else None
        )
        setattr(new_keyframe, 'frame_number', self.frame_count) # Track when it was created

        self.keyframes[kf_id] = new_keyframe
        self.reference_keyframe = new_keyframe # Update reference for tracking
        self.keyframe_counter += 1
        logger.info(f"Created KeyFrame ID: {kf_id} at frame {self.frame_count}")

        # TODO: Add this keyframe to the map/graph for optimization
        # TODO: Create new map points from this keyframe if possible (triangulation with previous KFs)

    def get_map_points_and_poses(self) -> Tuple[Optional[np.ndarray], Optional[List[np.ndarray]]]:
        """Returns current 3D map points and keyframe poses.
        Returns:
            Tuple (map_points_xyz, keyframe_poses_matrices)
            map_points_xyz: Nx3 numpy array of map point positions, or None.
            keyframe_poses_matrices: List of 4x4 keyframe pose matrices, or None.
        """
        if not self.map_points and not self.keyframes:
            return None, None

        map_points_xyz_list = [mp.position for mp_id, mp in self.map_points.items() if mp.position is not None]
        map_points_xyz = np.array(map_points_xyz_list) if map_points_xyz_list else None

        keyframe_poses_list = [kf.pose for kf_id, kf in self.keyframes.items() if kf.pose is not None]
        keyframe_poses_matrices = keyframe_poses_list if keyframe_poses_list else None
        
        return map_points_xyz, keyframe_poses_matrices

    # Placeholder for mapping and optimization functions
    def _mapping(self):
        logger.info("Mapping thread started (placeholder)")
        # This would involve: Triangulation of new map points, local bundle adjustment, loop closure detection
        time.sleep(1) # Simulate work
        logger.info("Mapping thread finished (placeholder)")

    def _track_local_map(self):
        # Placeholder: optimize current pose against local map points
        pass

    def _local_bundle_adjustment(self):
        # Placeholder: optimize a local window of keyframes and map points
        pass

    def get_current_trajectory(self) -> List[np.ndarray]:
        """Returns the trajectory of all keyframes created so far."""
        # This is a simplified trajectory from keyframes. 
        # For a dense trajectory, poses of all processed frames would be stored.
        return [kf.pose for kf_id, kf in sorted(self.keyframes.items())]

    def shutdown(self):
        logger.info("Shutting down VisualSLAM system...")
        self.executor.shutdown(wait=True)
        logger.info("ThreadPoolExecutor shut down.") 