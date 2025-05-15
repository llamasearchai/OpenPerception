from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Body
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
import io
import logging
import time
import tempfile
import os
import shutil
import threading
import uvicorn
# Updated imports for OpenPerception structure
from openperception.slam.visual_slam import VisualSLAM 
from openperception.sfm.structure_from_motion import StructureFromMotion
from openperception.config import WebServiceConfig, load_config # Assuming config is handled this way
from openperception import OpenPerception # For accessing main app functionalities

import json

logger = logging.getLogger(__name__)

# Global app instance (replace with dependency injection for better practice in larger apps)
# This needs to be initialized by the main application script or a factory.
# For now, it's None and endpoints will need to handle this gracefully or assume it's set.
OP_APP_INSTANCE = None 

def get_op_app():
    """Dependency to get the OpenPerception app instance."""
    if OP_APP_INSTANCE is None:
        # This is a critical error for a running web service if not handled by specific endpoints
        logger.error("OpenPerception application instance is not initialized.")
        # In a real app, you might raise HTTPException here, but some endpoints might not need it.
    return OP_APP_INSTANCE

app = FastAPI(
    title="OpenPerception API",
    description="API for interacting with the OpenPerception framework.",
    version="0.1.0"
)

# --- Request and Response Models ---
class StatusResponse(BaseModel):
    status: str
    version: str
    active_modules: List[str]

class SLAMVideoRequest(BaseModel):
    video_path: str # Path on the server, or could be adapted for upload
    output_dir: Optional[str] = None

class SfMImageDirRequest(BaseModel):
    image_dir: str # Path on the server
    output_dir: Optional[str] = None

class MissionPlanRequest(BaseModel):
    mission_description: str
    environment_description: str
    constraints: str

class MissionPlanResponse(BaseModel):
    plan: str

class FusedStateResponse(BaseModel):
    timestamp: float
    position: List[float]
    orientation: List[float] # Quaternion [x,y,z,w]
    velocity: List[float]

class CalibrationImageResponse(BaseModel):
    success: bool
    message: str
    image_count: Optional[int] = None

class CalibrationResultResponse(BaseModel):
    camera_matrix: Optional[List[List[float]]] = None
    dist_coeffs: Optional[List[float]] = None
    rms_error: Optional[float] = None # RMS error is not directly in CameraCalibrator.calibrate return, might need to fetch
    message: str

class CameraParams(BaseModel):
    """Camera parameters model"""
    camera_matrix: List[List[float]]
    dist_coeffs: List[float]

class ProcessRequest(BaseModel):
    """Process request model"""
    algorithm: str
    config: Dict[str, Any] = {}
    input_file: Optional[str] = None # e.g., filename from /upload
    input_files: Optional[List[str]] = None # For SfM from multiple images

# Define models for SLAM and SfM requests if they need specific parameters
class SlamProcessRequest(BaseModel):
    video_path: str # Relative to temp_dir or an accessible path
    camera_matrix: List[List[float]]
    dist_coeffs: List[float]

class SfMProcessRequest(BaseModel):
    image_paths: List[str] # Relative to temp_dir or accessible paths
    camera_matrix: List[List[float]]
    dist_coeffs: List[float]

class WebServiceAPI:
    """Web service for OpenPerception using FastAPI.
    
    This class provides the FastAPI app and routes.
    It should be instantiated and run by the main OpenPerception application.
    """
    
    def __init__(self, open_perception_app: OpenPerception, config: WebServiceConfig):
        """Initialize web service.
        
        Args:
            open_perception_app: Instance of the main OpenPerception application.
            config: Web service configuration dataclass.
        """
        self.app = FastAPI(title="OpenPerception API", version="0.1.0")
        self.open_perception_app = open_perception_app
        self.config = config
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.allowed_origins if self.config.enable_cors else ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
        self.temp_dir = tempfile.mkdtemp(prefix="op_ws_")
        logger.info(f"WebService temporary directory created at: {self.temp_dir}")
        
        # Processing state (simplified, consider Celery or other task queue for production)
        self.processing_lock = threading.Lock()
        self.active_processes: Dict[str, Dict[str, Any]] = {}

    def _setup_routes(self):
        """Setup API routes"""
        @self.app.get("/")
        async def root():
            return {"message": "Welcome to OpenPerception API", "version": self.open_perception_app.version}

        @self.app.get("/status")
        async def get_status():
            # Delegate to OpenPerceptionApp for status information
            return self.open_perception_app.get_status()

        @self.app.post("/upload/image")
        async def upload_image(file: UploadFile = File(...)):
            return await self._save_upload_file(file, "image")

        @self.app.post("/upload/video")
        async def upload_video(file: UploadFile = File(...)):
            return await self._save_upload_file(file, "video")

        @self.app.get("/files/{file_type}/{filename}")
        async def get_file(file_type: str, filename: str):
            # Ensure file_type is safe (e.g. 'results', 'uploads')
            safe_subdirs = {"uploads": self.temp_dir, "results": self.open_perception_app.config.general.output_dir}
            if file_type not in safe_subdirs:
                raise HTTPException(status_code=400, detail="Invalid file type specified.")
            
            base_path = safe_subdirs[file_type]
            file_path = os.path.join(base_path, filename) 
            
            if not os.path.normpath(file_path).startswith(os.path.normpath(base_path)):
                 raise HTTPException(status_code=403, detail="File path traversal attempt detected.")

            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"File '{filename}' not found in '{file_type}'.")
            return FileResponse(file_path)

        # --- SLAM Endpoints ---
        @self.app.post("/slam/run_on_video", summary="Run SLAM on an uploaded video file")
        async def run_slam_video_endpoint(background_tasks: BackgroundTasks, video_filename: str = Body(...)):
            video_path = os.path.join(self.temp_dir, video_filename)
            if not os.path.exists(video_path):
                raise HTTPException(status_code=404, detail=f"Uploaded video '{video_filename}' not found.")
            
            process_id = f"slam_{time.time_ns()}"
            background_tasks.add_task(self._run_slam_task, process_id, video_path)
            self.active_processes[process_id] = {"status": "starting", "task": "slam_video"}
            return {"process_id": process_id, "message": "SLAM processing started."}

        # --- SfM Endpoints ---
        @self.app.post("/sfm/run_on_images", summary="Run SfM on uploaded image files")
        async def run_sfm_images_endpoint(background_tasks: BackgroundTasks, image_filenames: List[str] = Body(...)):
            image_paths = []
            for fname in image_filenames:
                path = os.path.join(self.temp_dir, fname)
                if not os.path.exists(path):
                    raise HTTPException(status_code=404, detail=f"Uploaded image '{fname}' not found.")
                image_paths.append(path)
            
            if not image_paths:
                raise HTTPException(status_code=400, detail="No image files provided for SfM.")
            
            process_id = f"sfm_{time.time_ns()}"
            background_tasks.add_task(self._run_sfm_task, process_id, image_paths)
            self.active_processes[process_id] = {"status": "starting", "task": "sfm_images"}
            return {"process_id": process_id, "message": "SfM processing started."}

        # --- Mission Planning Endpoints ---
        @self.app.post("/mission/plan", summary="Create a mission plan")
        async def create_mission_plan_endpoint(mission_description: str = Body(...), 
                                               environment_description: str = Body(...), 
                                               constraints: str = Body(...)):
            try:
                plan = self.open_perception_app.create_mission_plan(
                    mission_description=mission_description,
                    environment_description=environment_description,
                    constraints=constraints
                )
                return {"mission_plan": plan}
            except Exception as e:
                logger.error(f"Error creating mission plan via API: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to create mission plan: {str(e)}")
        
        @self.app.get("/process/{process_id}/status")
        async def get_process_status(process_id: str):
            if process_id not in self.active_processes:
                raise HTTPException(status_code=404, detail="Process not found")
            return self.active_processes[process_id]

        # Placeholder for more direct control if needed, e.g., for real-time frame processing
        # self.slam_instances = {} # Could be used if managing SLAM state per client session

    async def _save_upload_file(self, file: UploadFile, file_type: str) -> Dict[str, str]:
        """Helper to save uploaded image or video file."""
        # Basic validation for file type (can be enhanced with python-magic)
        allowed_image_types = ["image/jpeg", "image/png"]
        allowed_video_types = ["video/mp4", "video/quicktime", "video/x-msvideo"]

        if file_type == "image" and file.content_type not in allowed_image_types:
            raise HTTPException(status_code=400, detail=f"Invalid image file type: {file.content_type}. Allowed: {allowed_image_types}")
        if file_type == "video" and file.content_type not in allowed_video_types:
             raise HTTPException(status_code=400, detail=f"Invalid video file type: {file.content_type}. Allowed: {allowed_video_types}")

        try:
            filename = f"{time.time_ns()}_{file.filename}"
            file_path = os.path.join(self.temp_dir, filename)
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
            logger.info(f"Uploaded {file_type} '{file.filename}' saved as '{filename}' in {self.temp_dir}")
            return {"filename": filename, "message": f"{file_type.capitalize()} uploaded successfully."}
        except Exception as e:
            logger.error(f"Error uploading file: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")

    def _run_slam_task(self, process_id: str, video_path: str):
        self.active_processes[process_id]["status"] = "running_slam"
        logger.info(f"Starting SLAM task {process_id} for video: {video_path}")
        try:
            # Use the main OpenPerception app to run SLAM
            # This assumes run_slam_from_video saves results and returns info
            slam_results_info = self.open_perception_app.run_slam_from_video(video_path)
            self.active_processes[process_id].update({
                "status": "completed", 
                "result": slam_results_info if slam_results_info else "SLAM finished, no specific result path returned.",
                "output_path": slam_results_info.get("output_dir") if isinstance(slam_results_info, dict) else None
            })
            logger.info(f"SLAM task {process_id} completed. Results: {slam_results_info}")
        except Exception as e:
            logger.error(f"Error in SLAM task {process_id}: {e}", exc_info=True)
            self.active_processes[process_id]["status"] = "failed"
            self.active_processes[process_id]["error"] = str(e)

    def _run_sfm_task(self, process_id: str, image_paths: List[str]):
        self.active_processes[process_id]["status"] = "running_sfm"
        logger.info(f"Starting SfM task {process_id} for images: {image_paths}")
        try:
            # Use the main OpenPerception app to run SfM
            sfm_results_info = self.open_perception_app.run_sfm_from_images_paths(image_paths) # Assuming this method exists
            self.active_processes[process_id].update({
                "status": "completed", 
                "result": sfm_results_info if sfm_results_info else "SfM finished, no specific result path returned.",
                "output_path": sfm_results_info.get("output_dir") if isinstance(sfm_results_info, dict) else None
            })
            logger.info(f"SfM task {process_id} completed. Results: {sfm_results_info}")
        except Exception as e:
            logger.error(f"Error in SfM task {process_id}: {e}", exc_info=True)
            self.active_processes[process_id]["status"] = "failed"
            self.active_processes[process_id]["error"] = str(e)

    def run_server(self):
        """Run the FastAPI server using uvicorn."""
        logger.info(f"Starting WebService on {self.config.host}:{self.config.port}")
        uvicorn.run(self.app, host=self.config.host, port=self.config.port, log_level=self.config.log_level.lower())

    def cleanup(self):
        """Clean up temporary files."""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up WebService temp dir: {e}", exc_info=True)

# --- API Endpoints ---

@app.get("/api/v1/status", response_model=StatusResponse)
async def get_status():
    """Get the current status of the OpenPerception system."""
    op_app = get_op_app()
    active_modules = []
    if op_app:
        if op_app.slam: active_modules.append("SLAM")
        if op_app.sfm: active_modules.append("SfM")
        if op_app.sensor_fusion: active_modules.append("SensorFusion")
        if op_app.mission_planner: active_modules.append("MissionPlanner")
        # Add other modules as needed
        version = getattr(op_app, '__version__', 'unknown') # Assuming main app has version
    else:
        version = 'unknown'
        # Can't determine active modules if app isn't up

    return {
        "status": "running" if op_app else "error_app_not_initialized", 
        "version": version,
        "active_modules": active_modules
    }

# --- SLAM Endpoints ---
@app.post("/api/v1/slam/run_video")
async def run_slam_video(request: SLAMVideoRequest):
    """Run SLAM processing on a video file (server-side path)."""
    op_app = get_op_app()
    if not op_app or not op_app.slam:
        raise HTTPException(status_code=503, detail="SLAM module not available or app not initialized.")
    try:
        # This is a blocking call, consider running in a threadpool for async
        op_app.run_slam_from_video(request.video_path, request.output_dir)
        return {"message": f"SLAM processing started for {request.video_path}. Results will be in {request.output_dir or op_app.config['general']['output_dir'] + '/slam'}"}
    except Exception as e:
        logger.error(f"SLAM run_video error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- SfM Endpoints ---
@app.post("/api/v1/sfm/run_images")
async def run_sfm_images(request: SfMImageDirRequest):
    """Run Structure from Motion on a directory of images (server-side path)."""
    op_app = get_op_app()
    if not op_app or not op_app.sfm:
        raise HTTPException(status_code=503, detail="SfM module not available or app not initialized.")
    try:
        # This is a blocking call
        op_app.run_sfm_from_images(request.image_dir, request.output_dir)
        return {"message": f"SfM processing started for {request.image_dir}. Results will be in {request.output_dir or op_app.config['general']['output_dir'] + '/sfm'}"}
    except Exception as e:
        logger.error(f"SfM run_images error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Mission Planner Endpoints ---
@app.post("/api/v1/mission_planner/create_plan", response_model=MissionPlanResponse)
async def create_mission_plan(request: MissionPlanRequest):
    """Create a mission plan using the AI mission planner."""
    op_app = get_op_app()
    if not op_app or not op_app.mission_planner:
        raise HTTPException(status_code=503, detail="Mission Planner module not available or app not initialized.")
    try:
        plan = op_app.mission_planner.create_mission_plan(
            request.mission_description,
            request.environment_description,
            request.constraints
        )
        return {"plan": plan}
    except Exception as e:
        logger.error(f"Mission plan creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Sensor Fusion Endpoints ---
@app.get("/api/v1/sensor_fusion/latest_state", response_model=FusedStateResponse)
async def get_latest_fused_state():
    """Get the latest fused state from the sensor fusion module."""
    op_app = get_op_app()
    if not op_app or not op_app.sensor_fusion:
        raise HTTPException(status_code=503, detail="Sensor Fusion module not available or app not initialized.")
    try:
        state = op_app.sensor_fusion.get_fused_state()
        return {
            "timestamp": state['timestamp'],
            "position": state['position'].tolist() if isinstance(state['position'], np.ndarray) else state['position'],
            "orientation": state['orientation'].tolist() if isinstance(state['orientation'], np.ndarray) else state['orientation'],
            "velocity": state['velocity'].tolist() if isinstance(state['velocity'], np.ndarray) else state['velocity'],
        }
    except Exception as e:
        logger.error(f"Get fused state error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Calibration Endpoints ---
@app.post("/api/v1/calibration/add_image", response_model=CalibrationImageResponse)
async def add_calibration_image(file: UploadFile = File(...)):
    """Add an image for camera calibration."""
    op_app = get_op_app()
    if not op_app or not hasattr(op_app, 'camera_calibrator'): # Assuming CameraCalibrator is an attribute
        # Or if CameraCalibrator is part of a specific sub-module:
        # if not op_app or not op_app.calibration_module or not op_app.calibration_module.calibrator:
        raise HTTPException(status_code=503, detail="Camera Calibrator not available or app not initialized.")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
            
        # Assuming the main app has a CameraCalibrator instance, e.g., op_app.camera_calibrator
        # This needs to be initialized in the OpenPerception class setup.
        # For now, let's assume it exists at op_app.camera_calibrator
        if not hasattr(op_app, 'camera_calibrator') or op_app.camera_calibrator is None:
            # Initialize it on the fly if not present (example, might need proper config)
            from openperception.calibration.camera_calibration import CameraCalibrator # Lazy import
            op_app.camera_calibrator = CameraCalibrator() # Using default params
            logger.info("Dynamically initialized CameraCalibrator for endpoint.")

        success = op_app.camera_calibrator.add_image(img)
        msg = "Chessboard detected and image added." if success else "Chessboard not detected."
        return {"success": success, "message": msg, "image_count": len(op_app.camera_calibrator.imgpoints)}
    except Exception as e:
        logger.error(f"Add calibration image error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/calibration/calibrate", response_model=CalibrationResultResponse)
async def perform_camera_calibration():
    """Perform camera calibration based on added images."""
    op_app = get_op_app()
    if not op_app or not hasattr(op_app, 'camera_calibrator') or op_app.camera_calibrator is None:
        raise HTTPException(status_code=503, detail="Camera Calibrator not available or not initialized.")
    
    calibrator = op_app.camera_calibrator
    if len(calibrator.imgpoints) < op_app.config.get("calibration", {}).get("min_images_for_calibration", 3):
         raise HTTPException(status_code=400, detail=f"Not enough images for calibration. Need at least {op_app.config.get('calibration', {}).get('min_images_for_calibration', 3)}. Got {len(calibrator.imgpoints)}.")

    try:
        cam_matrix, dist_coeffs = calibrator.calibrate() # RMS is logged by calibrate method, not returned
        # To get RMS, it would need to be stored on the calibrator instance by its calibrate() method
        # rms_error = getattr(calibrator, 'last_rms_error', None) 
        return {
            "camera_matrix": cam_matrix.tolist() if cam_matrix is not None else None,
            "dist_coeffs": dist_coeffs.tolist() if dist_coeffs is not None else None,
            # "rms_error": rms_error,
            "message": "Calibration successful."
        }
    except ValueError as ve:
        logger.warning(f"Calibration failed: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Calibration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/calibration/get_results", response_model=CalibrationResultResponse)
async def get_calibration_results():
    """Get the current camera calibration results."""
    op_app = get_op_app()
    if not op_app or not hasattr(op_app, 'camera_calibrator') or op_app.camera_calibrator is None:
        raise HTTPException(status_code=503, detail="Camera Calibrator not available or not initialized.")

    calibrator = op_app.camera_calibrator
    if calibrator.camera_matrix is None or calibrator.dist_coeffs is None:
        return {"message": "Camera not calibrated yet."}
    
    return {
        "camera_matrix": calibrator.camera_matrix.tolist(),
        "dist_coeffs": calibrator.dist_coeffs.tolist(),
        "message": "Current calibration results."
    }

# --- Main application setup for Uvicorn (example) ---
# This part is more for running this file directly, 
# the `app` object will be imported by the main OpenPerception class usually.

# To set the global app instance for testing this file directly:
# if __name__ == "__main__":
#     import uvicorn
#     from openperception.main import OpenPerception # Adjust if main structure changes
#     from openperception.config import get_config
    
#     # Create a dummy OpenPerception app instance for the web service to use
#     # This is a simplified setup for standalone testing of api.py.
#     # In the full app, OpenPerception.__init__ would set OP_APP_INSTANCE.
#     global OP_APP_INSTANCE
#     config = get_config() # Load default or file-based config
#     OP_APP_INSTANCE = OpenPerception(config_path=None) # Or provide a path to a test config
#     print(f"OpenPerception app instance created for FastAPI: {OP_APP_INSTANCE is not None}")
#     if OP_APP_INSTANCE:
#          print(f"SLAM enabled in config: {OP_APP_INSTANCE.config['slam']['enabled']}")
#          print(f"SLAM module loaded: {OP_APP_INSTANCE.slam is not None}")
#          # Initialize calibrator if not done by OpenPerception __init__ specifically for web
#          if not hasattr(OP_APP_INSTANCE, 'camera_calibrator'):
#             from openperception.calibration.camera_calibration import CameraCalibrator
#             OP_APP_INSTANCE.camera_calibrator = CameraCalibrator()
#             print("Camera calibrator attached to app instance.")

#     uvicorn.run(app, host=config.web_service.host, port=config.web_service.port)

# Function to be called by OpenPerception main class to set the app instance
def set_openperception_app_instance(instance):
    global OP_APP_INSTANCE
    OP_APP_INSTANCE = instance
    logger.info(f"OpenPerception FastAPI app linked with main instance: {OP_APP_INSTANCE is not None}")
    # You might want to initialize web-specific attributes on the instance here
    # e.g., ensure camera_calibrator is ready if it's primarily a web-facing feature
    if OP_APP_INSTANCE and not hasattr(OP_APP_INSTANCE, 'camera_calibrator'):
        from openperception.calibration.camera_calibration import CameraCalibrator
        # Consider if Calibrator needs specific config from OP_APP_INSTANCE.config
        calib_config = OP_APP_INSTANCE.config.get("calibration", {})
        OP_APP_INSTANCE.camera_calibrator = CameraCalibrator(
            chessboard_size=tuple(calib_config.get("chessboard_size", (9,6))),
            square_size=calib_config.get("square_size", 0.025)
        )
        logger.info("Camera calibrator initialized and attached to OpenPerception app instance for web service.") 