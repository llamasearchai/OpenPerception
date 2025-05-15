from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# from langchain.llms import OpenAI # Older import
from langchain_community.llms import OpenAI # Updated import for newer langchain
# from openai import OpenAI as OpenAIClient # Older import
from openai import OpenAI as OpenAIClient # Keep as is, this is the direct OpenAI SDK client

import logging
import os
import json
from typing import Dict, Any, List, Optional
import time
from pathlib import Path

# Assuming your config system will be available via a standard path
# from openperception.config import get_config # This will be the ideal way

logger = logging.getLogger(__name__)

ASSISTANT_ID_FILE = "assistant_id.json"

class MissionPlanner:
    """Mission planner using LangChain and OpenAI Assistants API"""
    
    def __init__(self, openai_api_key: Optional[str] = None, config_dir: Path = Path(".")):
        """Initialize mission planner
        
        Args:
            openai_api_key: OpenAI API key. If None, will use environment variable or config.
            config_dir: Directory to store assistant_id.json. Defaults to current dir.
                      In a packaged app, this should be a user-specific config/data directory.
        """
        self.config_dir = Path(config_dir)
        self.assistant_id_path = self.config_dir / ASSISTANT_ID_FILE
        self.config_dir.mkdir(parents=True, exist_ok=True) # Ensure config_dir exists

        # Attempt to get API key from arg, then env, then config (if integrated)
        if openai_api_key:
            self.openai_api_key = openai_api_key
        else:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
            # Example of how you might integrate with your new config system:
            # if not self.openai_api_key:
            #     try:
            #         app_config = get_config()
            #         self.openai_api_key = app_config.mission_planner.openai_api_key
            #     except Exception as e:
            #         logger.warning(f"Could not load API key from global config: {e}")

        if not self.openai_api_key:
            logger.warning("No OpenAI API key provided. Mission planning functionality will be limited.")
            self.openai_enabled = False
            self.llm = None
            self.mission_chain = None
            self.client = None
            self.assistant_id = None
        else:
            self.openai_enabled = True
            try:
                self.llm = OpenAI(temperature=0, openai_api_key=self.openai_api_key)
                
                self.mission_prompt = PromptTemplate(
                    input_variables=["mission_description", "environment_description", "constraints"],
                    template="""
                    You are an advanced mission planning system for aerial perception tasks.
                    
                    Mission Description: {mission_description}
                    Environment Description: {environment_description}
                    Constraints: {constraints}
                    
                    Create a detailed mission plan that includes:
                    1. Required sensors and their configurations
                    2. Flight path planning (waypoints, altitude, speed)
                    3. Data collection strategy
                    4. Processing pipeline configuration
                    5. Safety considerations and fallback procedures
                    
                    Format your response as a well-structured plan that can be followed by an autonomous drone system.
                    """
                )
                self.mission_chain = LLMChain(llm=self.llm, prompt=self.mission_prompt)
                self.client = OpenAIClient(api_key=self.openai_api_key)
                self.assistant_id = self._create_or_load_assistant()
            except ImportError as e:
                logger.error(f"Failed to import LangChain or OpenAI components. Ensure they are installed: {e}")
                self.openai_enabled = False
            except Exception as e:
                logger.error(f"Error initializing OpenAI components: {e}")
                self.openai_enabled = False
            
    def _create_or_load_assistant(self) -> Optional[str]:
        """Create a new OpenAI Assistant or load existing one from self.assistant_id_path
        
        Returns:
            Assistant ID or None if creation/loading failed.
        """
        if not self.openai_enabled or not self.client:
            return None

        if self.assistant_id_path.exists():
            try:
                with open(self.assistant_id_path, "r") as f:
                    data = json.load(f)
                    assistant_id = data.get("assistant_id")
                    if assistant_id:
                        # Verify assistant still exists
                        try:
                            self.client.beta.assistants.retrieve(assistant_id=assistant_id)
                            logger.info(f"Loaded existing assistant ID: {assistant_id} from {self.assistant_id_path}")
                            return assistant_id
                        except Exception as e:
                            logger.info(f"Stored assistant ID {assistant_id} not found or invalid, creating new one: {e}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error reading assistant ID file {self.assistant_id_path}: {e}")                
                    
        # Create new assistant
        try:
            assistant = self.client.beta.assistants.create(
                name="Aerial Perception Mission Planner",
                instructions="""
                You are an advanced mission planning assistant for aerial perception tasks.
                Your expertise includes computer vision, drone operations, path planning, sensor fusion,
                and 3D reconstruction. Help users plan aerial perception missions by providing
                detailed guidance on flight planning, sensor selection, data collection strategies,
                and processing pipeline configuration.
                """,
                model="gpt-4-turbo-preview" # Consider making model configurable
            )
            
            # Store assistant ID
            with open(self.assistant_id_path, "w") as f:
                json.dump({"assistant_id": assistant.id}, f)
            logger.info(f"Created new assistant and saved ID {assistant.id} to {self.assistant_id_path}")
            return assistant.id
        except Exception as e:
            logger.error(f"Failed to create OpenAI assistant: {e}")
            return None
        
    def create_mission_plan(self, mission_description: str, environment_description: str, 
                           constraints: str) -> Optional[str]:
        """Create a mission plan using LangChain
        
        Args:
            mission_description: Description of the mission
            environment_description: Description of the environment
            constraints: Constraints to consider
            
        Returns:
            Mission plan as a string, or None if not enabled or error occurs.
        """
        if not self.openai_enabled or not self.mission_chain:
            logger.warning("OpenAI mission planning is not enabled or not initialized.")
            return "Mission planning requires OpenAI API key and successful initialization."
        
        try:
            response = self.mission_chain.run(
                mission_description=mission_description,
                environment_description=environment_description,
                constraints=constraints
            )
            return response
        except Exception as e:
            logger.error(f"Error during LangChain mission plan creation: {e}")
            return None
        
    def analyze_mission_data(self, data_description: str, mission_context: str,
                            specific_questions: List[str]) -> Dict[str, Any]:
        """Analyze mission data using Assistants API
        
        Args:
            data_description: Description of the collected data
            mission_context: Context of the mission
            specific_questions: Specific questions to answer
            
        Returns:
            Analysis results or an error dictionary.
        """
        if not self.openai_enabled or not self.client or not self.assistant_id:
            logger.warning("OpenAI data analysis is not enabled or not initialized.")
            return {"error": "Mission data analysis requires OpenAI API key and assistant initialization."}
            
        try:
            # Create thread
            thread = self.client.beta.threads.create()
            
            message_content = f"""
            Mission Context: {mission_context}
            
            Data Description: {data_description}
            
            Please analyze this data and answer the following questions:
            {chr(10).join([f"- {q}" for q in specific_questions])}
            
            Provide a detailed analysis with actionable insights.
            """
            
            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=message_content
            )
            
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )
            
            # Wait for completion (with a timeout)
            start_time = time.time()
            timeout_seconds = 120 # 2 minutes timeout
            while time.time() - start_time < timeout_seconds:
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                if run_status.status == "completed":
                    break
                elif run_status.status in ["requires_action", "failed", "cancelled", "expired"]:
                    logger.error(f"Assistant run failed or requires action. Status: {run_status.status}")
                    return {"error": f"Assistant run ended with status: {run_status.status}"}
                time.sleep(1)
            else:
                logger.error("Assistant run timed out.")
                return {"error": "Assistant run timed out."}
                
            messages = self.client.beta.threads.messages.list(thread_id=thread.id)
            
            response = ""
            for message in messages.data:
                if message.role == "assistant":
                    for content_block in message.content:
                        if content_block.type == "text":
                            response += content_block.text.value + "\n"
                            
            return {"analysis": response.strip()} if response else {"error": "No response from assistant."}
        except Exception as e:
            logger.error(f"Error during mission data analysis with Assistant API: {e}")
            return {"error": f"An exception occurred: {str(e)}"}

# Example of how to use it, assuming config is handled externally:
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Create a dummy config dir for the example
    example_config_dir = Path("temp_planner_config")
    example_config_dir.mkdir(exist_ok=True)

    # Test with API key from environment variable "OPENAI_API_KEY"
    planner = MissionPlanner(config_dir=example_config_dir) 

    if planner.openai_enabled:
        print(f"Planner initialized. Assistant ID: {planner.assistant_id}")
        
        # Test LangChain part (Simpler call)
        # plan = planner.create_mission_plan(
        #     mission_description="Survey a 50x50m agricultural field for crop health.",
        #     environment_description="Open field, sunny conditions, light wind.",
        #     constraints="Maintain altitude of 30m, complete within 15 minutes."
        # )
        # if plan:
        #     print("\nGenerated Mission Plan (LangChain):")
        #     print(plan)
        # else:
        #     print("\nFailed to generate mission plan using LangChain.")

        # Test Assistant API part
        analysis_result = planner.analyze_mission_data(
            data_description="Collected 10GB of multispectral imagery and 2GB of LiDAR point cloud data.",
            mission_context="Post-flight analysis of agricultural field survey.",
            specific_questions=[
                "Identify areas of potential water stress.",
                "Estimate current biomass density.",
                "Are there any anomalies in the LiDAR data?"
            ]
        )
        print("\nMission Data Analysis (Assistant API):")
        if "error" in analysis_result:
            print(f"Error: {analysis_result['error']}")
        else:
            print(analysis_result.get('analysis', "No analysis content found."))
        
        # Clean up dummy config dir
        # (Path(example_config_dir) / ASSISTANT_ID_FILE).unlink(missing_ok=True)
        # example_config_dir.rmdir()
    else:
        print("Mission planner could not be initialized due to missing API key or other error.") 