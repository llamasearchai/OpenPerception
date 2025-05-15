from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
#LLM imports will depend on the chosen llm, e.g. from langchain.llms import OpenAI or from langchain_community.llms import Ollama etc
#For now, let's assume OpenAI is used, but this might need to be made more flexible or configurable
from langchain_openai import OpenAI # Corrected import for newer Langchain

from openai import OpenAI as OpenAIClient # Keep this for direct OpenAI API access if needed for Assistants
import logging
import os
import json
from typing import Dict, Any, List, Optional
import time

logger = logging.getLogger(__name__)

class MissionPlanner:
    """Mission planner using LangChain and OpenAI Assistants API"""
    
    def __init__(self, openai_api_key: Optional[str] = None, planning_algorithm: str = "rrt_star", replanning_interval: int = 5):
        """Initialize mission planner
        
        Args:
            openai_api_key: OpenAI API key. If None, will use environment variable.
            planning_algorithm: The algorithm to use for path planning (e.g., 'rrt_star').
            replanning_interval: Interval in seconds for replanning.
        """
        # Get API key from environment if not provided
        self.planning_algorithm = planning_algorithm #Store planning_algorithm
        self.replanning_interval = replanning_interval #Store replanning_interval

        if openai_api_key is None:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            
        if not openai_api_key:
            logger.warning("No OpenAI API key provided. OpenAI-based mission planning functionality will be limited.")
            self.openai_enabled = False
        else:
            self.openai_enabled = True
            
        # Initialize LangChain
        if self.openai_enabled:
            try:
                self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
                
                # Create mission planning prompt
                self.mission_prompt = PromptTemplate(
                    input_variables=["mission_description", "environment_description", "constraints"],
                    template="""
                    You are an advanced mission planning system for aerial perception tasks.
                    
                    Mission Description: {mission_description}
                    Environment Description: {environment_description}
                    Constraints: {constraints}
                    
                    Create a detailed mission plan that includes:
                    1. Required sensors and their configurations
                    2. Flight path planning (waypoints, altitude, speed) based on {planning_algorithm} principles if applicable.
                    3. Data collection strategy
                    4. Processing pipeline configuration
                    5. Safety considerations and fallback procedures
                    
                    Format your response as a well-structured plan that can be followed by an autonomous drone system.
                    """
                )
                
                # Create mission planning chain
                self.mission_chain = LLMChain(llm=self.llm, prompt=self.mission_prompt)
                
                # Initialize OpenAI client for Assistants API (if direct assistant use is still intended)
                self.client = OpenAIClient(api_key=openai_api_key)
                # Create or load assistant (optional, can be removed if Langchain is sufficient)
                # self.assistant_id = self._create_or_load_assistant() 
                self.assistant_id = None # Initialize, can be set if assistant is used

            except Exception as e:
                logger.error(f"Failed to initialize LangChain or OpenAI client: {e}")
                self.openai_enabled = False
        else:
            self.llm = None
            self.mission_chain = None
            self.client = None
            self.assistant_id = None
            
    def _create_or_load_assistant(self) -> Optional[str]:
        """Create a new OpenAI Assistant or load existing one
        
        Returns:
            Assistant ID or None if not enabled/failed
        """
        if not self.openai_enabled or not self.client:
            return None
        
        assistant_file = "assistant_id.json"
        # Check if we have a stored assistant ID
        if os.path.exists(assistant_file):
            try:
                with open(assistant_file, "r") as f:
                    data = json.load(f)
                    assistant_id = data.get("assistant_id")
                    if assistant_id:
                        # Verify assistant still exists
                        try:
                            self.client.beta.assistants.retrieve(assistant_id)
                            logger.info(f"Loaded existing assistant: {assistant_id}")
                            return assistant_id
                        except Exception as e:
                            logger.info(f"Stored assistant {assistant_id} not found or error: {e}. Creating new one.")
            except Exception as e:
                logger.error(f"Error loading assistant_id.json: {e}")

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
                model="gpt-4-turbo-preview" # Or a newer/preferred model
            )
            
            # Store assistant ID
            with open(assistant_file, "w") as f:
                json.dump({"assistant_id": assistant.id}, f)
            logger.info(f"Created new assistant: {assistant.id}")
            return assistant.id
        except Exception as e:
            logger.error(f"Failed to create OpenAI assistant: {e}")
            return None
        
    def create_mission_plan(self, mission_description: str, environment_description: str, 
                           constraints: str) -> str:
        """Create a mission plan using LangChain
        
        Args:
            mission_description: Description of the mission
            environment_description: Description of the environment
            constraints: Constraints to consider
            
        Returns:
            Mission plan as a string, or an error message.
        """
        if not self.openai_enabled or not self.mission_chain:
            return "Mission planning with OpenAI is not enabled or initialized."
        
        try:
            # Include planning_algorithm in the call to the chain if it's in the prompt
            response = self.mission_chain.run(
                mission_description=mission_description,
                environment_description=environment_description,
                constraints=constraints,
                planning_algorithm=self.planning_algorithm # Pass it to the prompt
            )
            return response
        except Exception as e:
            logger.error(f"Error during LangChain mission plan generation: {e}")
            return f"Error generating mission plan: {e}"
        
    def analyze_mission_data_with_assistant(self, data_description: str, mission_context: str,
                                            specific_questions: List[str]) -> Dict[str, Any]:
        """Analyze mission data using a dedicated OpenAI Assistant (if configured).
        This requires self.assistant_id to be set, e.g., by calling _create_or_load_assistant.
        Args:
            data_description: Description of the collected data
            mission_context: Context of the mission
            specific_questions: Specific questions to answer
            
        Returns:
            Analysis results as a dictionary.
        """
        if not self.openai_enabled or not self.client or not self.assistant_id:
            return {"error": "OpenAI Assistant for mission data analysis is not enabled or initialized."}

        try:
            # Create thread
            thread = self.client.beta.threads.create()
            
            # Add message
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
            
            # Run the assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )
            
            # Wait for completion
            start_time = time.time()
            timeout_seconds = 120 # 2 minutes timeout for the assistant run
            while True:
                if time.time() - start_time > timeout_seconds:
                    logger.error(f"Assistant run timed out after {timeout_seconds} seconds.")
                    return {"error": "Assistant run timed out."}
                
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                
                if run_status.status == "completed":
                    break
                elif run_status.status in ["requires_action"]:
                    # Handle requires_action if you have tools defined for the assistant
                    logger.info(f"Assistant requires action: {run_status.required_action}")
                    # For now, we'll just break and report, as no tools are defined here.
                    # In a real scenario, you would submit tool outputs.
                    return {"error": "Assistant requires action, but no tools are handled in this basic setup."}
                elif run_status.status in ["failed", "cancelled", "expired"]:
                    logger.error(f"Assistant run failed with status: {run_status.status}")
                    return {"error": f"Assistant run failed: {run_status.status}"}
                    
                time.sleep(1)
                
            # Get response messages
            messages = self.client.beta.threads.messages.list(thread_id=thread.id)
            
            # Extract assistant's response
            assistant_response = ""
            for message in reversed(messages.data): # Iterate from oldest to newest relevant messages
                if message.run_id == run.id and message.role == "assistant":
                    for content_block in message.content:
                        if content_block.type == "text":
                            assistant_response += content_block.text.value + "\n"
            
            if not assistant_response:
                 return {"analysis": "Assistant provided no textual response for this run."}

            return {"analysis": assistant_response.strip()}

        except Exception as e:
            logger.error(f"Error during OpenAI Assistant mission data analysis: {e}")
            return {"error": f"Failed to analyze mission data with Assistant: {e}"}

    def get_available_planning_algorithms(self) -> List[str]:
        """Returns a list of available planning algorithms (placeholder)."""
        # In a real system, this might come from registered plugins or a config file.
        return ["rrt_star", "rrt", "a_star", "simple_waypoint_follower"]

    def set_planning_algorithm(self, algorithm: str) -> bool:
        """Sets the planning algorithm if it's valid."""
        if algorithm in self.get_available_planning_algorithms():
            self.planning_algorithm = algorithm
            # Update the prompt if the chain is already created and planning_algorithm is part of it
            if self.openai_enabled and hasattr(self, 'mission_prompt') and self.llm:
                 self.mission_prompt.template = self.mission_prompt.template.replace(
                     # This is a bit naive, assumes the old value is known or can be templated out
                     # A better way would be to re-initialize the prompt or have a placeholder in the template
                     # that is always filled.
                     f"based on {{planning_algorithm}} principles if applicable.",
                     f"based on {self.planning_algorithm} principles if applicable."
                 )
                 # Recreate the chain if necessary, or ensure the prompt update is reflected.
                 # For simplicity, we assume `run` will use the current `self.planning_algorithm` as a variable.
            logger.info(f"Planning algorithm set to: {self.planning_algorithm}")
            return True
        logger.warning(f"Invalid planning algorithm: {algorithm}")
        return False 