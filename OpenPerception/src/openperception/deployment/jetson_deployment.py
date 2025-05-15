import os
import subprocess
import logging
import argparse
import yaml
import shutil
from pathlib import Path

# Updated imports (config will be passed to constructor or loaded via new system)
from openperception.config import DeploymentConfig # Assuming DeploymentConfig exists or will be created
                                                # Or load the full config and access deployment section

logger = logging.getLogger(__name__)

class JetsonDeployment:
    """Deployment tools for NVIDIA Jetson platforms for the OpenPerception project."""
    
    def __init__(self, config: Optional[DeploymentConfig] = None, project_root_dir: Optional[str] = None):
        """Initialize Jetson deployment.
        
        Args:
            config: Deployment configuration dataclass.
            project_root_dir: Absolute path to the OpenPerception project root.
        """
        if config is None:
            # This is a placeholder. In a real scenario, the main OpenPerception app
            # would load the full config and pass the relevant DeploymentConfig section here.
            logger.warning("DeploymentConfig not provided, using placeholder defaults. This may not work.")
            self.config = self._get_placeholder_config() # Fallback to a basic internal default
        else:
            self.config = config

        if project_root_dir is None:
            # Try to determine project root relative to this file
            # This file is in OpenPerception/src/openperception/deployment/
            self.project_root_dir = Path(__file__).parent.parent.parent.parent.resolve()
        else:
            self.project_root_dir = Path(project_root_dir).resolve()
        
        logger.info(f"JetsonDeployment initialized. Project root: {self.project_root_dir}")
        logger.debug(f"Deployment config: {self.config}")

    def _get_placeholder_config(self) -> DeploymentConfig:
        """Returns a placeholder DeploymentConfig if none is provided."""
        # This should ideally match the structure of DeploymentConfig from openperception.config
        # For now, let's assume a simplified structure for target info.
        class PlaceholderTargetConfig:
            ip: str = "192.168.1.100"
            username: str = "jetson"
            ssh_key: str = "~/.ssh/id_rsa"
            deploy_path: str = "/home/jetson/OpenPerception" # Updated path
        
        class PlaceholderDeploymentConfig:
            target = PlaceholderTargetConfig()
            # Add other sections like dependencies, optimization as needed from your actual DeploymentConfig
            dependencies = {"apt": [], "pip": []}
            optimization = {"enable_tensorrt": False, "pytorch_version": ""}
            services = {"enable_systemd": False, "service_name": "openperception", "user": "jetson", "startup": False}
            logging = {"log_path": "/var/log/openperception", "log_level": "INFO"}
        
        return PlaceholderDeploymentConfig()
        
    def prepare_deployment_package(self, output_dir_name: str = "deployment_package") -> Path:
        """Prepare deployment package relative to project_root_dir/deployment/.
        
        Args:
            output_dir_name: Name of the directory for the deployment package.
        Returns:
            Path to the created deployment package directory.
        """
        package_output_dir = self.project_root_dir / "deployment" / output_dir_name
        package_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Preparing deployment package in: {package_output_dir}")

        # Copy source code (the entire OpenPerception project, filtered)
        # The source to copy is the project_root_dir itself.
        # The destination inside the package will be a folder named 'OpenPerception' (or similar)
        packaged_project_src_dir = package_output_dir / self.project_root_dir.name # e.g., deployment_package/OpenPerception
        
        if packaged_project_src_dir.exists():
            shutil.rmtree(packaged_project_src_dir)
            
        shutil.copytree(
            self.project_root_dir, # Source is the project root
            packaged_project_src_dir,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "*.pyo", "*.pyd",
                ".git*", ".github", ".vscode", ".DS_Store",
                "deployment/package", "deployment_package", # Avoid copying itself
                "build", "dist", "*.egg-info",
                "venv", "env", ".env", ".pytest_cache",
                "tests/", "examples/", "docs/" # Optionally exclude these from runtime package
            )
        )
        logger.info(f"Copied project source to {packaged_project_src_dir}")
        
        # Create setup script within the package_output_dir
        setup_script_path = package_output_dir / "setup_jetson.sh"
        deploy_target_path = self.config.target.deploy_path 

        with open(setup_script_path, "w") as f:
            f.write("#!/bin.bash\n\n")
            f.write("# Setup script for OpenPerception on Jetson\n")
            f.write("set -e\n\n") # Exit on error

            f.write(f"echo \"Deploying OpenPerception to {deploy_target_path}\"\n")
            f.write(f"mkdir -p {deploy_target_path}\n")
            # The script will be run from within package_output_dir on the target, 
            # so packaged_project_src_dir.name is the folder to copy.
            f.write(f"echo \"Copying project files...\"\n")
            f.write(f"cp -r ./{self.project_root_dir.name}/* {deploy_target_path}/\n")
            f.write(f"cd {deploy_target_path}\n\n")

            # Install APT dependencies
            if self.config.dependencies.get('apt'):
                f.write("echo \"Installing APT dependencies...\"\n")
                apt_deps = " ".join(self.config.dependencies['apt'])
                f.write(f"sudo apt-get update && sudo apt-get install -y {apt_deps}\n\n")
            
            # Install PIP dependencies
            if self.config.dependencies.get('pip'):
                f.write("echo \"Installing PIP dependencies...\"\n")
                pip_deps = list(self.config.dependencies['pip'])
                
                # Special handling for PyTorch on Jetson
                if self.config.optimization.enable_tensorrt and self.config.optimization.pytorch_version:
                    f.write("echo \"Installing optimized PyTorch for Jetson...\"\n")
                    pytorch_url = self.config.optimization.pytorch_version # Assuming this is a URL or filename
                    pytorch_whl_name = Path(pytorch_url).name
                    f.write(f"wget {pytorch_url} -O {pytorch_whl_name}\n") # Ensure URL is correct
                    f.write(f"sudo pip3 install {pytorch_whl_name}\n")
                    pip_deps = [dep for dep in pip_deps if not dep.startswith("torch")] # Remove generic torch
                
                if pip_deps:
                    pip_deps_str = " ".join(pip_deps)
                    f.write(f"sudo pip3 install {pip_deps_str}\n\n")
            
            # Install OpenPerception package itself (from the copied source)
            f.write("echo \"Installing OpenPerception package...\"\n")
            f.write("sudo python3 setup.py develop # Or pip3 install -e .\n\n")
            
            # Configure CUDA environment if needed (often handled by JetPack)
            if self.config.optimization.enable_cuda:
                f.write("echo \"Ensuring CUDA environment variables are set (typically handled by JetPack)...\"\n")
                f.write("# Example: Add to ~/.bashrc if not present, though JetPack usually sets this system-wide\n")
                f.write("# echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc\n")
                f.write("# echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc\n\n")
            
            # Setup systemd service if enabled
            if self.config.services.enable_systemd:
                f.write("echo \"Setting up systemd service...\"\n")
                service_name = self.config.services.service_name
                user = self.config.services.user
                log_dir_on_jetson = Path(self.config.logging.log_path)
                main_cli_command = f"openperception web" # Example: run web service

                f.write(f"sudo mkdir -p {log_dir_on_jetson}\n")
                f.write(f"sudo chown {user}:{user} {log_dir_on_jetson}\n\n")
                
                service_file_content = f"""[Unit]
Description=OpenPerception Service
After=network.target

[Service]
User={user}
Group={user} # Add group for completeness
WorkingDirectory={deploy_target_path}
ExecStart=/usr/bin/python3 -m openperception.main {main_cli_command}
Restart=on-failure
StandardOutput=append:{log_dir_on_jetson / (service_name + '.log')}
StandardError=append:{log_dir_on_jetson / (service_name + '.err')}
Environment="PYTHONUNBUFFERED=1" # For immediate logging

[Install]
WantedBy=multi-user.target
"""
                f.write(f"echo '{service_file_content}' | sudo tee /etc/systemd/system/{service_name}.service > /dev/null\n")
                f.write("sudo systemctl daemon-reload\n")
                
                if self.config.services.startup:
                    f.write(f"sudo systemctl enable {service_name}.service\n")
                    f.write(f"sudo systemctl restart {service_name}.service # Use restart to ensure it starts/restarts\n")
                f.write(f"echo \"Systemd service {service_name} configured.\"\n")
            
            f.write("\necho \"Jetson setup script complete! Source ~/.bashrc or relogin if CUDA paths were added.\"\n")
        
        os.chmod(setup_script_path, 0o755) # Make executable
        logger.info(f"Created setup script: {setup_script_path}")
        return package_output_dir

    def deploy_to_jetson(self, package_dir: Path):
        """Deploy the prepared package to Jetson using SSH/SCP.
        
        Args:
            package_dir: Path to the directory containing the deployment package.
        """
        target_cfg = self.config.target
        ssh_dest = f"{target_cfg.username}@{target_cfg.ip}"
        remote_temp_dir = f"/tmp/openperception_deploy_{Path(package_dir).name}"

        logger.info(f"Starting deployment of {package_dir.name} to {ssh_dest}")

        # SCP package to Jetson
        scp_command = [
            "scp", "-r",
            str(package_dir), 
            f"{ssh_dest}:{remote_temp_dir}"
        ]
        if target_cfg.ssh_key and Path(target_cfg.ssh_key).expanduser().exists():
            scp_command.insert(1, "-i")
            scp_command.insert(2, str(Path(target_cfg.ssh_key).expanduser()))
        
        logger.info(f"Transferring package with: {' '.join(scp_command)}")
        try:
            subprocess.run(scp_command, check=True, capture_output=True, text=True)
            logger.info("Package transferred successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to SCP package to Jetson: {e}")
            logger.error(f"SCP stdout: {e.stdout}")
            logger.error(f"SCP stderr: {e.stderr}")
            return

        # SSH and run setup script
        ssh_command_base = ["ssh", ssh_dest]
        if target_cfg.ssh_key and Path(target_cfg.ssh_key).expanduser().exists():
            ssh_command_base.insert(1, "-i")
            ssh_command_base.insert(2, str(Path(target_cfg.ssh_key).expanduser()))

        ssh_setup_command = ssh_command_base + [
            f"cd {remote_temp_dir} && sudo bash ./setup_jetson.sh && rm -rf {remote_temp_dir}"
        ]

        logger.info(f"Executing setup script on Jetson with: {' '.join(ssh_setup_command)}")
        try:
            # Using Popen for potentially long-running setup with streaming output
            process = subprocess.Popen(ssh_setup_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                logger.info("Deployment and setup on Jetson completed successfully.")
                logger.info(f"Jetson stdout:\n{stdout}")
            else:
                logger.error(f"Jetson setup script failed with code {process.returncode}.")
                logger.error(f"Jetson stdout:\n{stdout}")
                logger.error(f"Jetson stderr:\n{stderr}")
        except Exception as e:
            logger.error(f"Error during SSH command execution: {e}")

# CLI for JetsonDeployment
def main_deploy_cli():
    parser = argparse.ArgumentParser(description="OpenPerception Jetson Deployment Tool")
    parser.add_argument("--config", type=str, help="Path to the main OpenPerception configuration TOML file.")
    parser.add_argument("action", choices=["prepare", "deploy"], help="Action to perform: 'prepare' package or 'deploy' it.")
    parser.add_argument("--package-dir", type=str, default="deployment/deployment_package", help="Directory for the deployment package (used by 'deploy', output by 'prepare').")
    parser.add_argument("--project-root", type=str, default=None, help="Path to OpenPerception project root (if not running script from within project)." )
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Load the full config and extract the deployment part
    # This requires the main config system to be in place.
    from openperception.config import load_config as load_main_app_config
    full_app_config = load_main_app_config(config_path=args.config) # args.config can be None, load_config handles it
    deploy_cfg_data = full_app_config.deployment # Assuming main config has a 'deployment' section of DeploymentConfig type

    deployer = JetsonDeployment(config=deploy_cfg_data, project_root_dir=args.project_root)

    if args.action == "prepare":
        package_path = deployer.prepare_deployment_package(output_dir_name=Path(args.package_dir).name)
        logger.info(f"Deployment package prepared at: {package_path}")
    elif args.action == "deploy":
        package_to_deploy = Path(args.package_dir) # This path is relative to where the script is run or absolute
        if not package_to_deploy.is_dir():
            # If relative and not found, try relative to project root's deployment subdir
            if args.project_root:
                package_to_deploy = Path(args.project_root) / "deployment" / args.package_dir
            else: # Try relative to script file parent.parent... if that makes sense
                script_dir_guess = Path(__file__).parent
                package_to_deploy = script_dir_guess / args.package_dir
            
            if not package_to_deploy.is_dir():
                 logger.error(f"Deployment package directory not found: {args.package_dir} or resolved {package_to_deploy}")
                 return
        logger.info(f"Attempting to deploy package from: {package_to_deploy.resolve()}")
        deployer.deploy_to_jetson(package_dir=package_to_deploy.resolve())

if __name__ == "__main__":
    # This allows running the deployment script directly.
    # Ensure openperception package is in PYTHONPATH or installed.
    main_deploy_cli() 