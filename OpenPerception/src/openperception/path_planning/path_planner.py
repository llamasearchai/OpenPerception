"""
Path planning algorithms for OpenPerception.

This module provides implementations of common path planning algorithms
like RRT, RRT*, and A* for robotic navigation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Set, Any, Optional, Callable, Union
import logging
import heapq
import time
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PlanningResult:
    """Results of a path planning operation."""
    path: List[Tuple[float, float]]  # List of waypoints (x, y)
    cost: float  # Total path cost
    execution_time: float  # Planning execution time in seconds
    explored_nodes: int  # Number of nodes explored during planning
    success: bool  # Whether planning was successful
    
class PathPlanner:
    """Base class for path planning algorithms."""
    
    def __init__(self, 
                map_data: Optional[np.ndarray] = None,
                obstacle_threshold: float = 0.5,
                resolution: float = 0.1,
                bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None):
        """Initialize path planner.
        
        Args:
            map_data: 2D occupancy grid where values > obstacle_threshold represent obstacles
            obstacle_threshold: Threshold above which a cell is considered an obstacle
            resolution: Map resolution in meters per pixel
            bounds: Bounds of the planning space as ((min_x, min_y), (max_x, max_y))
                   If None, inferred from map_data
        """
        self.map_data = map_data
        self.obstacle_threshold = obstacle_threshold
        self.resolution = resolution
        
        if bounds is None and map_data is not None:
            # If bounds not provided, infer from map
            h, w = map_data.shape
            self.bounds = ((0, 0), (w * resolution, h * resolution))
        else:
            self.bounds = bounds or ((0, 0), (10, 10))  # Default 10x10 map
            
        self.min_x, self.min_y = self.bounds[0]
        self.max_x, self.max_y = self.bounds[1]
        
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> PlanningResult:
        """Plan a path from start to goal.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            Path planning result
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def is_valid_position(self, position: Tuple[float, float]) -> bool:
        """Check if a position is valid (in bounds and not in obstacle).
        
        Args:
            position: Position to check (x, y)
            
        Returns:
            True if position is valid, False otherwise
        """
        x, y = position
        
        # Check bounds
        if x < self.min_x or x > self.max_x or y < self.min_y or y > self.max_y:
            return False
            
        # If no map data, only check bounds
        if self.map_data is None:
            return True
            
        # Convert to grid coordinates
        grid_x = int((x - self.min_x) / self.resolution)
        grid_y = int((y - self.min_y) / self.resolution)
        
        # Check map bounds (again, in case of rounding)
        if grid_x < 0 or grid_x >= self.map_data.shape[1] or grid_y < 0 or grid_y >= self.map_data.shape[0]:
            return False
            
        # Check obstacles
        return self.map_data[grid_y, grid_x] <= self.obstacle_threshold
    
    def distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Compute Euclidean distance between two points.
        
        Args:
            p1: First point (x, y)
            p2: Second point (x, y)
            
        Returns:
            Euclidean distance
        """
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def visualize_plan(self, result: PlanningResult, start: Tuple[float, float], 
                      goal: Tuple[float, float], show_explored: bool = False) -> None:
        """Visualize planning result.
        
        Args:
            result: Planning result
            start: Start position
            goal: Goal position
            show_explored: Whether to show explored nodes
        """
        plt.figure(figsize=(10, 8))
        
        # Plot obstacles if map_data is available
        if self.map_data is not None:
            plt.imshow(self.map_data.T, origin='lower', extent=[self.min_x, self.max_x, self.min_y, self.max_y],
                      cmap='gray', alpha=0.7)
        else:
            # Just plot bounds
            plt.xlim(self.min_x, self.max_x)
            plt.ylim(self.min_y, self.max_y)
        
        # Plot path
        if result.path:
            path = np.array(result.path)
            plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Planned Path')
        
        # Plot start and goal
        plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
        plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
        
        # Add info text
        info_text = f"Path Length: {result.cost:.2f}m\n"
        info_text += f"Planning Time: {result.execution_time:.3f}s\n"
        info_text += f"Explored Nodes: {result.explored_nodes}\n"
        info_text += f"Success: {result.success}"
        
        plt.text(self.min_x + 0.05 * (self.max_x - self.min_x), 
                self.max_y - 0.15 * (self.max_y - self.min_y),
                info_text, bbox=dict(facecolor='white', alpha=0.7))
        
        plt.title('Path Planning Result')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

class AStarPlanner(PathPlanner):
    """A* path planning algorithm."""
    
    def __init__(self, **kwargs):
        """Initialize A* planner."""
        super().__init__(**kwargs)
        
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> PlanningResult:
        """Plan a path using A* algorithm.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            Path planning result
        """
        start_time = time.time()
        
        # Check if start and goal are valid
        if not self.is_valid_position(start):
            logger.error(f"Invalid start position: {start}")
            return PlanningResult([], float('inf'), 0, 0, False)
            
        if not self.is_valid_position(goal):
            logger.error(f"Invalid goal position: {goal}")
            return PlanningResult([], float('inf'), 0, 0, False)
        
        # Open set (priority queue of nodes to visit)
        # Format: (f_score, node_id, position, parent_id)
        open_set = [(self.distance(start, goal), 0, start, None)]
        heapq.heapify(open_set)
        
        # Closed set (set of visited nodes)
        closed_set = set()
        
        # Node data store
        node_data = {0: {'pos': start, 'g_score': 0, 'parent': None}}
        next_node_id = 1
        
        # Discretization step for possible moves
        directions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                directions.append((dx, dy))
        
        # Main A* loop
        while open_set:
            # Get node with lowest f_score
            f_score, node_id, current, parent_id = heapq.heappop(open_set)
            
            # Check if we've already processed this node
            if current in closed_set:
                continue
                
            # Mark as visited
            closed_set.add(current)
            
            # Check if reached goal
            if self.distance(current, goal) < self.resolution:
                # Reconstruct path
                path = []
                current_id = node_id
                
                while current_id is not None:
                    node = node_data[current_id]
                    path.append(node['pos'])
                    current_id = node['parent']
                    
                path.reverse()
                
                # Add goal to path if not already added
                if self.distance(path[-1], goal) > 1e-6:
                    path.append(goal)
                
                # Calculate path cost
                cost = 0
                for i in range(len(path) - 1):
                    cost += self.distance(path[i], path[i+1])
                
                return PlanningResult(
                    path=path,
                    cost=cost,
                    execution_time=time.time() - start_time,
                    explored_nodes=len(closed_set),
                    success=True
                )
            
            # Generate neighbors
            for dx, dy in directions:
                step_size = self.resolution
                neighbor = (current[0] + dx * step_size, current[1] + dy * step_size)
                
                # Skip if neighbor is invalid or already visited
                if not self.is_valid_position(neighbor) or neighbor in closed_set:
                    continue
                
                # Calculate g_score (cost from start to neighbor)
                g_score = node_data[node_id]['g_score'] + self.distance(current, neighbor)
                
                # Calculate h_score (heuristic from neighbor to goal)
                h_score = self.distance(neighbor, goal)
                
                # Calculate f_score
                f_score = g_score + h_score
                
                # Create new node
                neighbor_id = next_node_id
                next_node_id += 1
                
                # Store node data
                node_data[neighbor_id] = {
                    'pos': neighbor,
                    'g_score': g_score,
                    'parent': node_id
                }
                
                # Add to open set
                heapq.heappush(open_set, (f_score, neighbor_id, neighbor, node_id))
        
        # If we get here, no path was found
        logger.warning(f"No path found from {start} to {goal}")
        return PlanningResult(
            path=[],
            cost=float('inf'),
            execution_time=time.time() - start_time,
            explored_nodes=len(closed_set),
            success=False
        )

class RRTPlanner(PathPlanner):
    """Rapidly-exploring Random Tree (RRT) path planning algorithm."""
    
    def __init__(self, 
                max_iterations: int = 1000, 
                step_size: float = 0.2,
                goal_sample_rate: float = 0.1,
                goal_threshold: float = 0.5,
                **kwargs):
        """Initialize RRT planner.
        
        Args:
            max_iterations: Maximum number of iterations
            step_size: Maximum distance between nodes
            goal_sample_rate: Probability of sampling the goal
            goal_threshold: Distance threshold for reaching the goal
            **kwargs: Additional arguments for PathPlanner
        """
        super().__init__(**kwargs)
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.goal_threshold = goal_threshold
        
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> PlanningResult:
        """Plan a path using RRT algorithm.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            Path planning result
        """
        start_time = time.time()
        
        # Check if start and goal are valid
        if not self.is_valid_position(start):
            logger.error(f"Invalid start position: {start}")
            return PlanningResult([], float('inf'), 0, 0, False)
            
        if not self.is_valid_position(goal):
            logger.error(f"Invalid goal position: {goal}")
            return PlanningResult([], float('inf'), 0, 0, False)
        
        # Node store: node_id -> (position, parent_id)
        nodes = {0: (start, None)}
        next_node_id = 1
        
        # Main RRT loop
        for i in range(self.max_iterations):
            # Sample random point
            if random.random() < self.goal_sample_rate:
                # Sample goal
                random_point = goal
            else:
                # Sample random point in bounds
                random_point = (
                    self.min_x + random.random() * (self.max_x - self.min_x),
                    self.min_y + random.random() * (self.max_y - self.min_y)
                )
            
            # Find nearest node
            nearest_id = min(nodes.keys(), key=lambda nid: self.distance(nodes[nid][0], random_point))
            nearest_point = nodes[nearest_id][0]
            
            # Get new point in direction of random_point
            direction = np.array(random_point) - np.array(nearest_point)
            norm = np.linalg.norm(direction)
            
            if norm < 1e-6:  # If random_point is very close to nearest_point
                continue
                
            # Scale to step_size
            if norm > self.step_size:
                direction = direction / norm * self.step_size
                
            new_point = tuple(np.array(nearest_point) + direction)
            
            # Check if new_point is valid
            if not self.is_valid_position(new_point):
                continue
                
            # Check if path is collision-free
            if not self._is_path_valid(nearest_point, new_point):
                continue
                
            # Add new node
            new_id = next_node_id
            next_node_id += 1
            nodes[new_id] = (new_point, nearest_id)
            
            # Check if reached goal
            if self.distance(new_point, goal) < self.goal_threshold:
                # Reconstruct path
                path = []
                current_id = new_id
                
                while current_id is not None:
                    node, parent_id = nodes[current_id]
                    path.append(node)
                    current_id = parent_id
                    
                path.reverse()
                
                # Add goal to path if not already close enough
                if self.distance(path[-1], goal) > self.resolution:
                    path.append(goal)
                
                # Calculate path cost
                cost = 0
                for i in range(len(path) - 1):
                    cost += self.distance(path[i], path[i+1])
                
                return PlanningResult(
                    path=path,
                    cost=cost,
                    execution_time=time.time() - start_time,
                    explored_nodes=len(nodes),
                    success=True
                )
        
        # If we get here, no path was found
        logger.warning(f"No path found from {start} to {goal} after {self.max_iterations} iterations")
        return PlanningResult(
            path=[],
            cost=float('inf'),
            execution_time=time.time() - start_time,
            explored_nodes=len(nodes),
            success=False
        )
        
    def _is_path_valid(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """Check if path between two points is collision-free.
        
        Args:
            p1: First point (x, y)
            p2: Second point (x, y)
            
        Returns:
            True if path is valid, False otherwise
        """
        # If no map data, assume all paths are valid
        if self.map_data is None:
            return True
            
        # Check path using discrete steps
        direction = np.array(p2) - np.array(p1)
        distance = np.linalg.norm(direction)
        
        # Number of steps to check
        steps = max(2, int(distance / (self.resolution / 2)))
        
        for i in range(steps + 1):
            # Interpolate point
            t = i / steps
            point = tuple(np.array(p1) + t * direction)
            
            # Check if point is valid
            if not self.is_valid_position(point):
                return False
                
        return True

class RRTStarPlanner(RRTPlanner):
    """RRT* path planning algorithm."""
    
    def __init__(self, 
                rewire_radius: float = 1.0,
                **kwargs):
        """Initialize RRT* planner.
        
        Args:
            rewire_radius: Radius for rewiring nearby nodes
            **kwargs: Additional arguments for RRTPlanner
        """
        super().__init__(**kwargs)
        self.rewire_radius = rewire_radius
        
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> PlanningResult:
        """Plan a path using RRT* algorithm.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            Path planning result
        """
        start_time = time.time()
        
        # Check if start and goal are valid
        if not self.is_valid_position(start):
            logger.error(f"Invalid start position: {start}")
            return PlanningResult([], float('inf'), 0, 0, False)
            
        if not self.is_valid_position(goal):
            logger.error(f"Invalid goal position: {goal}")
            return PlanningResult([], float('inf'), 0, 0, False)
        
        # Node store: node_id -> (position, parent_id, cost)
        nodes = {0: (start, None, 0.0)}
        next_node_id = 1
        
        # Best path to goal so far
        best_goal_id = None
        best_goal_cost = float('inf')
        
        # Main RRT* loop
        for i in range(self.max_iterations):
            # Sample random point
            if random.random() < self.goal_sample_rate:
                # Sample goal
                random_point = goal
            else:
                # Sample random point in bounds
                random_point = (
                    self.min_x + random.random() * (self.max_x - self.min_x),
                    self.min_y + random.random() * (self.max_y - self.min_y)
                )
            
            # Find nearest node
            nearest_id = min(nodes.keys(), key=lambda nid: self.distance(nodes[nid][0], random_point))
            nearest_point, nearest_parent, nearest_cost = nodes[nearest_id]
            
            # Get new point in direction of random_point
            direction = np.array(random_point) - np.array(nearest_point)
            norm = np.linalg.norm(direction)
            
            if norm < 1e-6:  # If random_point is very close to nearest_point
                continue
                
            # Scale to step_size
            if norm > self.step_size:
                direction = direction / norm * self.step_size
                
            new_point = tuple(np.array(nearest_point) + direction)
            
            # Check if new_point is valid
            if not self.is_valid_position(new_point):
                continue
                
            # Find nearby nodes for rewiring
            nearby_ids = [
                nid for nid in nodes.keys()
                if self.distance(nodes[nid][0], new_point) < self.rewire_radius
            ]
            
            # Find best parent for new node
            best_parent_id = None
            best_parent_cost = float('inf')
            
            for nid in nearby_ids:
                node_point, node_parent, node_cost = nodes[nid]
                
                # Check if path is collision-free
                if not self._is_path_valid(node_point, new_point):
                    continue
                    
                # Calculate potential cost through this node
                potential_cost = node_cost + self.distance(node_point, new_point)
                
                # Update best parent if better
                if potential_cost < best_parent_cost:
                    best_parent_id = nid
                    best_parent_cost = potential_cost
            
            # If no valid parent found, continue
            if best_parent_id is None:
                continue
                
            # Add new node
            new_id = next_node_id
            next_node_id += 1
            nodes[new_id] = (new_point, best_parent_id, best_parent_cost)
            
            # Rewire nearby nodes if new node provides better path
            for nid in nearby_ids:
                if nid == best_parent_id:
                    continue
                    
                node_point, node_parent, node_cost = nodes[nid]
                
                # Check if path through new node is better
                if best_parent_cost + self.distance(new_point, node_point) < node_cost:
                    # Check if rewired path is collision-free
                    if self._is_path_valid(new_point, node_point):
                        # Update node's parent and cost
                        nodes[nid] = (
                            node_point,
                            new_id,
                            best_parent_cost + self.distance(new_point, node_point)
                        )
            
            # Check if reached goal
            if self.distance(new_point, goal) < self.goal_threshold:
                # Calculate cost to goal
                goal_cost = best_parent_cost + self.distance(new_point, goal)
                
                # Update best path to goal if better
                if goal_cost < best_goal_cost:
                    best_goal_id = new_id
                    best_goal_cost = goal_cost
        
        # If a path to goal was found
        if best_goal_id is not None:
            # Reconstruct path
            path = []
            current_id = best_goal_id
            
            while current_id is not None:
                node, parent_id, _ = nodes[current_id]
                path.append(node)
                current_id = parent_id
                
            path.reverse()
            
            # Add goal to path if not already close enough
            if self.distance(path[-1], goal) > self.resolution:
                path.append(goal)
            
            # Calculate path cost (recalculate for accuracy)
            cost = 0
            for i in range(len(path) - 1):
                cost += self.distance(path[i], path[i+1])
            
            return PlanningResult(
                path=path,
                cost=cost,
                execution_time=time.time() - start_time,
                explored_nodes=len(nodes),
                success=True
            )
        
        # If we get here, no path was found
        logger.warning(f"No path found from {start} to {goal} after {self.max_iterations} iterations")
        return PlanningResult(
            path=[],
            cost=float('inf'),
            execution_time=time.time() - start_time,
            explored_nodes=len(nodes),
            success=False
        )

def get_planner(algorithm: str = "rrt_star", **kwargs) -> PathPlanner:
    """Get a path planner by algorithm name.
    
    Args:
        algorithm: Algorithm name ("a_star", "rrt", "rrt_star")
        **kwargs: Additional arguments for the planner
        
    Returns:
        Path planner instance
    """
    algorithm = algorithm.lower()
    
    if algorithm == "a_star":
        return AStarPlanner(**kwargs)
    elif algorithm == "rrt":
        return RRTPlanner(**kwargs)
    elif algorithm == "rrt_star":
        return RRTStarPlanner(**kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

def plan_path(start: Tuple[float, float], goal: Tuple[float, float], 
             algorithm: str = "rrt_star", visualize: bool = False, 
             **kwargs) -> PlanningResult:
    """Plan a path from start to goal.
    
    Args:
        start: Start position (x, y)
        goal: Goal position (x, y)
        algorithm: Planning algorithm to use
        visualize: Whether to visualize the result
        **kwargs: Additional arguments for the planner
        
    Returns:
        Path planning result
    """
    # Get planner
    planner = get_planner(algorithm, **kwargs)
    
    # Plan path
    result = planner.plan(start, goal)
    
    # Visualize if requested
    if visualize:
        planner.visualize_plan(result, start, goal)
    
    return result 