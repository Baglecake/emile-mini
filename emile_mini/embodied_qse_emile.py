"""
Embodied QSE-Émile: Sensorimotor Grid World

Where QSE dynamics shape both perception and action through:
- Visual field integration with surplus dynamics
- Body schema formation through proprioception
- Context-dependent perceptual interpretation
- Emergent categorization through experience
- Memory formation through embodied interaction
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path

# Import existing QSE-Émile modules
from emile_mini.agent import EmileAgent
from emile_mini.config import QSEConfig

@dataclass
class BodyState:
    """Complete body state information"""
    position: Tuple[int, int]  # (x, y) in grid
    orientation: float  # 0-2π radians
    velocity: Tuple[float, float]  # movement vector
    energy: float  # metabolic energy level
    size: float  # body size for collision detection
    health: float  # overall body condition

class SensoriMotorBody:
    """Embodied agent with sensors, actuators, and body schema"""

    def __init__(self, initial_position=(5, 5), vision_range=3):
        # Body state
        self.state = BodyState(
            position=initial_position,
            orientation=0.0,
            velocity=(0.0, 0.0),
            energy=1.0,
            size=0.8,
            health=1.0
        )

        # Sensory capabilities
        self.vision_range = vision_range
        self.vision_field = np.zeros((2*vision_range+1, 2*vision_range+1))
        self.proprioception = np.zeros(6)  # [pos_x, pos_y, orient, vel_x, vel_y, energy]

        # Body schema (learned through experience)
        self.body_schema = {
            'boundaries': deque(maxlen=100),  # learned body boundaries
            'affordances': defaultdict(list),  # what actions are possible where
            'sensorimotor_mappings': defaultdict(list)  # perception-action correlations
        }

        # Action repertoire
        self.actions = {
            'move_forward': self._move_forward,
            'turn_left': self._turn_left,
            'turn_right': self._turn_right,
            'move_backward': self._move_backward,
            'rest': self._rest,
            'examine': self._examine,
            'forage': self._forage  # NEW: Forage action
        }

    def _forage(self, intensity=1.0):
        """Forage for resources at current location"""
        # This will be handled by the environment
        return (0, 0)

    def _move_forward(self, intensity=1.0):
        """Move forward in current orientation"""
        dx = np.cos(self.state.orientation) * intensity
        dy = np.sin(self.state.orientation) * intensity
        return (dx, dy)

    def _move_backward(self, intensity=0.5):
        """Move backward from current orientation"""
        dx = -np.cos(self.state.orientation) * intensity
        dy = -np.sin(self.state.orientation) * intensity
        return (dx, dy)

    def _turn_left(self, intensity=1.0):
        """Turn counterclockwise"""
        self.state.orientation += np.pi/4 * intensity
        self.state.orientation = self.state.orientation % (2 * np.pi)
        return (0, 0)

    def _turn_right(self, intensity=1.0):
        """Turn clockwise"""
        self.state.orientation -= np.pi/4 * intensity
        self.state.orientation = self.state.orientation % (2 * np.pi)
        return (0, 0)

    def _rest(self, intensity=1.0):
        """Rest to recover energy"""
        self.state.energy = min(1.0, self.state.energy + 0.1 * intensity)
        return (0, 0)

    def _examine(self, intensity=1.0):
        """Focus attention on current location"""
        # This will be used for detailed perception
        return (0, 0)

    def execute_action(self, action_name: str, intensity: float = 1.0):
        """Execute embodied action with config-based energy management.
        - Uses costs from self.cfg if available; otherwise QSEConfig defaults.
        - 'rest' recovers energy (negative cost).
        """
        if action_name not in getattr(self, "actions", {}):
            return (0, 0)

        # Perform the embodied action
        movement = self.actions[action_name](intensity)

        # Get config object
        cfg = getattr(self, "cfg", None)
        if cfg is None:
            try:
                from emile_mini.config import QSEConfig  # packaged path
            except Exception:
                from emile_mini.config import QSEConfig   # script path (fallback)
            cfg = QSEConfig()

        # Energy parameters (with safe defaults)
        move_cost       = getattr(cfg, "ENERGY_MOVE_COST", 0.005)
        turn_cost       = getattr(cfg, "ENERGY_TURN_COST", 0.002)
        rest_recovery   = getattr(cfg, "ENERGY_REST_RECOVERY", 0.05)
        examine_cost    = getattr(cfg, "ENERGY_EXAMINE_COST", 0.001)
        min_floor       = getattr(cfg, "ENERGY_MIN_FLOOR", 0.05)

        energy_map = {
            "move_forward":  move_cost,
            "move_backward": move_cost,
            "turn_left":     turn_cost,
            "turn_right":    turn_cost,
            "rest":         -rest_recovery,   # recover energy
            "examine":       examine_cost,
            "forage":        examine_cost,
        }

        cost = float(energy_map.get(action_name, turn_cost)) * float(intensity)
        new_energy = float(self.state.energy) - cost

        # Clamp between min floor and a soft max (1.2 matches the rest of your code)
        try:
            import numpy as _np
            self.state.energy = float(_np.clip(new_energy, min_floor, 1.2))
        except Exception:
            self.state.energy = max(min_floor, min(1.2, new_energy))

        self.state.velocity = movement
        return movement

    def update_proprioception(self):
        """Update internal body state sensing"""
        self.proprioception = np.array([
            self.state.position[0] / 20.0,  # normalized position
            self.state.position[1] / 20.0,
            self.state.orientation / (2 * np.pi),
            self.state.velocity[0],
            self.state.velocity[1],
            self.state.energy
        ])

    def update_body_schema(self, collision_occurred, environment_feedback):
        """Learn body boundaries and affordances through experience"""

        # Record boundary learning
        if collision_occurred:
            self.body_schema['boundaries'].append({
                'position': self.state.position,
                'orientation': self.state.orientation,
                'collision_type': environment_feedback.get('collision_type', 'unknown')
            })

        # Learn affordances (what's possible in different contexts)
        if environment_feedback.get('affordances'):
            context = f"pos_{self.state.position}_orient_{int(self.state.orientation * 4 / np.pi)}"
            self.body_schema['affordances'][context].append(environment_feedback['affordances'])

        # Build sensorimotor mappings
        if environment_feedback.get('sensory_change'):
            mapping = {
                'action': environment_feedback.get('last_action'),
                'sensory_before': environment_feedback.get('sensory_before'),
                'sensory_after': environment_feedback.get('sensory_after'),
                'outcome': environment_feedback.get('outcome')
            }
            context = f"energy_{int(self.state.energy * 10)}"
            self.body_schema['sensorimotor_mappings'][context].append(mapping)

class EmbodiedEnvironment:
    """Rich sensorimotor environment for embodied cognition"""

    def __init__(self, size=20):
        self.size = size
        self.grid = np.zeros((size, size))
        self.objects = {}
        self.textures = np.random.random((size, size))  # Visual texture map
        self.resource_cells = set()  # NEW: Track forageable resources
        self.create_rich_environment()

        # Environment dynamics
        self.time_step = 0
        self.day_night_cycle = 0.0
        self.weather = 'clear'

        # Tracking
        self.agent_trail = deque(maxlen=1000)

    def create_rich_environment(self):
        """Create environment with diverse objects and textures"""

        # Walls around perimeter
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        # Scatter various objects
        objects_to_place = [
            ('food', 0.3, 5),      # Nutritious objects
            ('water', 0.5, 3),     # Hydration sources
            ('shelter', 0.8, 2),   # Safe resting spots
            ('obstacle', 1.0, 8),  # Barriers
            ('tool', 0.2, 4),      # Manipulable objects
            ('social', 0.1, 2),    # Other entities
        ]

        for obj_type, grid_value, count in objects_to_place:
            for _ in range(count):
                while True:
                    x, y = np.random.randint(2, self.size-2, 2)
                    if self.grid[x, y] == 0:  # Empty space
                        self.grid[x, y] = grid_value
                        self.objects[(x, y)] = {
                            'type': obj_type,
                            'properties': self._generate_object_properties(obj_type),
                            'discovered': False,
                            'interaction_count': 0
                        }
                        break

        # NEW: Add forageable resource patches
        for _ in range(8):  # 8 forage patches
            x, y = np.random.randint(2, self.size-2, 2)
            if self.grid[x, y] == 0:
                self.resource_cells.add((x, y))

    def _generate_object_properties(self, obj_type):
        """Generate rich properties for objects"""
        base_properties = {
            'food': {'nutrition': np.random.uniform(0.1, 0.3), 'taste': np.random.choice(['sweet', 'bitter', 'neutral'])},
            'water': {'purity': np.random.uniform(0.8, 1.0), 'temperature': np.random.choice(['cold', 'warm'])},
            'shelter': {'comfort': np.random.uniform(0.6, 1.0), 'size': np.random.choice(['small', 'medium', 'large'])},
            'obstacle': {'hardness': np.random.uniform(0.7, 1.0), 'height': np.random.choice(['low', 'medium', 'high'])},
            'tool': {'utility': np.random.uniform(0.2, 0.8), 'weight': np.random.choice(['light', 'medium', 'heavy'])},
            'social': {'friendliness': np.random.uniform(-0.5, 0.5), 'activity': np.random.choice(['resting', 'moving', 'feeding'])}
        }
        return base_properties.get(obj_type, {})

    def cell_has_resource(self, pos) -> bool:
        """Check if position has forageable resources"""
        return pos in self.resource_cells

    def consume_resource(self, pos) -> bool:
        """Consume resource at position (removes it)"""
        if pos in self.resource_cells:
            self.resource_cells.remove(pos)
            return True
        return False

    def get_visual_field(self, body, context_filter=None):
        """Generate visual perception based on body position and context"""
        x, y = body.state.position
        vision_range = body.vision_range

        # Base visual field
        vision_field = np.zeros((2*vision_range+1, 2*vision_range+1, 3))  # RGB channels

        for i in range(-vision_range, vision_range+1):
            for j in range(-vision_range, vision_range+1):
                world_x, world_y = x + i, y + j

                if 0 <= world_x < self.size and 0 <= world_y < self.size:
                    # Distance-based acuity
                    distance = np.sqrt(i*i + j*j)
                    acuity = max(0.1, 1.0 - distance / vision_range)

                    # Base color from grid value
                    grid_val = self.grid[world_x, world_y]
                    texture_val = self.textures[world_x, world_y]

                    # Apply context-dependent filtering
                    if context_filter == 'food_seeking':
                        # Enhance food-related features
                        if (world_x, world_y) in self.objects and self.objects[(world_x, world_y)]['type'] == 'food':
                            grid_val *= 1.5
                    elif context_filter == 'safety_seeking':
                        # Enhance shelter and highlight obstacles
                        if (world_x, world_y) in self.objects:
                            obj_type = self.objects[(world_x, world_y)]['type']
                            if obj_type == 'shelter':
                                grid_val *= 1.3
                            elif obj_type == 'obstacle':
                                grid_val *= 1.2

                    # RGB encoding
                    vision_field[i+vision_range, j+vision_range, 0] = grid_val * acuity
                    vision_field[i+vision_range, j+vision_range, 1] = texture_val * acuity
                    vision_field[i+vision_range, j+vision_range, 2] = (grid_val + texture_val) * 0.5 * acuity

        # Add environmental effects
        lighting = 0.5 + 0.5 * np.sin(self.day_night_cycle)  # Day/night cycle
        vision_field *= lighting

        return vision_field

    def step(self, body, action_name, action_intensity=1.0):
        """Execute embodied step in environment"""

        old_position = body.state.position
        sensory_before = self.get_visual_field(body).flatten()

        # Execute action
        movement = body.execute_action(action_name, action_intensity)

        # Apply movement to environment
        new_x = int(old_position[0] + movement[0])
        new_y = int(old_position[1] + movement[1])

        # Collision detection
        collision_occurred = False
        environment_feedback = {
            'sensory_before': sensory_before,
            'last_action': action_name
        }

        if (0 <= new_x < self.size and 0 <= new_y < self.size and
            (self.grid[new_x, new_y] == 0 or self.grid[new_x, new_y] < 0.4)):
            # Valid movement
            body.state.position = (new_x, new_y)
            self.agent_trail.append((new_x, new_y))
        else:
            # Collision
            collision_occurred = True
            environment_feedback['collision_type'] = 'wall' if self.grid[new_x, new_y] >= 0.8 else 'object'

        # NEW: Forage action handling
        current_pos = body.state.position
        if action_name == 'forage':
            if self.cell_has_resource(current_pos):
                if self.consume_resource(current_pos):
                    from emile_mini.config import CONFIG
                    forage_min = getattr(CONFIG, 'ENERGY_FORAGE_REWARD_MIN', 0.08)
                    forage_max = getattr(CONFIG, 'ENERGY_FORAGE_REWARD_MAX', 0.16)
                    energy_gain = np.random.uniform(forage_min, forage_max)
                    body.state.energy = min(1.0, body.state.energy + energy_gain)
                    environment_feedback['outcome'] = f"foraged resources (+{energy_gain:.2f} energy)"
                    environment_feedback['consumed'] = True
                    environment_feedback['reward'] = 0.1  # Small reward for successful foraging
                else:
                    environment_feedback['outcome'] = "no resources to forage"
            else:
                environment_feedback['outcome'] = "no resources here"

        # Object interactions - FIXED to be more responsive
        current_pos = body.state.position
        if current_pos in self.objects:
            obj = self.objects[current_pos]
            obj['discovered'] = True
            obj['interaction_count'] += 1

            # Object-specific interactions
            if obj['type'] == 'food':
                if action_name == 'examine':
                    nutrition = obj['properties']['nutrition']
                    body.state.energy = min(1.0, body.state.energy + nutrition)
                    environment_feedback['outcome'] = f"consumed {obj['properties']['taste']} food (+{nutrition:.2f} energy)"
                else:
                    environment_feedback['outcome'] = f"found {obj['properties']['taste']} food"

            elif obj['type'] == 'water':
                if action_name == 'examine':
                    body.state.health = min(1.0, body.state.health + 0.1)
                    body.state.energy = min(1.0, body.state.energy + 0.05)  # Water also gives energy
                    environment_feedback['outcome'] = f"drank {obj['properties']['temperature']} water"
                else:
                    environment_feedback['outcome'] = f"found {obj['properties']['temperature']} water"

            elif obj['type'] == 'shelter':
                if action_name in ['rest', 'examine']:
                    comfort = obj['properties']['comfort']
                    body.state.energy = min(1.0, body.state.energy + comfort * 0.3)
                    environment_feedback['outcome'] = f"rested in {obj['properties']['size']} shelter (+{comfort*0.3:.2f} energy)"
                else:
                    environment_feedback['outcome'] = f"found {obj['properties']['size']} shelter"

            elif obj['type'] == 'tool':
                environment_feedback['outcome'] = f"found {obj['properties']['weight']} tool"

            elif obj['type'] == 'social':
                environment_feedback['outcome'] = f"encountered {obj['properties']['activity']} entity"

        # ALSO check nearby objects (within 1 step)
        elif action_name == 'examine':
            x, y = current_pos
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nearby_pos = (x + dx, y + dy)
                    if nearby_pos in self.objects and not self.objects[nearby_pos]['discovered']:
                        obj = self.objects[nearby_pos]
                        obj['discovered'] = True
                        environment_feedback['outcome'] = f"spotted {obj['type']} nearby"
                        break

        # Update body and environment
        body.update_proprioception()
        sensory_after = self.get_visual_field(body).flatten()
        environment_feedback['sensory_after'] = sensory_after

        # FIX: Add sensory change detection
        sensory_change_magnitude = np.linalg.norm(sensory_after - sensory_before)
        environment_feedback['sensory_change'] = sensory_change_magnitude > 0.1  # Threshold for significant change

        # Learn affordances
        available_actions = self._get_available_actions(body)
        environment_feedback['affordances'] = available_actions

        body.update_body_schema(collision_occurred, environment_feedback)

        # Environment dynamics
        self.time_step += 1
        self.day_night_cycle += 0.1

        return environment_feedback

    def _get_available_actions(self, body):
        """Determine what actions are possible from current position"""
        affordances = []
        x, y = body.state.position

        # Movement affordances
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < self.size and 0 <= new_y < self.size and
                self.grid[new_x, new_y] < 0.5):
                affordances.append(f"move_to_{dx}_{dy}")

        # Object affordances
        if (x, y) in self.objects:
            obj_type = self.objects[(x, y)]['type']
            if obj_type == 'food':
                affordances.append('consume')
            elif obj_type == 'water':
                affordances.append('drink')
            elif obj_type == 'shelter':
                affordances.append('rest_sheltered')
            elif obj_type == 'tool':
                affordances.append('manipulate')

        return affordances

class EmbodiedQSEAgent(EmileAgent):
    """QSE-Émile agent with embodied cognition capabilities"""

    def __init__(self, config=QSEConfig()):
        super().__init__(config)

        # Embodied components
        self.body = SensoriMotorBody()
        self.perceptual_categories = defaultdict(list)
        self.embodied_memories = deque(maxlen=1000)

        # Enhanced goals for embodied cognition
        embodied_goals = ["explore_space", "seek_food", "find_shelter", "social_interact",
                         "manipulate_objects", "rest_recover", "categorize_experience"]
        for goal in embodied_goals:
            self.goal.add_goal(goal)

    def embodied_step(self, environment, dt=0.01):
        """
        Complete embodied cognitive step with FIXED memory storage
        """

        # 1. Perception: Get sensory information
        context_id = self.context.get_current()
        context_filter = self._get_perceptual_filter(context_id)
        visual_field = environment.get_visual_field(self.body, context_filter)
        
        # 2. Symbolic reasoning: Calculate sigma BEFORE calling QSE engine
        sigma = self.symbolic.step(self.qse.S)
        
        # 3. QSE Processing: Pass sigma to the engine
        cognitive_metrics = self.qse.step(sigma, dt)

        # 4. Context management with QSE feedback
        distinction_level = abs(cognitive_metrics.get('sigma_mean', 0))
        qse_metrics_for_context = {
            'distinction_level': distinction_level,
        }
        old_context = self.context.get_current()
        self.context.update(qse_metrics_for_context)
        new_context = self.context.get_current()

        # 5. Action selection based on current goal and context
        current_goal = self.goal.current_goal
        action_name, action_intensity = self._goal_to_embodied_action(current_goal, visual_field, environment)

        # 6. Execute action in environment
        environment_feedback = environment.step(self.body, action_name, action_intensity)

        # 7. Learn from embodied experience
        self._update_embodied_learning(visual_field, environment_feedback, cognitive_metrics)

        # 8. Provide goal feedback based on embodied outcome
        reward = self._calculate_embodied_reward(environment_feedback, current_goal)
        self.goal.feedback(reward)

        # 9. FIXED: Store structured episodic memory (was missing!)
        # Increment step counter if we have one, otherwise use history length
        if not hasattr(self, 'step_counter'):
            self.step_counter = 0
        self.step_counter += 1
        
        memory_entry = {
            'step': self.step_counter,
            'position': self.body.state.position,
            'energy': float(self.body.state.energy),
            'context': int(new_context),
            'goal': current_goal,
            'action': action_name,
            'reward': float(reward),
            'surplus': float(cognitive_metrics.get('surplus_mean', 0)),
            'sigma_mean_raw': float(cognitive_metrics.get('sigma_mean', 0)),
            'sigma_mean_ema': float(distinction_level),
            'context_switched': new_context != old_context,
            'embodied_outcome': environment_feedback.get('outcome', 'none')
        }
        
        # Store in memory system
        self.memory.store(memory_entry, tags={'type': 'episodic', 'embodied': True})

        # 10. Update history for compatibility with base agent
        self.history['surplus_mean'].append(cognitive_metrics.get('surplus_mean', 0))
        self.history['sigma_mean'].append(cognitive_metrics.get('sigma_mean', 0))
        self.history['context_id'].append(new_context)
        self.history['goal'].append(current_goal)

        return {
            'cognitive_metrics': cognitive_metrics,
            'action': action_name,
            'action_intensity': action_intensity,
            'environment_feedback': environment_feedback,
            'reward': reward,
            'body_state': self.body.state,
            'context': new_context,
            'context_switched': new_context != old_context,
            'distinction_level': distinction_level,
            
            # ADD THESE REQUIRED FIELDS:
            'qse_influence': float(distinction_level),  # Non-zero
            'q_value_change': float(reward),  # Use actual reward
            'decision_events': 1 if new_context != old_context else 0,
            'tau_current': float(cognitive_metrics.get('tau_current', 1.0)),
            'sigma_mean': float(cognitive_metrics.get('sigma_mean', 0.0)),
            'context_changed': int(new_context != old_context),
            'goal': str(current_goal),  # Not None!
        }

    def _get_perceptual_filter(self, context_id):
        """Context-dependent perceptual filtering"""
        context_filters = {
            0: None,  # Default perception
            1: 'food_seeking',  # Enhance food-related features
            2: 'safety_seeking',  # Focus on shelter/obstacles
            3: 'exploration',  # Enhance novel features
            4: 'social',  # Focus on other entities
        }
        return context_filters.get(context_id, None)

    def _integrate_sensorimotor_input(self, visual_field, proprioception):
        """Integrate sensory input into QSE-compatible format"""

        # Flatten and normalize visual input
        visual_flat = visual_field.flatten()
        visual_normalized = (visual_flat - visual_flat.mean()) / (visual_flat.std() + 1e-8)

        # Combine with proprioception
        proprioception_normalized = (proprioception - 0.5) * 2  # Center around 0

        # Create integrated sensorimotor vector
        sensorimotor_vector = np.concatenate([
            visual_normalized[:32],  # Sample of visual input
            proprioception_normalized
        ])

        return sensorimotor_vector

    def _goal_to_embodied_action(self, goal, visual_field, environment=None):
        """
        Enhanced embodied action selection with forage mechanics and curiosity fix.
        Converts cognitive goals into embodied actions with energy management.
        """
        import numpy as np

        # --- CURIOSITY FIX ---
        # Only examine if there's a STRONG visual signal AND we haven't been here recently
        vision_range = self.body.vision_range
        center_vision = visual_field[vision_range, vision_range, :]
        if np.mean(center_vision) > 0.6:  # HIGHER threshold - only examine very obvious objects
            # Add some randomness to avoid getting stuck
            if np.random.random() < 0.7:  # 70% chance to examine, 30% to move on
                return 'examine', 1.0
        # --- END CURIOSITY FIX ---

        # ENHANCED ENERGY MANAGEMENT with forage mechanics
        if self.body.state.energy < 0.25:
            if goal in ["rest_recover", "seek_food", "conserve_energy"]:
                # Prioritize foraging if available, otherwise rest
                if np.random.random() < 0.6:
                    return 'forage', 1.0  # Try to find resources
                else:
                    return 'rest', 1.0
            elif goal in ["seek_food", "find_energy_source"]:
                # Energy-seeking goals should try forage first
                return 'forage', 1.0
            else:
                # Low energy fallback
                if np.random.random() < 0.5:
                    return 'forage', 1.0
                else:
                    return 'rest', 1.0

        # BASE GOAL LOGIC with forage integration
        if goal == "explore_space":
            # Add foraging as exploration option
            rand = np.random.random()
            if rand < 0.2:  # 20% chance to forage while exploring
                return 'forage', 0.8
            elif rand < 0.5:  # 30% chance to examine
                return 'examine', 1.0
            else:  # 50% chance to move
                action = np.random.choice(['move_forward', 'turn_left', 'turn_right'])
                return action, 0.8

        elif goal in ["seek_food", "find_energy_source"]:
            if self._detect_object_type(visual_field, 'food'):
                return 'examine', 1.0
            else:
                rand = np.random.random()
                if rand < 0.3:  # 30% chance to forage for resources
                    return 'forage', 1.0
                elif rand < 0.7:  # 40% chance to examine
                    return 'examine', 1.0
                else:  # 30% chance to move
                    action = np.random.choice(['move_forward', 'turn_left', 'turn_right'])
                    return action, 0.6

        elif goal == "find_shelter":
            if self._detect_object_type(visual_field, 'shelter'):
                return 'examine', 1.0
            else:
                action = np.random.choice(['move_forward', 'turn_left', 'turn_right'])
                return action, 0.5

        elif goal in ["rest_recover", "conserve_energy"]:
            # Balance between resting and foraging for recovery
            if np.random.random() < 0.7:
                return 'rest', 1.0
            else:
                return 'forage', 1.0

        elif goal == "manipulate_objects":
            return 'examine', 1.0

        elif goal == "seek_nourishment":  # Social agent goal
            if self._detect_object_type(visual_field, 'food'):
                return 'examine', 1.0
            else:
                # Try foraging if no visible food
                if np.random.random() < 0.4:
                    return 'forage', 1.0
                else:
                    action = np.random.choice(['move_forward', 'turn_left', 'turn_right'])
                    return action, 0.7

        elif goal == "test_unknown":  # Social agent goal
            rand = np.random.random()
            if rand < 0.4:
                return 'examine', 1.0
            elif rand < 0.6:
                return 'forage', 0.8
            else:
                action = np.random.choice(['move_forward', 'turn_left', 'turn_right'])
                return action, 0.6

        else:  # Default behavior for any unhandled goals
            rand = np.random.random()
            if rand < 0.3:  # 30% examine
                return 'examine', 0.8
            elif rand < 0.4:  # 10% forage
                return 'forage', 0.6
            else:  # 60% move
                action = np.random.choice(['move_forward', 'turn_left', 'turn_right'])
                return action, 0.7

    def _detect_object_type(self, visual_field, object_type):
        """FIXED: Better object detection in visual field"""

        # More sensitive detection thresholds
        threshold_map = {
            'food': 0.25,      # Lower thresholds
            'water': 0.45,
            'shelter': 0.75,
            'obstacle': 0.9
        }

        threshold = threshold_map.get(object_type, 0.5)

        # Check multiple visual channels
        detected = (np.any(visual_field[:, :, 0] > threshold) or
                   np.any(visual_field[:, :, 1] > threshold) or
                   np.any(visual_field[:, :, 2] > threshold))

        return detected

    def _update_embodied_learning(self, visual_field, environment_feedback, cognitive_metrics):
        """Learn from embodied experience"""

        # Create embodied memory entry
        memory_entry = {
            'timestamp': len(self.embodied_memories),
            'visual_snapshot': visual_field.copy(),
            'body_state': self.body.state,
            'context': self.context.get_current(),
            'goal': self.goal.current_goal,
            'environment_feedback': environment_feedback,
            'surplus_mean': cognitive_metrics.get('surplus_mean', 0),
            'sigma_mean': cognitive_metrics.get('sigma_mean', 0)
        }

        self.embodied_memories.append(memory_entry)

        # Update perceptual categories based on experience
        if environment_feedback.get('outcome'):
            outcome = environment_feedback['outcome']
            visual_signature = np.mean(visual_field, axis=(0, 1))  # Simple visual signature

            self.perceptual_categories[outcome].append(visual_signature)

        # Store in regular memory system too
        self.memory.store(
            {
                'embodied_experience': memory_entry,
                'outcome': environment_feedback.get('outcome', 'exploration')
            },
            tags={'type': 'episodic', 'embodied': True}
        )

    def _calculate_embodied_reward(self, environment_feedback, current_goal):
        """Calculate reward based on embodied interaction"""

        base_reward = 0.0

        # Goal-specific rewards
        if current_goal == "seek_food" and 'consumed' in environment_feedback.get('outcome', ''):
            base_reward += 0.8
        elif current_goal == "find_shelter" and 'rested' in environment_feedback.get('outcome', ''):
            base_reward += 0.6
        elif current_goal == "explore_space":
            # Reward for discovering new objects
            base_reward += 0.2 if environment_feedback.get('outcome') else 0.1

        # Body state rewards
        if self.body.state.energy > 0.8:
            base_reward += 0.1
        if self.body.state.health > 0.9:
            base_reward += 0.1

        # Penalties
        if environment_feedback.get('collision_type'):
            base_reward -= 0.2

        return base_reward

def run_embodied_experiment(steps=1000, visualize=True):  # LONGER DURATION
    """Run embodied QSE-Émile experiment with FIXED parameters"""

    print("🤖 EMBODIED QSE-ÉMILE EXPERIMENT (FIXED VERSION)")
    print("=" * 50)
    print("Simulating embodied cognition in sensorimotor grid world")
    print("FIXES: Lower energy costs, better object detection, context switching enabled")

    # Create environment and agent
    environment = EmbodiedEnvironment(size=20)
    agent = EmbodiedQSEAgent()

    print(f"Environment created with {len(environment.objects)} objects")
    print(f"Context threshold: {agent.cfg.RECONTEXTUALIZATION_THRESHOLD}")

    # Tracking
    trajectory = []
    context_switches = []
    object_discoveries = []
    embodied_metrics = {
        'energy_over_time': [],
        'health_over_time': [],
        'context_over_time': [],
        'goal_over_time': [],
        'surplus_over_time': [],
        'sigma_over_time': []
    }

    print(f"Running {steps} embodied steps...")

    for step in range(steps):
        if step % 200 == 0:  # Less frequent updates for longer runs
            print(f"  Step {step}/{steps} - Energy: {agent.body.state.energy:.3f}, Discoveries: {len(object_discoveries)}")

        # Record pre-step state
        old_context = agent.context.get_current()

        # Execute embodied step
        result = agent.embodied_step(environment)

        # Track trajectory and metrics
        trajectory.append(agent.body.state.position)
        embodied_metrics['energy_over_time'].append(agent.body.state.energy)
        embodied_metrics['health_over_time'].append(agent.body.state.health)
        embodied_metrics['context_over_time'].append(result['context'])
        embodied_metrics['goal_over_time'].append(result['cognitive_metrics'].get('goal', None))
        embodied_metrics['surplus_over_time'].append(result['cognitive_metrics'].get('surplus_mean', 0))
        embodied_metrics['sigma_over_time'].append(result['cognitive_metrics'].get('sigma_mean', 0))

        # Track context switches
        if result['context'] != old_context:
            context_switches.append({
                'step': step,
                'position': agent.body.state.position,
                'old_context': old_context,
                'new_context': result['context'],
                'goal': agent.goal.current_goal,
                'body_energy': agent.body.state.energy
            })

        # Track object discoveries
        if result['environment_feedback'].get('outcome'):
            object_discoveries.append({
                'step': step,
                'position': agent.body.state.position,
                'outcome': result['environment_feedback']['outcome'],
                'context': result['context']
            })

    print(f"Experiment complete!")
    print(f"Context switches: {len(context_switches)}")
    print(f"Object discoveries: {len(object_discoveries)}")
    print(f"Final energy: {agent.body.state.energy:.3f}")

    # Analysis and visualization
    if visualize:
        create_embodied_visualization(environment, agent, trajectory, context_switches,
                                    object_discoveries, embodied_metrics)

    results = {
        'environment': environment,
        'agent': agent,
        'trajectory': trajectory,
        'context_switches': context_switches,
        'object_discoveries': object_discoveries,
        'embodied_metrics': embodied_metrics,
        'total_steps': steps
    }

    print_embodied_analysis(results)

    return results

def create_embodied_visualization(environment, agent, trajectory, context_switches,
                                object_discoveries, embodied_metrics):
    """Create comprehensive visualization of embodied cognition"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Embodied QSE-Émile: Sensorimotor Cognition', fontsize=16, fontweight='bold')

    # 1. Environment map with trajectory
    ax1 = axes[0, 0]

    # Show environment
    im1 = ax1.imshow(environment.grid, cmap='terrain', alpha=0.7)

    # Plot trajectory
    if trajectory:
        traj_x = [pos[1] for pos in trajectory]
        traj_y = [pos[0] for pos in trajectory]
        ax1.plot(traj_x, traj_y, 'r-', alpha=0.6, linewidth=2, label='Trajectory')

    # Mark context switches
    if context_switches:
        switch_x = [sw['position'][1] for sw in context_switches]
        switch_y = [sw['position'][0] for sw in context_switches]
        ax1.scatter(switch_x, switch_y, c='purple', s=50, marker='^',
                   label=f'Context Switches ({len(context_switches)})', alpha=0.8)

    # Mark object discoveries
    if object_discoveries:
        disc_x = [obj['position'][1] for obj in object_discoveries]
        disc_y = [obj['position'][0] for obj in object_discoveries]
        ax1.scatter(disc_x, disc_y, c='gold', s=80, marker='*',
                   label=f'Discoveries ({len(object_discoveries)})', alpha=0.9)

    # Mark final position
    final_pos = agent.body.state.position
    ax1.scatter(final_pos[1], final_pos[0], c='red', s=100, marker='X',
               label='Final Position', edgecolors='white', linewidth=2)

    ax1.set_title('Embodied Exploration & Discovery')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Body state over time
    ax2 = axes[0, 1]
    steps = range(len(embodied_metrics['energy_over_time']))

    ax2.plot(steps, embodied_metrics['energy_over_time'], 'g-', label='Energy', linewidth=2)
    ax2.plot(steps, embodied_metrics['health_over_time'], 'b-', label='Health', linewidth=2)

    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Body State (0-1)')
    ax2.set_title('Body State Dynamics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. QSE dynamics in embodied context
    ax3 = axes[0, 2]

    ax3.plot(steps, embodied_metrics['surplus_over_time'], 'r-', label='Surplus', alpha=0.7)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(steps, embodied_metrics['sigma_over_time'], 'orange', label='Sigma', alpha=0.7)

    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Surplus', color='red')
    ax3_twin.set_ylabel('Sigma', color='orange')
    ax3.set_title('QSE Dynamics During Embodied Experience')
    ax3.grid(True, alpha=0.3)

    # 4. Context switches over time
    ax4 = axes[1, 0]

    if context_switches:
        switch_steps = [sw['step'] for sw in context_switches]
        switch_contexts = [sw['new_context'] for sw in context_switches]

        ax4.scatter(switch_steps, switch_contexts, c='purple', s=60, alpha=0.7)
        ax4.plot(steps, embodied_metrics['context_over_time'], 'k-', alpha=0.5, linewidth=1)

    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Context ID')
    ax4.set_title('Context Evolution During Embodied Experience')
    ax4.grid(True, alpha=0.3)

    # 5. Discovery timeline
    ax5 = axes[1, 1]

    if object_discoveries:
        discovery_types = [obj['outcome'] for obj in object_discoveries]
        discovery_steps = [obj['step'] for obj in object_discoveries]

        # Count discoveries by type
        from collections import Counter
        disc_counts = Counter(discovery_types)

        # Plot discovery events
        for i, (discovery_type, count) in enumerate(disc_counts.items()):
            type_steps = [step for step, outcome in zip(discovery_steps, discovery_types)
                         if outcome == discovery_type]
            ax5.scatter(type_steps, [i] * len(type_steps),
                       label=f'{discovery_type} ({count})', s=50, alpha=0.7)

        ax5.set_xlabel('Time Steps')
        ax5.set_ylabel('Discovery Type')
        ax5.set_title('Object Discovery Timeline')
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, 'No discoveries recorded', ha='center', va='center',
                transform=ax5.transAxes)
        ax5.set_title('Object Discovery Timeline')

    ax5.grid(True, alpha=0.3)

    # 6. Learning progression
    ax6 = axes[1, 2]

    # Show perceptual category formation
    if agent.perceptual_categories:
        category_counts = {cat: len(examples) for cat, examples in agent.perceptual_categories.items()}
        categories = list(category_counts.keys())
        counts = list(category_counts.values())

        bars = ax6.bar(range(len(categories)), counts, alpha=0.7, color='skyblue')
        ax6.set_xticks(range(len(categories)))
        ax6.set_xticklabels(categories, rotation=45, ha='right')
        ax6.set_ylabel('Examples Learned')
        ax6.set_title('Perceptual Category Formation')

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
    else:
        ax6.text(0.5, 0.5, 'No categories formed', ha='center', va='center',
                transform=ax6.transAxes)
        ax6.set_title('Perceptual Category Formation')

    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('embodied_qse_emile_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("📊 Embodied cognition visualization saved as 'embodied_qse_emile_simulation.png'")

def print_embodied_analysis(results):
    """Print analysis of embodied cognition experiment"""

    print(f"\n🧠 EMBODIED COGNITION ANALYSIS")
    print("=" * 50)

    agent = results['agent']
    context_switches = results['context_switches']
    object_discoveries = results['object_discoveries']
    trajectory = results['trajectory']

    # Basic metrics
    print(f"Total Steps: {results['total_steps']}")
    print(f"Context Switches: {len(context_switches)}")
    print(f"Object Discoveries: {len(object_discoveries)}")
    print(f"Unique Positions Visited: {len(set(trajectory))}")

    # Body state analysis
    final_energy = agent.body.state.energy
    final_health = agent.body.state.health
    print(f"Final Energy: {final_energy:.3f}")
    print(f"Final Health: {final_health:.3f}")

    # Perceptual learning
    categories_learned = len(agent.perceptual_categories)
    total_examples = sum(len(examples) for examples in agent.perceptual_categories.values())
    print(f"Perceptual Categories Formed: {categories_learned}")
    print(f"Total Perceptual Examples: {total_examples}")

    if agent.perceptual_categories:
        print("Categories learned:")
        for category, examples in agent.perceptual_categories.items():
            print(f"  - {category}: {len(examples)} examples")

    # Context switching analysis
    if context_switches:
        context_reasons = []
        for switch in context_switches:
            if switch['body_energy'] < 0.3:
                context_reasons.append('low_energy')
            elif 'consumed' in str(switch.get('goal', '')):
                context_reasons.append('food_seeking')
            else:
                context_reasons.append('exploration')

        from collections import Counter
        reason_counts = Counter(context_reasons)
        print("Context switch triggers:")
        for reason, count in reason_counts.items():
            print(f"  - {reason}: {count} switches")

    # Discovery analysis
    if object_discoveries:
        discovery_contexts = [disc['context'] for disc in object_discoveries]
        context_discovery_rate = {}
        for context_id in set(discovery_contexts):
            rate = discovery_contexts.count(context_id) / len(object_discoveries)
            context_discovery_rate[context_id] = rate

        print("Discovery rate by context:")
        for context_id, rate in sorted(context_discovery_rate.items()):
            print(f"  - Context {context_id}: {rate:.2%}")

    # Embodied learning insights
    print(f"\n🔍 EMBODIED INSIGHTS:")

    # Body schema development
    boundary_experiences = len(agent.body.body_schema['boundaries'])
    affordance_contexts = len(agent.body.body_schema['affordances'])
    sensorimotor_mappings = len(agent.body.body_schema['sensorimotor_mappings'])

    print(f"✅ Body Schema Development:")
    print(f"   - Boundary experiences: {boundary_experiences}")
    print(f"   - Affordance contexts learned: {affordance_contexts}")
    print(f"   - Sensorimotor mappings: {sensorimotor_mappings}")

    # Context-driven perception
    context_switches_near_discoveries = 0
    for switch in context_switches:
        for discovery in object_discoveries:
            if abs(switch['step'] - discovery['step']) <= 5:
                context_switches_near_discoveries += 1
                break

    if context_switches and object_discoveries:
        context_discovery_correlation = context_switches_near_discoveries / len(context_switches)
        print(f"✅ Context-Discovery Correlation: {context_discovery_correlation:.2%}")

    # Exploration efficiency
    exploration_efficiency = len(set(trajectory)) / len(trajectory)
    print(f"✅ Exploration Efficiency: {exploration_efficiency:.3f}")

    print(f"\n🎯 EMBODIED COGNITION SUCCESS METRICS:")
    if categories_learned > 0:
        print(f"✅ Emergent categorization achieved ({categories_learned} categories)")
    if boundary_experiences > 5:
        print(f"✅ Body schema formation in progress ({boundary_experiences} experiences)")
    if len(context_switches) > 0:
        print(f"✅ Context-dependent perception active ({len(context_switches)} switches)")
    if object_discoveries:
        print(f"✅ Meaningful world interaction ({len(object_discoveries)} discoveries)")

def main():
    """Run embodied QSE-Émile experiment"""

    results = run_embodied_experiment(steps=1000, visualize=True)  # Use longer duration
    return results

if __name__ == "__main__":
    results = main()
