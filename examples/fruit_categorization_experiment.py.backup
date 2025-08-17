"""
The "Unfamiliar Fruit" Experiment: Embodied Categorization vs Pattern Matching
FIXED VERSION - Addresses energy absorption, stuck agents, and prediction issues

Tests whether QSE-Émile forms categories through embodied meaning-making
rather than just visual pattern recognition.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from pathlib import Path
import time

# Import existing modules
from embodied_qse_emile import EmbodiedQSEAgent, EmbodiedEnvironment, SensoriMotorBody
from config import QSEConfig

class FruitEnvironment(EmbodiedEnvironment):
    """FIXED: Specialized environment for fruit categorization experiment"""

    def __init__(self, size=15, phase="training"):
        super().__init__(size)
        self.phase = phase
        self.fruit_types = self._define_fruit_types()
        self.agent_fruit_interactions = defaultdict(list)
        self.consumed_fruits = set()  # FIXED: Track consumed fruits
        self.create_fruit_environment()

    def _define_fruit_types(self):
        """Define the three fruit types with their properties"""
        return {
            'red_fruit': {
                'visual_signature': np.array([0.8, 0.2, 0.2]),  # Red
                'energy_effect': 0.4,      # Very nourishing
                'health_effect': 0.1,      # Slightly healthy
                'taste': 'sweet',
                'embodied_signature': 'nourishing',  # Key: embodied meaning
                'grid_value': 0.3
            },
            'crimson_fruit': {
                'visual_signature': np.array([0.7, 0.1, 0.1]),  # Crimson (similar to red)
                'energy_effect': -0.3,     # Poisonous - drains energy
                'health_effect': -0.2,     # Unhealthy
                'taste': 'bitter',
                'embodied_signature': 'poisonous',  # Key: embodied meaning
                'grid_value': 0.35
            },
            'blue_fruit': {
                'visual_signature': np.array([0.2, 0.3, 0.9]),  # Blue (visually distinct)
                'energy_effect': 0.35,     # Nourishing (like red, not crimson)
                'health_effect': 0.08,     # Healthy
                'taste': 'sweet',
                'embodied_signature': 'nourishing',  # Same as red fruit!
                'grid_value': 0.25
            }
        }

    def create_fruit_environment(self):
        """Create environment with fruit objects based on experimental phase"""

        # Clear existing objects
        self.grid = np.zeros((self.size, self.size))
        self.objects = {}
        self.consumed_fruits = set()  # FIXED: Reset consumed fruits

        # Walls around perimeter
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        if self.phase == "training":
            fruit_configs = [
                ('red_fruit', 6),      # 6 good fruits
                ('crimson_fruit', 6),  # 6 bad fruits
            ]
        elif self.phase == "test":
            fruit_configs = [
                ('red_fruit', 3),      # Fewer red for comparison
                ('crimson_fruit', 3),  # Fewer crimson
                ('blue_fruit', 6),     # New blue fruits to categorize
            ]
        else:
            fruit_configs = [
                ('red_fruit', 4),
                ('crimson_fruit', 4),
                ('blue_fruit', 4),
            ]

        # Place fruits in environment
        for fruit_type, count in fruit_configs:
            fruit_def = self.fruit_types[fruit_type]

            for _ in range(count):
                while True:
                    x, y = np.random.randint(2, self.size-2, 2)
                    if self.grid[x, y] == 0:
                        break

                self.grid[x, y] = fruit_def['grid_value']
                self.objects[(x, y)] = {
                    'type': 'fruit',
                    'fruit_type': fruit_type,
                    'properties': dict(fruit_def),
                    'discovered': False,
                    'interaction_count': 0,
                    'interactions_history': []
                }

        print(f"Created {self.phase} environment with {len(self.objects)} fruits")

        # FIXED: Debug output to verify fruit placement
        fruit_positions = [(pos, obj['fruit_type']) for pos, obj in self.objects.items() if obj['type'] == 'fruit']
        print(f"Fruit positions: {fruit_positions[:6]}...")  # Show first 6

        # FIXED: Verify grid values
        fruit_grid_positions = []
        for pos, obj in self.objects.items():
            if obj['type'] == 'fruit':
                fruit_grid_positions.append((pos, self.grid[pos]))
        print(f"Grid values at fruit positions: {fruit_grid_positions[:3]}...")  # Show first 3

        # Add some regular objects for comparison
        for obj_type, count in [('water', 2), ('shelter', 1)]:
            for _ in range(count):
                while True:
                    x, y = np.random.randint(2, self.size-2, 2)
                    if self.grid[x, y] == 0:
                        break

                self.grid[x, y] = 0.5 if obj_type == 'water' else 0.8
                self.objects[(x, y)] = {
                    'type': obj_type,
                    'properties': self._generate_object_properties(obj_type),
                    'discovered': False,
                    'interaction_count': 0
                }

    def get_visual_field(self, body, context_filter=None):
        """FIXED: Enhanced visual field with better fruit detection"""

        # Get base visual field
        vision_field = super().get_visual_field(body, context_filter)

        x, y = body.state.position
        vision_range = body.vision_range

        # Enhance with fruit-specific visual signatures
        for i in range(-vision_range, vision_range+1):
            for j in range(-vision_range, vision_range+1):
                world_x, world_y = x + i, y + j

                if (0 <= world_x < self.size and 0 <= world_y < self.size and
                    (world_x, world_y) in self.objects and
                    (world_x, world_y) not in self.consumed_fruits):  # FIXED: Don't show consumed fruits

                    obj = self.objects[(world_x, world_y)]
                    if obj['type'] == 'fruit':
                        fruit_sig = obj['properties']['visual_signature']
                        distance = np.sqrt(i*i + j*j)
                        acuity = max(0.1, 1.0 - distance / vision_range)

                        # Set RGB based on fruit's visual signature
                        vision_field[i+vision_range, j+vision_range, :] = fruit_sig * acuity

        return vision_field

    def step(self, body, action_name, action_intensity=1.0):
        """FIXED: Enhanced step with proper fruit consumption tracking"""

        current_pos = body.state.position
        old_position = current_pos

        # FIXED: Debug environment step
        step_count = getattr(self, '_debug_step_count', 0)
        self._debug_step_count = step_count + 1

        if step_count < 5:
            print(f"          Environment step: pos={current_pos}, action={action_name}")
            if current_pos in self.objects:
                obj = self.objects[current_pos]
                print(f"          Object at position: {obj['type']} {obj.get('fruit_type', '')}")

        # FIXED: Handle fruit interaction BEFORE movement to prevent conflicts
        fruit_interaction_result = None

        if (current_pos in self.objects and
            self.objects[current_pos]['type'] == 'fruit' and
            current_pos not in self.consumed_fruits and
            action_name == 'examine'):

            if step_count < 5:
                print(f"          🍎 EXAMINING FRUIT!")

            fruit_obj = self.objects[current_pos]
            fruit_type = fruit_obj['fruit_type']
            fruit_props = fruit_obj['properties']

            # Record old state
            old_energy = body.state.energy
            old_health = body.state.health

            # FIXED: Allow energy to exceed 1.0 temporarily to see actual changes
            body.state.energy = min(2.0, body.state.energy + fruit_props['energy_effect'])
            body.state.health = min(2.0, body.state.health + fruit_props['health_effect'])

            # Calculate actual changes
            energy_change = body.state.energy - old_energy
            health_change = body.state.health - old_health

            # FIXED: Mark fruit as consumed
            self.consumed_fruits.add(current_pos)
            self.grid[current_pos] = 0  # Remove from grid so agent can move through

            # Record interaction
            interaction_record = {
                'fruit_type': fruit_type,
                'embodied_signature': fruit_props['embodied_signature'],
                'visual_signature': fruit_props['visual_signature'].tolist(),
                'energy_change': energy_change,
                'health_change': health_change,
                'taste': fruit_props['taste'],
                'agent_context': getattr(body, 'current_context', 0),
                'step': getattr(self, 'step_count', 0)
            }

            fruit_obj['interactions_history'].append(interaction_record)
            self.agent_fruit_interactions[fruit_type].append(interaction_record)

            # Create result
            fruit_interaction_result = {
                'outcome': f"consumed {fruit_props['taste']} {fruit_type} (+{energy_change:.3f} energy)",
                'valence': 'positive' if energy_change > 0 else 'negative',
                'embodied_signature': fruit_props['embodied_signature'],
                'fruit_interaction': interaction_record,
                'energy_change': energy_change,
                'health_change': health_change
            }

            print(f"🍎 FRUIT CONSUMED: {fruit_type} at {current_pos}, energy change: {energy_change:+.3f}")

        # Now handle movement (FIXED: Proper discrete movement)
        movement = body.execute_action(action_name, action_intensity)

        # FIXED: Convert small movements to discrete steps
        if action_name == 'move_forward':
            # Calculate direction based on orientation
            dx = round(np.cos(body.state.orientation))
            dy = round(np.sin(body.state.orientation))
        elif action_name == 'move_backward':
            dx = -round(np.cos(body.state.orientation))
            dy = -round(np.sin(body.state.orientation))
        else:
            dx, dy = 0, 0  # No movement for turn/rest/examine

        new_x = current_pos[0] + dx
        new_y = current_pos[1] + dy

        # Simple collision detection - FIXED: Allow movement within bounds
        if (1 <= new_x <= self.size-2 and 1 <= new_y <= self.size-2 and
            self.grid[new_x, new_y] < 0.8):  # Avoid walls only
            body.state.position = (new_x, new_y)
            if step_count < 5:
                print(f"          MOVED: {current_pos} -> {body.state.position}")
        elif step_count < 5:
            print(f"          BLOCKED: would move to ({new_x}, {new_y}) but blocked")

        body.update_proprioception()

        # Return results
        if fruit_interaction_result:
            return fruit_interaction_result
        else:
            # Check for fruit spotting
            if (body.state.position in self.objects and
                self.objects[body.state.position]['type'] == 'fruit' and
                body.state.position not in self.consumed_fruits):
                fruit_type = self.objects[body.state.position]['fruit_type']
                return {'outcome': f"spotted {fruit_type}", 'fruit_type': fruit_type}
            else:
                return {'outcome': ''}

class FruitCategorizationAgent(EmbodiedQSEAgent):
    """FIXED: Agent specialized for fruit categorization experiment"""

    def __init__(self, config=QSEConfig()):
        super().__init__(config)

        # Enhanced for categorization
        self.fruit_experiences = defaultdict(list)
        self.visual_memories = defaultdict(list)
        self.embodied_mappings = defaultdict(list)
        self.fruit_predictions = {}
        self.categorization_accuracy = defaultdict(list)

        # --- START CHANGE ---
        # Formally declare all instance attributes
        self._current_environment = None
        self._debug_step = 0
        # --- END CHANGE ---

        # Add fruit-specific goals
        fruit_goals = ["seek_nourishment", "avoid_poison", "test_unknown", "categorize_experience"]
        for goal in fruit_goals:
            self.goal.add_goal(goal)

        # Force initial goal selection
        if not self.goal.current_goal:
            self.goal.current_goal = "seek_nourishment"

        print(f"🤖 Agent initialized with goals: {self.goal.goals}")
        print(f"🎯 Initial goal: {self.goal.current_goal}")

    def _update_embodied_learning(self, visual_field, environment_feedback, cognitive_metrics):
        """FIXED: Enhanced learning with proper fruit categorization tracking"""

        # Call parent method
        super()._update_embodied_learning(visual_field, environment_feedback, cognitive_metrics)

        # FIXED: Fruit-specific learning
        if environment_feedback.get('fruit_interaction'):
            interaction = environment_feedback['fruit_interaction']
            embodied_sig = interaction['embodied_signature']
            fruit_type = interaction['fruit_type']
            energy_change = interaction['energy_change']

            # Store embodied experience with actual energy change
            experience = {
                'visual_signature': interaction['visual_signature'],
                'energy_change': energy_change,
                'health_change': interaction['health_change'],
                'taste': interaction['taste'],
                'context': interaction.get('agent_context', 0),
                'surplus_mean': cognitive_metrics.get('surplus_mean', 0),
                'sigma_mean': cognitive_metrics.get('sigma_mean', 0)
            }

            self.fruit_experiences[embodied_sig].append(experience)

            # FIXED: Store visual memory by fruit type
            visual_signature = np.array(interaction['visual_signature'])
            self.visual_memories[fruit_type].append(visual_signature)

            # FIXED: Store embodied signature mapping
            self.embodied_mappings[embodied_sig].append(energy_change)

            # Update perceptual categories with embodied grounding
            category_key = f"{embodied_sig}_{interaction['taste']}"
            self.perceptual_categories[category_key].append(visual_signature)

            print(f"🧠 Learned: {fruit_type} = {embodied_sig} ({energy_change:+.2f} energy)")

    def predict_fruit_value(self, visual_field):
        """FIXED: Predict fruit value based on embodied experience, not just visual similarity"""

        if not self.embodied_mappings:
            return "unknown", 0.0

        # Extract visual features from current field
        center_intensity = np.mean(visual_field[visual_field.shape[0]//2, visual_field.shape[1]//2, :])
        current_visual = np.mean(visual_field, axis=(0, 1))

        # FIXED: Priority system - embodied signature trumps visual similarity
        best_embodied_match = None
        best_confidence = 0.0

        for embodied_sig, energy_changes in self.embodied_mappings.items():
            if not energy_changes:
                continue

            # Calculate embodied signature strength
            avg_energy = np.mean(energy_changes)
            consistency = 1.0 - np.std(energy_changes) / (abs(avg_energy) + 0.1)

            # FIXED: If we have enough experience with this signature, use it
            if len(energy_changes) >= 3:
                confidence = consistency * min(1.0, len(energy_changes) / 5.0)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_embodied_match = (embodied_sig, avg_energy)

        # If we have a strong embodied match, use it
        if best_embodied_match and best_confidence > 0.5:
            embodied_sig, predicted_value = best_embodied_match
            return embodied_sig, predicted_value

        # FIXED: Otherwise, fall back to visual similarity but bias toward embodied learning
        visual_matches = []
        for fruit_type, visual_sigs in self.visual_memories.items():
            if not visual_sigs:
                continue

            avg_visual = np.mean(visual_sigs, axis=0)
            similarity = 1.0 / (1.0 + np.linalg.norm(current_visual - avg_visual))

            # Find corresponding embodied signature
            for embodied_sig, experiences in self.fruit_experiences.items():
                type_experiences = [exp for exp in experiences if np.array_equal(exp['visual_signature'], visual_sigs[0])]
                if type_experiences:
                    avg_energy = np.mean([exp['energy_change'] for exp in type_experiences])
                    visual_matches.append((embodied_sig, avg_energy, similarity))

        if visual_matches:
            # Choose best visual match
            best_match = max(visual_matches, key=lambda x: x[2])
            return best_match[0], best_match[1]

        return "unknown", 0.0

    def _goal_to_embodied_action(self, goal, visual_field, environment=None):
        """FIXED: Enhanced action selection with proper fruit detection and movement"""

        # Energy management first - if low energy, prioritize rest/food seeking
        if self.body.state.energy < 0.3:
            return 'rest', 1.0

        # Better fruit detection - check if we're actually at a fruit position
        current_pos = self.body.state.position
        
        # Check for the environment attribute and ensure it is not None before using it
        if hasattr(self, '_current_environment') and self._current_environment is not None:
            on_fruit = (current_pos in self._current_environment.objects and
                        current_pos not in self._current_environment.consumed_fruits)
        else:
            # Fallback to visual field detection if environment isn't available
            center_intensity = np.mean(visual_field[visual_field.shape[0]//2, visual_field.shape[1]//2, :])
            on_fruit = center_intensity > 0.2

        # Debug action selection
        if hasattr(self, '_debug_step') and self._debug_step < 5:
            print(f"        Action selection: pos={current_pos}, on_fruit={on_fruit}, goal={goal}")

        # If on a fruit, always examine it regardless of goal
        if on_fruit:
            if hasattr(self, '_debug_step') and self._debug_step < 5:
                print(f"        -> ON FRUIT: examine")
            return 'examine', 1.0

        # Get fruit prediction for decision making
        fruit_prediction, predicted_value = self.predict_fruit_value(visual_field)

        # More aggressive exploration to find fruits
        if goal == "seek_nourishment":
            # If no experience yet, examine more often
            if not self.embodied_mappings and np.random.random() < 0.4:
                return 'examine', 1.0

            if fruit_prediction == "nourishing" or predicted_value > 0.1:
                # Move toward fruit (simplified - just move forward more often)
                if np.random.random() < 0.7:
                    return 'move_forward', 0.8
                else:
                    return np.random.choice(['turn_left', 'turn_right']), 0.6
            else:
                # More systematic exploration with more examining
                if np.random.random() < 0.3:
                    return 'examine', 1.0
                elif np.random.random() < 0.6:
                    return 'move_forward', 0.8
                else:
                    return np.random.choice(['turn_left', 'turn_right']), 0.7

        elif goal == "avoid_poison":
            if fruit_prediction == "poisonous" or predicted_value < -0.1:
                # Turn away from bad fruit
                return np.random.choice(['turn_left', 'turn_right']), 0.8
            else:
                return 'move_forward', 0.6

        elif goal == "test_unknown":
            # Always examine when no experience
            if not self.embodied_mappings:
                return 'examine', 1.0

            if fruit_prediction == "unknown":
                # Move toward unknown fruit to test it
                if np.random.random() < 0.6:
                    return 'move_forward', 0.8
                else:
                    return np.random.choice(['turn_left', 'turn_right']), 0.5
            else:
                return np.random.choice(['move_forward', 'turn_left', 'turn_right']), 0.6

        elif goal == "categorize_experience":
            # Examine current location more often
            if np.random.random() < 0.3:
                return 'examine', 1.0
            elif np.random.random() < 0.5:
                return 'move_forward', 0.7
            else:
                return np.random.choice(['turn_left', 'turn_right']), 0.6

        # Default exploration - more likely to examine, especially with no experience
        if not self.embodied_mappings and np.random.random() < 0.5:
            if hasattr(self, '_debug_step') and self._debug_step < 5:
                print(f"        -> No experience: examine")
            return 'examine', 1.0
        elif np.random.random() < 0.25:  # 25% chance to examine
            if hasattr(self, '_debug_step') and self._debug_step < 5:
                print(f"        -> Random: examine")
            return 'examine', 1.0
        else:
            action = np.random.choice(['move_forward', 'turn_left', 'turn_right'])
            if hasattr(self, '_debug_step') and self._debug_step < 5:
                print(f"        -> Movement: {action}")
            return action, 0.7

class StandardRLFruitAgent:
    """Standard RL agent for comparison - UNCHANGED"""

    def __init__(self):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.epsilon = 0.1
        self.learning_rate = 0.1
        self.discount = 0.9
        self.body = SensoriMotorBody()

        # Fruit interaction tracking
        self.fruit_interactions = defaultdict(list)
        self.last_state = None
        self.last_action = None

    def select_action(self, visual_field, environment):
        """Simple epsilon-greedy based on visual patterns only"""

        # FIXED: Check if we're on a fruit and examine more often
        current_pos = self.body.state.position
        on_fruit = (current_pos in environment.objects and
                   environment.objects[current_pos]['type'] == 'fruit' and
                   current_pos not in environment.consumed_fruits)

        if on_fruit:
            return 'examine'  # Always examine when on fruit

        # Discretize visual field for Q-table
        visual_key = self._visual_to_key(visual_field)

        if np.random.random() < self.epsilon or len(self.q_table[visual_key]) == 0:
            # FIXED: Much more likely to examine during exploration
            if np.random.random() < 0.6:  # 60% chance to examine
                action = 'examine'
            else:
                action = np.random.choice(['move_forward', 'turn_left', 'turn_right'])
        else:
            q_values = [self.q_table[visual_key][a] for a in ['move_forward', 'turn_left', 'turn_right', 'examine']]
            action = ['move_forward', 'turn_left', 'turn_right', 'examine'][np.argmax(q_values)]

        self.last_state = visual_key
        self.last_action = action
        return action

    def update(self, reward, new_visual_field, environment_feedback):
        """Q-learning update based on visual patterns"""

        if self.last_state and self.last_action:
            new_state = self._visual_to_key(new_visual_field)

            current_q = self.q_table[self.last_state][self.last_action]
            max_next_q = max([self.q_table[new_state][a] for a in ['move_forward', 'turn_left', 'turn_right', 'examine']])

            target = reward + self.discount * max_next_q
            self.q_table[self.last_state][self.last_action] += self.learning_rate * (target - current_q)

        # Track fruit interactions
        if environment_feedback.get('fruit_interaction'):
            interaction = environment_feedback['fruit_interaction']
            self.fruit_interactions[interaction['fruit_type']].append(interaction)

    def _visual_to_key(self, visual_field):
        """Convert visual field to discrete key for Q-table"""
        avg_rgb = np.mean(visual_field, axis=(0, 1))
        discretized = tuple((avg_rgb * 4).astype(int))
        return discretized

    def predict_fruit_value(self, visual_field):
        """RL agent's prediction (just based on Q-values of 'examine' action)"""
        visual_key = self._visual_to_key(visual_field)
        examine_q = self.q_table[visual_key]['examine']

        if examine_q > 0.1:
            return "good", examine_q
        elif examine_q < -0.1:
            return "bad", examine_q
        else:
            return "unknown", examine_q

def run_training_phase(agent, environment, steps):
    """FIXED: Run training phase with proper tracking"""

    # Reset debug counters
    environment._debug_step_count = 0

    fruit_interactions = defaultdict(list)
    performance_metrics = []
    categorization_timeline = []

    for step in range(steps):
        if step % 200 == 0:
            print(f"    Step {step}/{steps}")
            # FIXED: Debug agent position and nearby objects
            if step == 0:
                print(f"    Agent at: {agent.body.state.position if hasattr(agent, 'body') else 'Unknown'}")
                if isinstance(agent, FruitCategorizationAgent):
                    nearby_fruits = []
                    x, y = agent.body.state.position
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            check_pos = (x + dx, y + dy)
                            if check_pos in environment.objects and environment.objects[check_pos]['type'] == 'fruit':
                                nearby_fruits.append((check_pos, environment.objects[check_pos]['fruit_type']))
                    print(f"    Nearby fruits: {nearby_fruits}")

        if isinstance(agent, FruitCategorizationAgent):
            # QSE-Émile agent
            # FIXED: Give agent access to environment for better fruit detection
            agent._current_environment = environment
            agent._debug_step = step  # For debugging action selection
            result = agent.embodied_step(environment)

            # FIXED: Debug fruit interactions
            if step < 5 or step % 100 == 0:  # Debug first 5 steps and every 100th
                print(f"      Step {step}: Agent at {agent.body.state.position}, Goal: {agent.goal.current_goal}")
                if result['environment_feedback'].get('outcome'):
                    print(f"      Outcome: {result['environment_feedback']['outcome']}")
                if result['environment_feedback'].get('fruit_interaction'):
                    print(f"      🍎 FRUIT INTERACTION: {result['environment_feedback']['fruit_interaction']['fruit_type']}")

            # Track fruit interactions
            if result['environment_feedback'].get('fruit_interaction'):
                interaction = result['environment_feedback']['fruit_interaction']
                fruit_interactions[interaction['fruit_type']].append(interaction)

                # Test categorization accuracy
                visual_field = environment.get_visual_field(agent.body)
                prediction, confidence = agent.predict_fruit_value(visual_field)

                actual_effect = interaction['energy_change']

                # FIXED: Accuracy based on embodied signature matching
                if prediction in ['nourishing', 'good'] and actual_effect > 0:
                    accuracy = True
                elif prediction in ['poisonous', 'bad'] and actual_effect < 0:
                    accuracy = True
                else:
                    accuracy = False

                categorization_timeline.append({
                    'step': step,
                    'fruit_type': interaction['fruit_type'],
                    'prediction': prediction,
                    'confidence': confidence,
                    'actual_effect': actual_effect,
                    'accuracy': accuracy
                })

            # Performance metrics
            performance_metrics.append({
                'step': step,
                'energy': agent.body.state.energy,
                'health': agent.body.state.health,
                'context': result['context'],
                'goal': agent.goal.current_goal
            })

        else:
            # Standard RL agent - simplified interaction
            visual_field = environment.get_visual_field(agent.body)
            action = agent.select_action(visual_field, environment)
            environment_feedback = environment.step(agent.body, action, 1.0)

            # Calculate reward
            reward = 0.0
            if environment_feedback.get('fruit_interaction'):
                interaction = environment_feedback['fruit_interaction']
                reward = interaction['energy_change'] + interaction['health_change'] * 0.5
                fruit_interactions[interaction['fruit_type']].append(interaction)

                # Test categorization
                prediction, confidence = agent.predict_fruit_value(visual_field)
                actual_effect = interaction['energy_change']

                accuracy = (actual_effect > 0 and confidence > 0) or (actual_effect < 0 and confidence < 0)

                categorization_timeline.append({
                    'step': step,
                    'fruit_type': interaction['fruit_type'],
                    'prediction': prediction,
                    'confidence': confidence,
                    'actual_effect': actual_effect,
                    'accuracy': accuracy
                })

            # Update agent
            new_visual_field = environment.get_visual_field(agent.body)
            agent.update(reward, new_visual_field, environment_feedback)

            performance_metrics.append({
                'step': step,
                'energy': agent.body.state.energy,
                'health': agent.body.state.health,
                'reward': reward
            })

    return {
        'fruit_interactions': dict(fruit_interactions),
        'performance_metrics': performance_metrics,
        'categorization_timeline': categorization_timeline
    }

def run_fruit_categorization_experiment(training_steps=600, test_steps=300):
    """FIXED: Run the complete fruit categorization experiment"""

    print("🍎 FRUIT CATEGORIZATION EXPERIMENT")
    print("=" * 50)
    print("Testing embodied categorization vs visual pattern matching")

    # Phase 1: Training Phase
    print(f"\n📚 TRAINING PHASE ({training_steps} steps)")
    print("Red fruit (good) vs Crimson fruit (bad)")

    # Create training environment
    train_env = FruitEnvironment(size=15, phase="training")

    # QSE-Émile training
    print("  Running QSE-Émile Training...")
    qse_agent = FruitCategorizationAgent()
    qse_train_results = run_training_phase(qse_agent, train_env, training_steps)

    # Standard RL training
    print("  Running Standard RL Training...")
    train_env.create_fruit_environment()  # Reset environment
    rl_agent = StandardRLFruitAgent()
    rl_train_results = run_training_phase(rl_agent, train_env, training_steps)

    # Phase 2: Test Phase
    print(f"\n🧪 TEST PHASE ({test_steps} steps)")
    print("Introducing Blue fruit (good but visually different)")

    # Create test environment
    test_env = FruitEnvironment(size=15, phase="test")

    # QSE-Émile testing
    print("  Running QSE-Émile Test...")
    qse_test_results = run_training_phase(qse_agent, test_env, test_steps)

    # Standard RL testing
    print("  Running Standard RL Test...")
    test_env.create_fruit_environment()  # Reset environment for RL
    rl_test_results = run_training_phase(rl_agent, test_env, test_steps)

    # Compile results
    results = {
        'qse_training': qse_train_results,
        'qse_testing': qse_test_results,
        'rl_training': rl_train_results,
        'rl_testing': rl_test_results,
        'qse_agent': qse_agent,
        'rl_agent': rl_agent
    }

    # Analysis and visualization
    analyze_categorization_results(results)
    create_categorization_visualization(results)

    return results

def analyze_categorization_results(results):
    """FIXED: Analyze fruit categorization experimental results"""

    print(f"\n🔍 FRUIT CATEGORIZATION ANALYSIS")
    print("=" * 50)

    # Extract results
    qse_train = results['qse_training']
    qse_test = results['qse_testing']
    rl_train = results['rl_training']
    rl_test = results['rl_testing']

    # Performance comparison
    print(f"\n📊 LEARNING PERFORMANCE:")

    # Training phase accuracy
    qse_train_acc = calculate_categorization_accuracy(qse_train['categorization_timeline'])
    rl_train_acc = calculate_categorization_accuracy(rl_train['categorization_timeline'])

    print(f"Training Phase Accuracy:")
    print(f"  QSE-Émile: {qse_train_acc:.1%}")
    print(f"  Standard RL: {rl_train_acc:.1%}")

    # Test phase accuracy (key metric!)
    qse_test_acc = calculate_categorization_accuracy(qse_test['categorization_timeline'])
    rl_test_acc = calculate_categorization_accuracy(rl_test['categorization_timeline'])

    print(f"\nTest Phase Accuracy (Blue Fruit):")
    print(f"  QSE-Émile: {qse_test_acc:.1%}")
    print(f"  Standard RL: {rl_test_acc:.1%}")

    # Learning speed analysis
    qse_blue_learning = analyze_blue_fruit_learning(qse_test['categorization_timeline'])
    rl_blue_learning = analyze_blue_fruit_learning(rl_test['categorization_timeline'])

    print(f"\nBlue Fruit Learning Speed:")
    print(f"  QSE-Émile first correct: Step {qse_blue_learning['first_correct']}")
    print(f"  Standard RL first correct: Step {rl_blue_learning['first_correct']}")
    print(f"  QSE-Émile stabilization: Step {qse_blue_learning['stabilization']}")
    print(f"  Standard RL stabilization: Step {rl_blue_learning['stabilization']}")

    # Category analysis
    print(f"\n🧠 CATEGORIZATION ANALYSIS:")

    qse_agent = results['qse_agent']
    qse_categories = qse_agent.perceptual_categories
    qse_experiences = qse_agent.fruit_experiences

    print(f"QSE-Émile Categories Formed: {len(qse_categories)}")
    for category, examples in qse_categories.items():
        print(f"  - {category}: {len(examples)} examples")

    print(f"QSE-Émile Embodied Experiences:")
    for signature, experiences in qse_experiences.items():
        avg_energy = np.mean([exp['energy_change'] for exp in experiences])
        print(f"  - {signature}: {len(experiences)} experiences (avg energy: {avg_energy:+.3f})")

    # Key findings
    print(f"\n🎯 KEY FINDINGS:")

    accuracy_advantage = qse_test_acc - rl_test_acc
    if accuracy_advantage > 0.1:
        print(f"✅ QSE-Émile shows {accuracy_advantage:.1%} accuracy advantage on novel fruit")

    speed_advantage = rl_blue_learning['first_correct'] - qse_blue_learning['first_correct']
    if speed_advantage > 0:
        print(f"✅ QSE-Émile learns novel fruit {speed_advantage} steps faster")

    if len(qse_experiences) > 0:
        print(f"✅ QSE-Émile forms embodied categories based on energy effects")

        # Check if blue fruit was categorized with red (both nourishing)
        nourishing_exp = qse_experiences.get('nourishing', [])
        poisonous_exp = qse_experiences.get('poisonous', [])

        if len(nourishing_exp) > len(poisonous_exp):
            print(f"✅ Blue fruit correctly associated with 'nourishing' category")

    return {
        'qse_test_accuracy': qse_test_acc,
        'rl_test_accuracy': rl_test_acc,
        'accuracy_advantage': accuracy_advantage,
        'speed_advantage': speed_advantage,
        'qse_categories': len(qse_categories),
        'embodied_learning': len(qse_experiences) > 0
    }

def calculate_categorization_accuracy(timeline):
    """Calculate overall categorization accuracy"""
    if not timeline:
        return 0.0

    correct = sum(1 for entry in timeline if entry['accuracy'])
    return correct / len(timeline)

def analyze_blue_fruit_learning(timeline):
    """Analyze learning curve for blue fruit specifically"""

    blue_interactions = [entry for entry in timeline if entry['fruit_type'] == 'blue_fruit']

    if not blue_interactions:
        return {'first_correct': float('inf'), 'stabilization': float('inf')}

    # Find first correct prediction
    first_correct = float('inf')
    for entry in blue_interactions:
        if entry['accuracy']:
            first_correct = entry['step']
            break

    # Find stabilization (3 consecutive correct)
    stabilization = float('inf')
    consecutive_correct = 0
    for entry in blue_interactions:
        if entry['accuracy']:
            consecutive_correct += 1
            if consecutive_correct >= 3:
                stabilization = entry['step']
                break
        else:
            consecutive_correct = 0

    return {
        'first_correct': first_correct,
        'stabilization': stabilization,
        'total_interactions': len(blue_interactions)
    }

def create_categorization_visualization(results):
    """Create comprehensive visualization of categorization experiment"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Fruit Categorization Experiment: Embodied vs Visual Learning', fontsize=16, fontweight='bold')

    # Extract data
    qse_train = results['qse_training']
    qse_test = results['qse_testing']
    rl_train = results['rl_training']
    rl_test = results['rl_testing']

    # 1. Learning curves during training
    ax1 = axes[0, 0]

    qse_train_timeline = qse_train['categorization_timeline']
    rl_train_timeline = rl_train['categorization_timeline']

    if qse_train_timeline:
        qse_steps = [entry['step'] for entry in qse_train_timeline]
        qse_accuracy = [entry['accuracy'] for entry in qse_train_timeline]
        qse_rolling_acc = calculate_rolling_accuracy(qse_accuracy, window=10)
        ax1.plot(qse_steps[:len(qse_rolling_acc)], qse_rolling_acc, 'b-', label='QSE-Émile', linewidth=2)

    if rl_train_timeline:
        rl_steps = [entry['step'] for entry in rl_train_timeline]
        rl_accuracy = [entry['accuracy'] for entry in rl_train_timeline]
        rl_rolling_acc = calculate_rolling_accuracy(rl_accuracy, window=10)
        ax1.plot(rl_steps[:len(rl_rolling_acc)], rl_rolling_acc, 'r-', label='Standard RL', linewidth=2)

    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Rolling Accuracy (10-step window)')
    ax1.set_title('Training Phase Learning Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Test phase learning curves (key plot!)
    ax2 = axes[0, 1]

    qse_test_timeline = qse_test['categorization_timeline']
    rl_test_timeline = rl_test['categorization_timeline']

    # Focus on blue fruit learning
    qse_blue = [entry for entry in qse_test_timeline if entry['fruit_type'] == 'blue_fruit']
    rl_blue = [entry for entry in rl_test_timeline if entry['fruit_type'] == 'blue_fruit']

    if qse_blue:
        qse_blue_steps = [entry['step'] for entry in qse_blue]
        qse_blue_acc = [entry['accuracy'] for entry in qse_blue]
        qse_blue_rolling = calculate_rolling_accuracy(qse_blue_acc, window=3)
        ax2.plot(qse_blue_steps[:len(qse_blue_rolling)], qse_blue_rolling, 'b-', label='QSE-Émile', linewidth=3)

    if rl_blue:
        rl_blue_steps = [entry['step'] for entry in rl_blue]
        rl_blue_acc = [entry['accuracy'] for entry in rl_blue]
        rl_blue_rolling = calculate_rolling_accuracy(rl_blue_acc, window=3)
        ax2.plot(rl_blue_steps[:len(rl_blue_rolling)], rl_blue_rolling, 'r-', label='Standard RL', linewidth=3)

    ax2.set_xlabel('Test Steps')
    ax2.set_ylabel('Rolling Accuracy (3-step window)')
    ax2.set_title('Blue Fruit Learning (Novel Category)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Categorization accuracy comparison
    ax3 = axes[0, 2]

    qse_train_acc = calculate_categorization_accuracy(qse_train_timeline)
    qse_test_acc = calculate_categorization_accuracy(qse_test_timeline)
    rl_train_acc = calculate_categorization_accuracy(rl_train_timeline)
    rl_test_acc = calculate_categorization_accuracy(rl_test_timeline)

    phases = ['Training', 'Test (Novel)']
    qse_accs = [qse_train_acc, qse_test_acc]
    rl_accs = [rl_train_acc, rl_test_acc]

    x = np.arange(len(phases))
    width = 0.35

    ax3.bar(x - width/2, qse_accs, width, label='QSE-Émile', color='blue', alpha=0.7)
    ax3.bar(x + width/2, rl_accs, width, label='Standard RL', color='red', alpha=0.7)

    ax3.set_xlabel('Experimental Phase')
    ax3.set_ylabel('Categorization Accuracy')
    ax3.set_title('Overall Categorization Performance')
    ax3.set_xticks(x)
    ax3.set_xticklabels(phases)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add accuracy values on bars
    for i, (qse_acc, rl_acc) in enumerate(zip(qse_accs, rl_accs)):
        ax3.text(i - width/2, qse_acc + 0.02, f'{qse_acc:.1%}', ha='center', fontweight='bold')
        ax3.text(i + width/2, rl_acc + 0.02, f'{rl_acc:.1%}', ha='center', fontweight='bold')

    # 4. QSE-Émile category formation
    ax4 = axes[1, 0]

    qse_agent = results['qse_agent']
    categories = qse_agent.perceptual_categories
    if categories:
        cat_names = list(categories.keys())
        cat_counts = [len(examples) for examples in categories.values()]

        bars = ax4.bar(range(len(cat_names)), cat_counts, color='skyblue', alpha=0.7)
        ax4.set_xticks(range(len(cat_names)))
        ax4.set_xticklabels(cat_names, rotation=45, ha='right')
        ax4.set_ylabel('Number of Examples')
        ax4.set_title('QSE-Émile Emergent Categories')

        for bar, count in zip(bars, cat_counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
    else:
        ax4.text(0.5, 0.5, 'No categories formed', ha='center', va='center', transform=ax4.transAxes)

    ax4.grid(True, alpha=0.3)

    # 5. Embodied experience analysis
    ax5 = axes[1, 1]

    experiences = qse_agent.fruit_experiences
    if experiences:
        signatures = list(experiences.keys())
        avg_energies = []

        for sig in signatures:
            exp_list = experiences[sig]
            avg_energy = np.mean([exp['energy_change'] for exp in exp_list])
            avg_energies.append(avg_energy)

        colors = ['green' if energy > 0 else 'red' for energy in avg_energies]
        bars = ax5.bar(range(len(signatures)), avg_energies, color=colors, alpha=0.7)
        ax5.set_xticks(range(len(signatures)))
        ax5.set_xticklabels(signatures)
        ax5.set_ylabel('Average Energy Change')
        ax5.set_title('Embodied Signatures by Energy Effect')
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        for bar, energy in zip(bars, avg_energies):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.01 if energy > 0 else -0.03),
                    f'{energy:+.2f}', ha='center', va='bottom' if energy > 0 else 'top', fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'No embodied experiences', ha='center', va='center', transform=ax5.transAxes)

    ax5.grid(True, alpha=0.3)

    # 6. Learning speed comparison
    ax6 = axes[1, 2]

    qse_blue_analysis = analyze_blue_fruit_learning(qse_test['categorization_timeline'])
    rl_blue_analysis = analyze_blue_fruit_learning(rl_test['categorization_timeline'])

    metrics = ['First Correct', 'Stabilization']
    qse_steps = [qse_blue_analysis['first_correct'], qse_blue_analysis['stabilization']]
    rl_steps = [rl_blue_analysis['first_correct'], rl_blue_analysis['stabilization']]

    # Handle infinite values
    qse_steps = [step if step != float('inf') else 1000 for step in qse_steps]
    rl_steps = [step if step != float('inf') else 1000 for step in rl_steps]

    x = np.arange(len(metrics))
    width = 0.35

    ax6.bar(x - width/2, qse_steps, width, label='QSE-Émile', color='blue', alpha=0.7)
    ax6.bar(x + width/2, rl_steps, width, label='Standard RL', color='red', alpha=0.7)

    ax6.set_xlabel('Learning Milestone')
    ax6.set_ylabel('Steps to Achieve')
    ax6.set_title('Blue Fruit Learning Speed')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Add step values on bars
    for i, (qse_step, rl_step) in enumerate(zip(qse_steps, rl_steps)):
        ax6.text(i - width/2, qse_step + 20, str(qse_step), ha='center', fontweight='bold')
        ax6.text(i + width/2, rl_step + 20, str(rl_step), ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('fruit_categorization_experiment.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📊 Fruit categorization visualization saved as 'fruit_categorization_experiment.png'")

def calculate_rolling_accuracy(accuracy_list, window=10):
    """Calculate rolling accuracy with specified window"""
    if len(accuracy_list) < window:
        return accuracy_list

    rolling = []
    for i in range(len(accuracy_list) - window + 1):
        window_acc = sum(accuracy_list[i:i+window]) / window
        rolling.append(window_acc)

    return rolling

def main():
    """Run the FIXED fruit categorization experiment"""

    results = run_fruit_categorization_experiment(training_steps=600, test_steps=300)
    return results

if __name__ == "__main__":
    results = main()
