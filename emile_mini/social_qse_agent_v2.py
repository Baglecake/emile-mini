
"""
Social QSE-Émile Agent System: Emergent Social Cognition

Demonstrates how QSE dynamics create social behavior patterns:
- Social detection and communication through embodied signals
- Context-dependent social strategies (cooperation, competition, teaching)
- Emergent social learning and knowledge transfer
- Multi-agent QSE resonance effects
"""
from typing import DefaultDict, List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from pathlib import Path
import time

# Import existing modules
from .embodied_qse_emile import EmbodiedQSEAgent, EmbodiedEnvironment, SensoriMotorBody
from .config import QSEConfig

# --- Social ranges / thresholds (single source of truth) ---
CLOSE_RANGE = QSEConfig().SOCIAL_DETECT_RADIUS  # e.g., 4
TEACH_MIN_CONF = 0.80         # don't teach if confidence below this
NOVELTY_EPS   = 0.05          # don't reteach if learner already knows within ±0.05



class SocialSignal:
    """Embodied social signals between agents"""

    def __init__(self, signal_type, sender_id, intensity=1.0, data=None):
        self.type = signal_type  # 'help', 'warning', 'share', 'compete', 'teach'
        self.sender_id = sender_id
        self.target_agent_id = None
        self.intensity = intensity
        self.data = data or {}
        self.timestamp = time.time()
        self.embodied_signature = self._compute_signature()

    def _compute_signature(self):
        """Compute embodied signature of the signal"""
        type_mapping = {
            'help': 0.8,      # Positive embodied feeling
            'warning': -0.5,  # Negative embodied feeling
            'share': 0.6,     # Positive sharing feeling
            'compete': -0.2,  # Slight negative competitive feeling
            'teach': 0.7,     # Positive teaching feeling
            'follow': 0.3,    # Neutral following
            'avoid': -0.4     # Negative avoidance
        }
        return type_mapping.get(self.type, 0.0) * self.intensity

class SocialQSEAgent(EmbodiedQSEAgent):
    """QSE-Émile agent with social cognition capabilities"""

    def __init__(self, agent_id, config=QSEConfig()):
        super().__init__(config)
        self.pending_teach_requests = set()

        self.agent_id = agent_id

        # Initialize embodied_mappings if not already done by parent
        if not hasattr(self, 'embodied_mappings'):
            self.embodied_mappings = defaultdict(list)

        # Social cognition components
        self.social_memory = deque(maxlen=100)
        self.social_relationships = defaultdict(lambda: defaultdict(float))
        self.social_strategies = ['cooperative', 'competitive', 'teaching', 'learning', 'independent']
        self.current_social_strategy = 'independent'

        # Social signals
        self.outgoing_signals = deque(maxlen=10)
        self.incoming_signals = deque(maxlen=20)

        # Social learning
        self.social_knowledge_transfer = defaultdict(list)
        self.teaching_success = defaultdict(int)

        # Episodic surplus memory for knowledge transfer
        self.episodic_surplus_memory = deque(maxlen=200)
        self.symbolic_field_history = deque(maxlen=100)

        # Spatial tracking for analysis
        self.position_history = deque(maxlen=1000)
        self.proximity_history = defaultdict(list)

        # Existential pressure system - track behavioral variety (distinction)
        self.recent_action_history = deque(maxlen=20)

        self.teaching_history = defaultdict(set)  # What we've taught to whom
        self.learning_cooldown = defaultdict(int)  # Cooldown after learning something

        # Enhanced goals for social interaction
        social_goals = ["social_explore", "help_others", "compete", "teach", "learn_from_others", "form_alliance"]
        for goal in social_goals:
            self.goal.add_goal(goal)

        # Social personality (emerges from QSE dynamics)
        self.social_personality = {
            'cooperativeness': 0.5,
            'competitiveness': 0.5,
            'teaching_tendency': 0.5,
            'learning_openness': 0.5,
            'social_confidence': 0.5
        }

        self.step_idx = 0  # Track current step for teaching cooldowns

        print(f"🤝 Social Agent {self.agent_id} initialized")

    def _sigma_for_memory(self) -> float:
        """Best-effort sigma value for episodic memory, regardless of backend."""
        # 1) prefer qse.sigma if present
        if hasattr(self, "qse") and hasattr(self.qse, "sigma"):
            try:
                if getattr(self.qse, "sigma") is not None and self.qse.sigma.size > 0:
                    return float(np.mean(self.qse.sigma))
            except Exception:
                pass
        # 2) fall back to symbolic's EMA if available
        if hasattr(self, "symbolic") and hasattr(self.symbolic, "sigma_ema"):
            try:
                return float(self.symbolic.sigma_ema)
            except Exception:
                pass
        # 3) last-ditch: normalized mean surplus, if S exists
        if hasattr(self, "qse") and hasattr(self.qse, "S"):
            try:
                return float(np.clip(np.mean(self.qse.S), 0.0, 1.0))
            except Exception:
                pass
        return 0.0

    def _goal_to_embodied_action(self, goal, visual_field, environment=None):
        """
        Overrides the parent method to include social-specific action logic,
        including help-seeking and stuck detection.
        """
        if goal == 'learn_from_others':
            # Check if we are stuck (FIXED: convert deque to list before slicing)
            if len(self.position_history) > 10 and len(set(tuple(p) for p in list(self.position_history)[-10:])) <= 2:
                print(f"🆘 {self.agent_id} is stuck while learning, trying a random move.")
                return np.random.choice(['move_forward', 'turn_left', 'turn_right']), 0.7
            else:
                # Use help-seeking logic, which requires the environment
                if environment:
                    return self._seek_help_action(environment)
                else:
                    # Fallback if environment is not available for some reason
                    return 'examine', 0.5

        # For all other goals, use the parent class's logic
        return super()._goal_to_embodied_action(goal, visual_field, environment)

    def detect_nearby_agents(self, environment):
        """Detect other agents in the environment"""
        nearby_agents = []
        my_pos = self.body.state.position

        for other_agent in environment.agents.values():
            if other_agent.agent_id != self.agent_id:
                other_pos = other_agent.body.state.position
                distance = np.sqrt((my_pos[0] - other_pos[0])**2 + (my_pos[1] - other_pos[1])**2)

                # Record proximity for analysis
                self.proximity_history[other_agent.agent_id].append(distance)

                if distance <= CLOSE_RANGE:  # single source of truth
                    # Gather social information
                    social_info = {
                        'agent_id': other_agent.agent_id,
                        'position': other_pos,
                        'distance': distance,
                        'energy': other_agent.body.state.energy,
                        'current_goal': other_agent.goal.current_goal,
                        'context': other_agent.context.get_current(),
                        'recent_success': self._assess_other_success(other_agent)
                    }
                    nearby_agents.append(social_info)

        return nearby_agents

    def _assess_other_success(self, other_agent):
        """Assess another agent's recent success for social learning"""
        if hasattr(other_agent, 'body'):
            return other_agent.body.state.energy > 0.7
        return False

    def generate_social_signal(self, signal_type, target_agent_id=None, data=None):
        """Generate a social signal to communicate with other agents"""

        # Intensity based on current QSE state and social personality
        if hasattr(self, 'qse') and self.qse.S.size > 0:
            surplus_mean = float(np.mean(self.qse.S))
        else:
            surplus_mean = 0.5

        confidence = self.social_personality.get('social_confidence', 0.5)
        intensity = float(surplus_mean * confidence)

        signal = SocialSignal(signal_type, self.agent_id, intensity, data)
        signal.target_agent_id = target_agent_id
        self.outgoing_signals.append(signal)

        return signal

    def receive_social_signal(self, signal):
        """Process incoming social signal (no immediate teaching without env)."""
        self.incoming_signals.append(signal)

        # Update relationship
        relationship_change = signal.embodied_signature * 0.1
        self.social_relationships[signal.sender_id][signal.type] += relationship_change

        # If we receive concrete teaching, learn immediately
        if signal.type == 'teach' and signal.data:
            self._process_social_learning(signal)

        # If someone requests knowledge, queue it; we'll teach next step when env is available
        if signal.type == 'share' and signal.data.get('request') == 'knowledge_sharing':
            self.pending_teach_requests.add(signal.sender_id)
            print(f"🤝 {self.agent_id} queued teach for {signal.sender_id}")

        # Store social memory
        social_memory_entry = {
            'timestamp': signal.timestamp,
            'sender': signal.sender_id,
            'type': signal.type,
            'my_context': self.context.get_current(),
            'my_goal': self.goal.current_goal,
            'embodied_response': self._compute_embodied_response(signal)
        }
        self.social_memory.append(social_memory_entry)

    def _compute_embodied_response(self, signal):
        """Compute embodied response to social signal"""
        my_surplus = np.mean(self.qse.S) if hasattr(self, 'qse') else 0.5
        signal_effect = signal.embodied_signature

        # Personality modulation
        if signal.type == 'help':
            response = signal_effect * self.social_personality['cooperativeness']
        elif signal.type == 'compete':
            response = signal_effect * self.social_personality['competitiveness']
        elif signal.type == 'teach':
            response = signal_effect * self.social_personality['learning_openness']
        else:
            response = signal_effect * 0.5

        return response * (my_surplus + 0.1)

    def _process_social_learning(self, signal):
        """Learn knowledge from social teaching signal"""
        if 'knowledge' in signal.data:
            knowledge = signal.data['knowledge']
            sender = signal.sender_id

            # Store socially acquired knowledge
            self.social_knowledge_transfer[sender].append({
                'knowledge': knowledge,
                'timestamp': signal.timestamp,
                'confidence': signal.intensity,
                'my_context': self.context.get_current()
            })

            # Apply knowledge if we trust the sender (low barrier for openness)
            trust_level = sum(self.social_relationships[sender].values())
            if trust_level >= 0:  # Accept first lessons immediately
                self._apply_social_knowledge(knowledge, signal.intensity)
                print(f"🧠 {self.agent_id} learned about {knowledge.get('fruit_type', 'unknown')} from {sender}!")

    def _apply_social_knowledge(self, knowledge, confidence):
        """Apply socially learned knowledge with better value preservation"""
        if 'fruit_value' in knowledge:
            fruit_type = knowledge['fruit_type']
            value = knowledge['fruit_value']

            # NEW: Much better value retention (90% minimum vs your current heavy discounting)
            if hasattr(self, 'embodied_mappings'):
                adjusted_value = value * max(0.90, confidence)  # 90% minimum retention
                self.embodied_mappings[fruit_type].append(adjusted_value)
                print(f"📚 {self.agent_id} learned: {fruit_type} = {adjusted_value:.2f} (retained {adjusted_value/value:.1%})")

    def store_episodic_surplus_memory(self, symbolic_fields, context, goal, outcome, semantic_tags=None):
        """Store episodic memory with surplus embeddings"""
        memory_entry = {
            'timestamp': time.time(),
            'symbolic_fields': symbolic_fields,  # (psi, phi, sigma)
            'surplus_state': np.mean(self.qse.S) if hasattr(self, 'qse') else 0.5,
            'context_id': context,
            'goal': goal,
            'outcome': outcome,
            'semantic_tags': semantic_tags or [],
            'position': self.body.state.position,
            'social_context': len(self.social_memory)
        }
        self.episodic_surplus_memory.append(memory_entry)

    def select_social_strategy(self, nearby_agents, environment_state):
        """Select social strategy with existential pressure + practical teaching."""
        old_strategy = self.current_social_strategy

        # Existential pressure: break loops
        if len(self.recent_action_history) == self.recent_action_history.maxlen:
            recent_positions = list(self.position_history)[-15:] if len(self.position_history) >= 15 else list(self.position_history)
            is_stuck_in_place = len(recent_positions) > 10 and len(set(tuple(p) for p in recent_positions)) <= 2
            action_variety = len(set(self.recent_action_history))
            if is_stuck_in_place or action_variety <= 2:
                self.current_social_strategy = 'independent'
                self.goal.current_goal = 'explore'
                if old_strategy != self.current_social_strategy:
                    print(f"⚠️ {self.agent_id} existential pressure: breaking repetitive loop! Forcing exploration to create distinction.")
                for k in self.social_personality:
                    self.social_personality[k] = np.clip(self.social_personality[k], 0.0, 1.0)
                return

        # No one around → move toward others (your _seek_help_action handles motion)
        if not nearby_agents:
            self.current_social_strategy = 'independent'
            self.goal.current_goal = 'learn_from_others'
            return

        # If I have any embodied knowledge and someone is nearby, teach now
        if any(len(vals) >= 1 for vals in getattr(self, "embodied_mappings", {}).values()):
            self.current_social_strategy = 'teaching'
            self.goal.current_goal = 'teach'
            return

        # Otherwise pick a sensible default based on resources
        surplus_mean = np.mean(self.qse.S) if hasattr(self, 'qse') else 0.5

        if self.body.state.energy < 0.6:
            self.current_social_strategy = 'learning'
            self.goal.current_goal = 'learn_from_others'
            self.social_personality['learning_openness'] += 0.01
        elif surplus_mean > 0.6 and self.body.state.energy > 0.7:
            # high resources → cooperate (unless above branch triggered teaching)
            self.current_social_strategy = 'cooperative'
            self.goal.current_goal = 'explore'
            self.social_personality['cooperativeness'] += 0.01
        else:
            self.current_social_strategy = 'cooperative'
            self.goal.current_goal = 'explore'

        if old_strategy != self.current_social_strategy:
            print(f"🔄 {self.agent_id}: {old_strategy} → {self.current_social_strategy} "
                  f"(energy={self.body.state.energy:.2f}, surplus={surplus_mean:.2f})")

        for k in self.social_personality:
            self.social_personality[k] = np.clip(self.social_personality[k], 0.0, 1.0)


    def _seek_help_action(self, environment):
        """HELP-SEEKING: When learning, actively move towards other agents"""

        # Detect all agents (not just nearby ones within 3 units)
        all_agents = []
        my_pos = self.body.state.position

        for other_agent in environment.agents.values():
            if other_agent.agent_id != self.agent_id:
                other_pos = other_agent.body.state.position
                distance = np.sqrt((my_pos[0] - other_pos[0])**2 + (my_pos[1] - other_pos[1])**2)
                all_agents.append({
                    'agent_id': other_agent.agent_id,
                    'position': other_pos,
                    'distance': distance
                })

        if not all_agents:
            return 'examine', 0.5  # No other agents - default behavior

        # Find the closest agent
        closest_agent = min(all_agents, key=lambda x: x['distance'])

        print(f"🆘 {self.agent_id} seeking help from {closest_agent['agent_id']} "
              f"(distance: {closest_agent['distance']:.2f})")

        # Calculate direction to move towards closest agent
        my_pos = np.array(self.body.state.position)
        target_pos = np.array(closest_agent['position'])
        move_vector = target_pos - my_pos

        # Choose best direction (simplified to 4 cardinal directions)
        if abs(move_vector[0]) > abs(move_vector[1]):
            # Move horizontally
            if move_vector[0] > 0:
                return 'move_forward', 1.0  # Move towards higher x
            else:
                return 'move_backward', 1.0  # Move towards lower x
        else:
            # Move vertically
            if move_vector[1] > 0:
                return 'turn_right', 1.0  # This maps to movement in embodied system
            else:
                return 'turn_left', 1.0   # This maps to movement in embodied system

    def execute_social_strategy(self, nearby_agents, environment):
        """Execute current social strategy through actions and signals"""

        if not nearby_agents:
            return None

        strategy = self.current_social_strategy

        if strategy == 'teaching':
            return self._execute_teaching(nearby_agents, environment)
        elif strategy == 'learning':
            return self._execute_learning(nearby_agents, environment)
        elif strategy == 'cooperative':
            return self._execute_cooperation(nearby_agents, environment)
        elif strategy == 'competitive':
            return self._execute_competition(nearby_agents, environment)
        else:
            return None

    def _execute_teaching(self, nearby_agents, environment):
        """Teaching strategy: Share knowledge with cooldown, confidence and novelty checks."""
        # Choose the lowest-energy neighbor (or any)
        target = None
        if nearby_agents:
            for agent_info in sorted(nearby_agents, key=lambda x: x['energy']):
                if len(self.teaching_history[agent_info['agent_id']]) < 3:
                    target = agent_info
                    break
            if not target:
                target = nearby_agents[0]

        if not target or not getattr(self, 'embodied_mappings', None):
            return None  # be quiet if nothing to do

        target_id = target['agent_id']
        print(f"🧑‍🏫 {self.agent_id} attempting to teach. Nearby agents: {[a['agent_id'] for a in nearby_agents]}")
        print(f"   📚 {self.agent_id} has knowledge: {list(self.embodied_mappings.keys())}")

        best = None
        best_conf = 0.0

        for topic, values in self.embodied_mappings.items():
            if not values:
                continue
            # per-target/topic cooldown
            last_teach_step = getattr(self, f'_last_teach_{target_id}_{topic}', -999)
            cooldown_steps = getattr(self.cfg, 'TEACH_COOLDOWN_STEPS', 25)
            if (self.step_idx - last_teach_step) <= cooldown_steps:
                continue

            # teacher confidence
            if len(values) > 1:
                conf = 1.0 - (np.std(values) / (abs(np.mean(values)) + 0.1))
            else:
                conf = 1.0
            if conf < TEACH_MIN_CONF:
                continue

            taught_value = float(np.mean(values))

            # novelty: skip reteaching if learner already knows ~the same
            learner = environment.agents.get(target_id)
            if learner and hasattr(learner, 'embodied_mappings'):
                lv = learner.embodied_mappings.get(topic, [])
                if lv and abs(float(np.mean(lv)) - taught_value) < NOVELTY_EPS:
                    continue

            if conf > best_conf:
                best_conf = conf
                best = (topic, taught_value, conf)

        if not best:
            return None  # nothing novel/allowed to teach right now

        topic, taught_value, teacher_conf = best
        learner_agent = environment.agents.get(target_id)
        if not learner_agent or not hasattr(learner_agent, 'embodied_mappings'):
            return None

        # retain ≥90% of value (or teacher_conf if higher)
        retained_value = float(taught_value * max(0.90, teacher_conf))
        learner_agent.embodied_mappings[topic].append(retained_value)

        # bookkeeping
        self.teaching_history[target_id].add(topic)
        setattr(self, f'_last_teach_{target_id}_{topic}', self.step_idx)

        # dashboards
        self.social_knowledge_transfer[target_id].append({
            "topic": topic, "value": retained_value, "source": self.agent_id, "step": int(self.step_idx)
        })
        if hasattr(learner_agent, "social_knowledge_transfer"):
            learner_agent.social_knowledge_transfer[self.agent_id].append({
                "topic": topic, "value": retained_value, "source": self.agent_id, "step": int(self.step_idx)
            })

        # memory (teacher + learner)
        sigma_now = self._sigma_for_memory()
        self.store_episodic_surplus_memory(
            symbolic_fields=(0, 0, sigma_now),
            context=self.context.get_current(),
            goal=self.goal.current_goal,
            outcome=f"taught:{topic}",
            semantic_tags=['social', 'teach', topic, f"to:{target_id}"]
        )
        learner_agent.store_episodic_surplus_memory(
            symbolic_fields=(0, 0, sigma_now),
            context=learner_agent.context.get_current(),
            goal=learner_agent.goal.current_goal,
            outcome=f"learned:{topic}",
            semantic_tags=['social', 'learn', topic, f"from:{self.agent_id}"]
        )

        # send a real 'teach' signal so learner logs it too
        teach_signal = SocialSignal(
            'teach', self.agent_id, intensity=float(teacher_conf),
            data={"knowledge": {"fruit_type": topic, "fruit_value": float(taught_value)}}
        )
        teach_signal.target_agent_id = target_id
        environment.social_signals[target_id].append(teach_signal)

        # memory.log (if your memory system exists)
        if hasattr(self, "memory"):
            self.memory.store({
                "type": "social_transfer",
                "from": self.agent_id, "to": target_id,
                "topic": topic,
                "value": float(taught_value),
                "retained_value": retained_value,
                "confidence": float(teacher_conf),
                "step": int(self.step_idx)
            }, tags={"type": "social"})

        print(f"🎁 {self.agent_id} taught {target_id}: {topic} = {taught_value:.2f} → {retained_value:.2f}")
        return f"Teaching {target_id}: {topic} = {retained_value:.2f}"


    def _execute_learning(self, nearby_agents, environment):
        """Learning strategy: Seek knowledge from successful agents"""

        # Find most successful nearby agent
        best_agent = max(nearby_agents, key=lambda x: x['energy']) if nearby_agents else None

        if best_agent and best_agent['recent_success']:
            signal = self.generate_social_signal('share', best_agent['agent_id'],
                                               {'request': 'knowledge_sharing'})
            return f"Requesting knowledge from {best_agent['agent_id']}"

        return "No learning opportunity"

    def _execute_cooperation(self, nearby_agents, environment):
        """
        Cooperative strategy: Prioritize helping agents in a 'learning' state.
        """

        # Find an agent that needs help (critically low energy OR in a learning state)
        target_to_help = None
        for agent_info in nearby_agents:
            target_agent_obj = environment.agents.get(agent_info['agent_id'])
            if target_agent_obj:
                # Prioritize helping agents who are actively learning
                if target_agent_obj.current_social_strategy == 'learning':
                    target_to_help = agent_info
                    break
                # Also help those with critically low energy
                if agent_info['energy'] < 0.4:
                    target_to_help = agent_info
                    # Don't break here, keep looking for a learner which is higher priority

        if target_to_help:
            # We found someone to help! Send a signal.
            signal = self.generate_social_signal('help', target_to_help['agent_id'],
                                              {'offer': 'assistance'})
            print(f"🤝 {self.agent_id} is actively helping {target_to_help['agent_id']}!")
            return f"Helping {target_to_help['agent_id']}"
        else:
            # No one needs help, so perform a routine cooperative check
            if nearby_agents and np.random.random() < 0.1: # 10% chance to signal, only if agents exist
                random_agent = nearby_agents[np.random.randint(len(nearby_agents))]
                self.generate_social_signal('share', random_agent['agent_id'],
                                          {'message': 'cooperative_check'})
                return f"Cooperative check with {random_agent['agent_id']}"

        return "Cooperative monitoring"

    def _execute_competition(self, nearby_agents, environment):
        """Competitive strategy: Compete for resources"""

        # Signal competition to all nearby agents
        for agent_info in nearby_agents:
            signal = self.generate_social_signal('compete', agent_info['agent_id'],
                                               {'message': 'competing_for_resources'})

        return "Competing for resources"

    def social_embodied_step(self, environment, dt=0.01):
        """Enhanced embodied step with existential pressure + social pipeline."""
        self.step_idx += 1

        # Movement + existential pressure
        self.position_history.append(self.body.state.position)
        pos_before = self.position_history[-2] if len(self.position_history) >= 2 else self.body.state.position
        result = self.embodied_step(environment, dt)
        self.recent_action_history.append(result.get('action'))

        pos_after = self.body.state.position
        if pos_before == pos_after:
            if hasattr(self.qse, 'S'):
                self.qse.S = np.clip(self.qse.S - 0.015, 0, 1)
            self.body.state.energy = max(0.1, self.body.state.energy - 0.01)
        else:
            if hasattr(self.qse, 'S'):
                self.qse.S = np.clip(self.qse.S + 0.004, 0, 1)
            self.body.state.energy = min(1.2, self.body.state.energy + 0.001)

        # Social sensing
        nearby_agents = self.detect_nearby_agents(environment)

        # Inbox
        if hasattr(environment, 'social_signals'):
            inbox = environment.social_signals.get(self.agent_id, [])
            if inbox:
                print(f"📨 {self.agent_id} receiving {len(inbox)} signals")
            for signal in inbox:
                self.receive_social_signal(signal)
                print(f"🔔 {self.agent_id} processed signal: {signal.type} from {signal.sender_id}")
            environment.social_signals[self.agent_id] = []

        # Strategy
        self.select_social_strategy(nearby_agents, environment)

        # Opportunistic queued teaching (serve only when learner is nearby)
        if self.pending_teach_requests and nearby_agents:
            teach_targets = [info for info in nearby_agents if info['agent_id'] in self.pending_teach_requests]
            if teach_targets:
                outcome = self._execute_teaching(teach_targets, environment)
                if isinstance(outcome, str) and outcome.startswith("Teaching"):
                    for info in teach_targets:
                        self.pending_teach_requests.discard(info['agent_id'])

        # Act
        social_action = self.execute_social_strategy(nearby_agents, environment)

        if social_action and social_action not in ("No teaching opportunity", "No learning opportunity"):
            print(f"🤝 {self.agent_id} social action: {social_action}")
        if self.outgoing_signals:
            print(f"📤 {self.agent_id} generated {len(self.outgoing_signals)} outgoing signals")

        # —— ALWAYS write an episodic memory entry ——
        sigma_now = self._sigma_for_memory()
        self.store_episodic_surplus_memory(
            symbolic_fields=(0, 0, sigma_now),
            context=self.context.get_current(),
            goal=self.goal.current_goal,
            outcome=result.get('outcome', 'neutral'),
            semantic_tags=['social', social_action or 'none', self.current_social_strategy]
        )

        # Broadcast
        if hasattr(environment, 'social_signals') and self.outgoing_signals:
            for signal in list(self.outgoing_signals):
                tid = signal.target_agent_id
                if tid and tid in environment.agents:
                    environment.social_signals[tid].append(signal)
                elif nearby_agents:
                    for info in nearby_agents:
                        environment.social_signals[info['agent_id']].append(signal)
            self.outgoing_signals.clear()

        # For charts
        if 'cognitive_metrics' in result and 'sigma_mean' in result['cognitive_metrics']:
            self.last_sigma = [result['cognitive_metrics']['sigma_mean']]

        result['social_info'] = {
            'nearby_agents': len(nearby_agents),
            'current_strategy': self.current_social_strategy,
            'social_action': social_action,
            'personality': dict(self.social_personality),
            'outgoing_signals': 0,
            'incoming_signals': len(self.incoming_signals),
            'episodic_memories': len(self.episodic_surplus_memory)
        }
        return result


class SocialEnvironment(EmbodiedEnvironment):
    """Environment supporting multiple social agents"""

    def __init__(self, size=20):
        super().__init__(size)
        self.agents = {}
        self.social_signals = defaultdict(list)
        self.social_history = []

        # Add some social objects
        self.create_social_environment()

    def create_social_environment(self):
        """Create environment conducive to social interaction"""

        # Add shared resources that encourage interaction
        resource_types = [
            ('shared_food', 0.4, 3),
            ('teaching_stone', 0.6, 2),
            ('competition_prize', 0.8, 1)
        ]

        for obj_type, grid_value, count in resource_types:
            for _ in range(count):
                while True:
                    x, y = np.random.randint(3, self.size-3, 2)
                    if self.grid[x, y] == 0:
                        break

                self.grid[x, y] = grid_value
                self.objects[(x, y)] = {
                    'type': obj_type,
                    'properties': {'social_value': grid_value},
                    'discovered': False,
                    'interaction_count': 0
                }

        # Add more fruits for agents to categorize and share knowledge about
        fruit_types = [
            ('red_fruit', 0.3, 2, 0.4, 'nourishing'),    # Reduced count
            ('blue_fruit', 0.25, 2, 0.35, 'nourishing'), # Reduced count
            ('crimson_fruit', 0.35, 6, -0.9, 'poisonous') # INCREASED dangerous fruit count!
        ]

        for fruit_type, grid_value, count, energy_effect, embodied_sig in fruit_types:
            for _ in range(count):
                while True:
                    x, y = np.random.randint(3, self.size-3, 2)
                    if self.grid[x, y] == 0:
                        break

                self.grid[x, y] = grid_value
                self.objects[(x, y)] = {
                    'type': 'fruit',
                    'fruit_type': fruit_type,
                    'properties': {
                        'energy_effect': energy_effect,
                        'embodied_signature': embodied_sig,
                        'visual_signature': np.random.rand(3)  # Random color
                    },
                    'discovered': False,
                    'interaction_count': 0
                }

    def add_agent(self, agent):
        """Add a social agent to the environment"""
        self.agents[agent.agent_id] = agent

        # Place agent in environment
        while True:
            x, y = np.random.randint(2, self.size-2, 2)
            if self.grid[x, y] == 0:
                agent.body.state.position = (x, y)
                break

        print(f"Added agent {agent.agent_id} at position {agent.body.state.position}")

    def step_all_agents(self, dt=0.01):
        """Step all agents and process social interactions (consumable fruit)."""
        results = {}

        for agent_id, agent in self.agents.items():
            result = agent.social_embodied_step(self, dt)
            results[agent_id] = result

            pos = agent.body.state.position
            obj = self.objects.get(pos)

            if not obj:
                continue

            # Fruit interaction: only once per fruit
            if obj['type'] == 'fruit' and not obj.get('consumed', False):
                if result.get('action') == 'examine':
                    fruit_type   = obj['fruit_type']
                    energy_eff   = obj['properties']['energy_effect']

                    # add knowledge & apply energy
                    agent.embodied_mappings[fruit_type].append(energy_eff)
                    agent.body.state.energy = np.clip(agent.body.state.energy + energy_eff, 0, 1.2)

                    # mark consumed and clear the grid cell
                    obj['consumed'] = True
                    self.grid[pos] = 0.0
                    print(f"🍎 {agent_id} consumed {fruit_type} (effect: {energy_eff:+.2f})")

        # snapshot
        social_snapshot = {
            'timestamp': time.time(),
            'agent_positions': {aid: a.body.state.position for aid, a in self.agents.items()},
            'agent_strategies': {aid: a.current_social_strategy for aid, a in self.agents.items()},
            'agent_contexts':   {aid: a.context.get_current() for aid, a in self.agents.items()},
            'social_signals_count': sum(len(sigs) for sigs in self.social_signals.values())
        }
        self.social_history.append(social_snapshot)
        return results

def analyze_spatial_dynamics(agents, steps):
    """Analyze spatial movement and proximity patterns"""

    print(f"\n🔍 SPATIAL DYNAMICS ANALYSIS")
    print("=" * 40)

    for agent in agents:
        positions = list(agent.position_history)
        if len(positions) > 1:
            # Calculate movement distance
            total_distance = 0
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                total_distance += np.sqrt(dx*dx + dy*dy)

            avg_distance_per_step = total_distance / len(positions)

            print(f"Agent {agent.agent_id}:")
            print(f"  Total movement: {total_distance:.2f} units")
            print(f"  Avg per step: {avg_distance_per_step:.3f} units")
            print(f"  Start position: {positions[0]}")
            print(f"  End position: {positions[-1]}")

            # Analyze proximity to other agents
            for other_agent_id, distances in agent.proximity_history.items():
                if distances:
                    avg_proximity = np.mean(distances)
                    min_proximity = np.min(distances)
                    close_encounters = sum(1 for d in distances if d <= CLOSE_RANGE)
                    print(f"  Proximity to {other_agent_id}: avg={avg_proximity:.2f}, min={min_proximity:.2f}, close_encounters={close_encounters}")

def create_enhanced_social_visualization(env, agents, results):
    """Enhanced visualization with spatial and episodic analysis"""

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('Enhanced Social QSE-Émile: Spatial & Episodic Analysis', fontsize=16, fontweight='bold')

    steps = len(results)
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    # 1. Agent trajectories with proximity zones
    ax1 = axes[0, 0]

    for i, agent in enumerate(agents):
        positions = list(agent.position_history)
        if positions:
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            ax1.plot(x_coords, y_coords, color=colors[i % len(colors)],
                    alpha=0.7, linewidth=2, label=agent.agent_id)

            # Mark final position
            ax1.scatter(x_coords[-1], y_coords[-1], color=colors[i % len(colors)],
                       s=100, marker='o', edgecolors='black', linewidth=2)

    ax1.set_title('Agent Movement Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Proximity heatmap over time
    ax2 = axes[0, 1]

    # Create proximity matrix over time
    if len(agents) >= 2:
        agent_pairs = [(agents[0], agents[1])]
        if len(agents) >= 3:
            agent_pairs.extend([(agents[0], agents[2]), (agents[1], agents[2])])

        for pair in agent_pairs:
            agent_a, agent_b = pair
            if agent_b.agent_id in agent_a.proximity_history:
                distances = agent_a.proximity_history[agent_b.agent_id]
                if distances:
                    ax2.plot(range(len(distances)), distances,
                            label=f"{agent_a.agent_id}-{agent_b.agent_id}", linewidth=2)

        ax2.axhline(y=CLOSE_RANGE, color='red', linestyle='--', alpha=0.5, label='Social Detection Range')
        ax2.set_title(f'Inter-Agent Proximity Over Time (detect ≤ {CLOSE_RANGE})')
        ax2.set_title('Inter-Agent Proximity Over Time')
        ax2.set_ylabel('Distance')
        ax2.set_xlabel('Time Steps')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. Episodic memory formation
    ax3 = axes[0, 2]

    for i, agent in enumerate(agents):
        memory_count = [0]
        for step in range(1, steps + 1):
            if step <= len(agent.episodic_surplus_memory):
                memory_count.append(len(agent.episodic_surplus_memory))
            else:
                memory_count.append(memory_count[-1])

        if len(memory_count) > 1:
            ax3.plot(range(len(memory_count)), memory_count,
                    color=colors[i % len(colors)], linewidth=2, label=agent.agent_id)

    ax3.set_title('Episodic Memory Formation')
    ax3.set_ylabel('Memory Entries')
    ax3.set_xlabel('Time Steps')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Social strategy evolution
    ax4 = axes[1, 0]

    strategy_timeline = defaultdict(list)
    for step_result in results:
        for agent_id, result in step_result.items():
            if 'social_info' in result:
                strategy_timeline[agent_id].append(result['social_info']['current_strategy'])

    strategy_encoding = {'independent': 0, 'cooperative': 1, 'competitive': 2, 'teaching': 3, 'learning': 4}

    for i, (agent_id, strategies) in enumerate(strategy_timeline.items()):
        encoded_strategies = [strategy_encoding.get(s, 0) for s in strategies]
        ax4.plot(range(len(encoded_strategies)), encoded_strategies,
                color=colors[i % len(colors)], linewidth=2, label=agent_id, alpha=0.7)

    ax4.set_title('Social Strategy Evolution')
    ax4.set_ylabel('Strategy Type')
    ax4.set_xlabel('Time Steps')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Social signal frequency
    ax5 = axes[1, 1]

    signal_counts = []
    for snapshot in env.social_history:
        signal_counts.append(snapshot['social_signals_count'])

    if signal_counts:
        ax5.plot(range(len(signal_counts)), signal_counts, 'purple', linewidth=2)
        ax5.set_title('Social Communication Volume')
        ax5.set_ylabel('Number of Signals')
        ax5.set_xlabel('Time Steps')
        ax5.grid(True, alpha=0.3)

    # 6. Context synchronization
    ax6 = axes[1, 2]

    context_timeline = defaultdict(list)
    for step_result in results:
        for agent_id, result in step_result.items():
            if 'context' in result:
                context_timeline[agent_id].append(result['context'])

    for i, (agent_id, contexts) in enumerate(context_timeline.items()):
        ax6.plot(range(len(contexts)), contexts,
                color=colors[i % len(colors)], linewidth=2, label=agent_id, alpha=0.7)

    ax6.set_title('QSE Context Evolution')
    ax6.set_ylabel('Context ID')
    ax6.set_xlabel('Time Steps')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Personality development
    ax7 = axes[2, 0]

    personality_traits = ['cooperativeness', 'competitiveness', 'teaching_tendency', 'learning_openness']

    for i, agent in enumerate(agents):
        trait_values = [agent.social_personality[trait] for trait in personality_traits]

        x_positions = np.arange(len(personality_traits)) + i * 0.15
        bars = ax7.bar(x_positions, trait_values, width=0.15,
                      color=colors[i % len(colors)], alpha=0.7, label=agent.agent_id)

    ax7.set_title('Final Personality Profiles')
    ax7.set_ylabel('Trait Strength')
    ax7.set_xticks(np.arange(len(personality_traits)) + 0.15)
    ax7.set_xticklabels(personality_traits, rotation=45, ha='right')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Social relationships matrix
    ax8 = axes[2, 1]

    # Create relationship strength matrix
    agent_ids = [agent.agent_id for agent in agents]
    relationship_matrix = np.zeros((len(agents), len(agents)))

    for i, agent in enumerate(agents):
        for j, other_agent_id in enumerate(agent_ids):
            if other_agent_id in agent.social_relationships:
                total_strength = sum(agent.social_relationships[other_agent_id].values())
                relationship_matrix[i, j] = total_strength

    if np.any(relationship_matrix):
        im = ax8.imshow(relationship_matrix, cmap='viridis', alpha=0.8)
        ax8.set_xticks(range(len(agent_ids)))
        ax8.set_yticks(range(len(agent_ids)))
        ax8.set_xticklabels(agent_ids)
        ax8.set_yticklabels(agent_ids)
        ax8.set_title('Social Relationship Matrix')
        plt.colorbar(im, ax=ax8)

    # 9. Knowledge transfer events
    ax9 = axes[2, 2]

    transfer_counts = []
    for agent in agents:
        total_transfers = sum(len(transfers) for transfers in agent.social_knowledge_transfer.values())
        transfer_counts.append(total_transfers)

    bars = ax9.bar(range(len(agents)), transfer_counts,
                   color=[colors[i % len(colors)] for i in range(len(agents))], alpha=0.7)
    ax9.set_xticks(range(len(agents)))
    ax9.set_xticklabels([agent.agent_id for agent in agents])
    ax9.set_title('Knowledge Transfer Events')
    ax9.set_ylabel('Transfer Count')
    ax9.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, transfer_counts)):
        ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('enhanced_social_qse_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📊 Enhanced social analysis saved as 'enhanced_social_qse_analysis.png'")

def _cluster_spawn(env, agents, radius=2):
    """Place first agent at center, others within `radius` so distance ≤ CLOSE_RANGE."""
    cx, cy = env.size // 2, env.size // 2

    # Anchor
    agents[0].body.state.position = (cx, cy)

    # Place others very close to anchor (radius ≤ CLOSE_RANGE-1)
    tight = min(radius, max(1, CLOSE_RANGE - 1))
    for a in agents[1:]:
        dx = int(np.random.randint(-tight, tight + 1))
        dy = int(np.random.randint(-tight, tight + 1))
        a.body.state.position = (cx + dx, cy + dy)

def run_social_experiment(n_agents=3, steps=500, cluster_spawn=True, cluster_radius=6):
    """Run social QSE-Émile experiment with enhanced analysis"""

    print("🤝 ENHANCED SOCIAL QSE-ÉMILE EXPERIMENT")
    print("=" * 50)
    print(f"Creating {n_agents} social agents for {steps} steps")

    # Create social environment
    env = SocialEnvironment(size=15)

    # Create and add social agents
    agents = []
    for i in range(n_agents):
        agent = SocialQSEAgent(f"Agent_{i}")
        env.add_agent(agent)
        agents.append(agent)

    # 🔻 move everyone near center right after adding them
    if cluster_spawn:
        _cluster_spawn(env, agents, radius=cluster_radius)
        # Debug: initial pairwise distances
        if len(agents) >= 2:
            import numpy as _np
            p = [a.body.state.position for a in agents]
            for i in range(len(p)):
                for j in range(i+1, len(p)):
                    d = float(_np.hypot(p[i][0]-p[j][0], p[i][1]-p[j][1]))
                    print(f"   init distance {agents[i].agent_id}-{agents[j].agent_id}: {d:.2f}")

    # CREATE KNOWLEDGE GAP: Make Agent_0 an expert on dangerous fruit
    if agents:
        expert_agent = agents[0]
        # Give it 5 experiences of crimson fruit being very dangerous
        expert_agent.embodied_mappings['crimson_fruit'].extend([-0.9] * 5)
        print(f"🧠 Priming {expert_agent.agent_id} with expert knowledge: crimson_fruit is VERY dangerous!")
        print(f"   {expert_agent.agent_id} now knows: crimson_fruit = {np.mean(expert_agent.embodied_mappings['crimson_fruit']):.2f}")

    print(f"🎯 SOCIAL LEARNING CONDITIONS SET:")
    print(f"   ✅ Expert agent with critical survival knowledge")
    print(f"   ✅ Dangerous fruit that will create learning pressure")
    print(f"   ✅ Teaching/learning thresholds optimized")
    print(f"   ✅ Existential pressure drives active exploration")
    print(f"   ✅ Social strategies connected to embodied goals")

    # Run social simulation
    print("\n🚀 Running enhanced social simulation...")
    all_results = []

    for step in range(steps):
        if step % 100 == 0:
            print(f"  Step {step}/{steps}")
            # Print social status
            for agent in agents:
                print(f"    {agent.agent_id}: Strategy={agent.current_social_strategy}, "
                      f"Energy={agent.body.state.energy:.2f}, "
                      f"Context={agent.context.get_current()}, "
                      f"Memories={len(agent.episodic_surplus_memory)}")

        step_results = env.step_all_agents()
        all_results.append(step_results)

    # Enhanced Analysis
    print("\n📊 ENHANCED SOCIAL INTERACTION ANALYSIS")
    print("=" * 50)

    # Spatial dynamics analysis
    analyze_spatial_dynamics(agents, steps)

    # Social development analysis
    for agent in agents:
        print(f"\n🤖 {agent.agent_id} Enhanced Profile:")
        print(f"  Final Personality: {agent.social_personality}")
        print(f"  Social Memory Events: {len(agent.social_memory)}")
        print(f"  Episodic Surplus Memories: {len(agent.episodic_surplus_memory)}")
        print(f"  Relationships: {dict(agent.social_relationships)}")
        print(f"  Knowledge Transfers: {sum(len(transfers) for transfers in agent.social_knowledge_transfer.values())}")
        print(f"  Position History Length: {len(agent.position_history)}")

        # Show embodied mappings (fruit knowledge)
        if agent.embodied_mappings:
            print(f"  Embodied Knowledge:")
            for category, values in agent.embodied_mappings.items():
                avg_value = np.mean(values)
                print(f"    {category}: {len(values)} experiences, avg={avg_value:.2f}")

        # Analyze proximity patterns
        if agent.proximity_history:
            print(f"  Proximity patterns:")
            for other_id, distances in agent.proximity_history.items():
                if distances:
                    avg_dist = np.mean(distances)
                    close_encounters = sum(1 for d in distances if d <= CLOSE_RANGE)
                    print(f"    To {other_id}: avg_distance={avg_dist:.2f}, close_encounters={close_encounters}")

    # Create enhanced visualization
    create_enhanced_social_visualization(env, agents, all_results)

    return env, agents, all_results

def main():
    """Run the enhanced social QSE-Émile experiment"""

    env, agents, results = run_social_experiment(n_agents=3, steps=1000)
    return env, agents, results

if __name__ == "__main__":
    env, agents, results = main()
