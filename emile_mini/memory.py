
import random
import time
from .config import CONFIG

class MemoryModule:
    """
    Hierarchical memory for Émile:
      - working: recent items (FIFO, capacity limited)
      - episodic: tagged event history (capacity limited)
      - semantic: key→value store for facts/concepts

    Retrieval uses a 'richness' parameter to vary creativity.
    ENHANCED: Better structured storage and validation
    """

    def __init__(self, cfg=CONFIG):
        self.cfg = cfg

        # Working memory (short-term)
        self.working = []
        self.working_capacity = 10

        # Episodic memory (events) - Much larger capacity for experiments
        self.episodic = []
        self.episodic_capacity = getattr(cfg, 'EPISODIC_MEMORY_CAPACITY', 1000)  # Was 100, now 1000!

        # Semantic memory (facts)
        self.semantic = {}
        
        # Memory statistics for debugging
        self.stats = {
            'total_stores': 0,
            'episodic_stores': 0,
            'semantic_stores': 0,
            'working_stores': 0,
            'validation_errors': 0
        }

    def store(self, info, tags=None):
        """
        Store 'info' in memory.
        - Always push into working memory.
        - If tags['type']=='episodic', also append to episodic.
        - If tags['type']=='semantic' and tags['key'] exists, store in semantic.
        ENHANCED: Better validation and JSON-safe storage
        """
        self.stats['total_stores'] += 1
        
        # NEW: Validate and sanitize info for JSON safety
        sanitized_info = self._sanitize_for_json(info)
        
        # 1) Working memory (most recent first)
        self.working.insert(0, sanitized_info)
        self.stats['working_stores'] += 1
        if len(self.working) > self.working_capacity:
            self.working.pop()

        # 2) Episodic or semantic
        if tags:
            mtype = tags.get("type")
            if mtype == "episodic":
                # NEW: Add timestamp and validate structure
                entry = {
                    "data": sanitized_info, 
                    "tags": tags.copy(),
                    "timestamp": time.time(),
                    "memory_id": self.stats['episodic_stores']
                }
                
                # Validate essential fields for episodic memories
                if self._validate_episodic_entry(sanitized_info):
                    self.episodic.append(entry)
                    self.stats['episodic_stores'] += 1
                    if len(self.episodic) > self.episodic_capacity:
                        self.episodic.pop(0)
                else:
                    self.stats['validation_errors'] += 1
                    
            elif mtype == "semantic":
                key = tags.get("key")
                if key is not None:
                    self.semantic[key] = sanitized_info
                    self.stats['semantic_stores'] += 1

    def _sanitize_for_json(self, obj):
        """Ensure object is JSON-serializable"""
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, dict):
            return {str(k): self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(item) for item in obj]
        elif hasattr(obj, '__iter__') and not isinstance(obj, str):
            return [self._sanitize_for_json(item) for item in obj]
        else:
            # Convert complex objects to string representation
            return str(obj)
    
    def _validate_episodic_entry(self, info):
        """Validate that episodic memory entry has essential QSE fields"""
        if not isinstance(info, dict):
            return True  # Allow non-dict entries for backward compatibility
            
        # Check for essential QSE cognitive fields
        essential_fields = ['surplus', 'context']
        has_essential = any(field in info for field in essential_fields)
        
        # Check for step tracking
        has_step = 'step' in info
        
        return has_essential or has_step

    def store_structured_episodic(self, step=None, position=None, energy=None, 
                                 context=None, sigma=None, surplus=None, 
                                 goal=None, reward=None, **kwargs):
        """
        NEW: Structured episodic memory storage with explicit parameters
        Ensures all key QSE fields are captured consistently
        """
        entry = {
            'step': step,
            'position': position,
            'energy': energy,
            'context': context,
            'sigma': sigma,
            'surplus': surplus,
            'goal': goal,
            'reward': reward
        }
        
        # Add any additional fields
        entry.update(kwargs)
        
        # Remove None values to keep entries clean
        entry = {k: v for k, v in entry.items() if v is not None}
        
        self.store(entry, tags={'type': 'episodic', 'structured': True})

    def retrieve(self, query=None, richness=0.0):
        """
        Retrieve a memory item.
        - If query matches a semantic key, return that.
        - Otherwise use episodic memory:
            richness=0.0 -> most recent event
            richness=1.0 -> uniform random
            in between -> bias toward older entries proportionally
        Returns None if nothing found.
        """
        # Semantic lookup
        if query is not None and query in self.semantic:
            return self.semantic[query]

        # No episodic memories?
        if not self.episodic:
            return None

        # Richness controls distance:
        n = len(self.episodic)
        if richness <= 0.0:
            return self.episodic[-1]["data"]
        if richness >= 1.0:
            return random.choice(self.episodic)["data"]

        # Bias index: 0 -> most recent, n-1 -> oldest
        idx = int(richness * (n - 1))
        return self.episodic[-1 - idx]["data"]

    def get_working(self):
        """Return a copy of working memory list."""
        return list(self.working)

    def get_episodic(self):
        """Return a copy of episodic memory entries."""
        return list(self.episodic)

    def get_semantic(self):
        """Return a copy of the semantic memory dict."""
        return dict(self.semantic)
    
    def get_stats(self):
        """NEW: Return memory usage statistics"""
        stats = self.stats.copy()
        stats.update({
            'working_count': len(self.working),
            'episodic_count': len(self.episodic),
            'semantic_count': len(self.semantic),
            'episodic_utilization': len(self.episodic) / self.episodic_capacity,
            'working_utilization': len(self.working) / self.working_capacity
        })
        return stats
    
    def search_episodic(self, field, value):
        """NEW: Search episodic memories by field value"""
        matches = []
        for entry in self.episodic:
            if isinstance(entry['data'], dict) and entry['data'].get(field) == value:
                matches.append(entry)
        return matches
