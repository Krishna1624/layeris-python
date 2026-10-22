"""Simulated expert cache and prefetch execution pipeline for Layeris MoE evaluation.

Models:
1. Finite capacity expert cache with bounded residency per layer or global budget.
2. LRU and Predictive-LFRU eviction policies.
3. Realistic asynchronous prefetch lead-time, background I/O queueing, and pipeline stalls.
4. Separate accounting of measured CPU execution time versus simulated I/O stall latency.
5. Accurate tracking of useful vs wasted prefetches, timely vs late arrivals, and cache hit rates.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class CacheEntry:
    """State of an expert slab resident or in-flight in cache."""
    layer: int
    expert_id: int
    admitted_time_us: float
    last_accessed_time_us: float
    access_count: int = 1
    is_prefetched: bool = False
    was_accessed: bool = False
    ready_time_us: float = 0.0  # When I/O load completes


@dataclass
class SimulationStats:
    """Cumulative simulation metrics for a cache and prefetch run."""
    total_expert_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    demand_loads: int = 0
    prefetch_dispatches: int = 0
    redundant_prefetches_filtered: int = 0
    timely_prefetches: int = 0
    late_prefetches: int = 0
    useful_prefetches: int = 0
    wasted_prefetches: int = 0
    simulated_compute_time_us: float = 0.0
    simulated_stall_time_us: float = 0.0
    simulated_total_time_us: float = 0.0
    measured_overhead_us: float = 0.0
    peak_resident_experts: int = 0
    disk_bytes_read: int = 0

    @property
    def hit_rate(self) -> float:
        return (
            (self.cache_hits / self.total_expert_requests) * 100.0
            if self.total_expert_requests > 0
            else 0.0
        )

    @property
    def miss_rate(self) -> float:
        return (
            (self.cache_misses / self.total_expert_requests) * 100.0
            if self.total_expert_requests > 0
            else 0.0
        )

    @property
    def useful_prefetch_rate(self) -> float:
        return (
            (self.useful_prefetches / self.prefetch_dispatches) * 100.0
            if self.prefetch_dispatches > 0
            else 0.0
        )

    @property
    def wasted_prefetch_rate(self) -> float:
        return (
            (self.wasted_prefetches / self.prefetch_dispatches) * 100.0
            if self.prefetch_dispatches > 0
            else 0.0
        )

    @property
    def timely_rate(self) -> float:
        return (
            (self.timely_prefetches / self.prefetch_dispatches) * 100.0
            if self.prefetch_dispatches > 0
            else 0.0
        )


class ExpertCacheSimulator:
    """Finite capacity expert cache with asynchronous I/O and prefetch simulation."""

    def __init__(
        self,
        capacity_per_layer: int = 4,
        num_layers: int = 32,
        num_experts: int = 64,
        expert_size_bytes: int = 16 * 1024 * 1024,
        layer_compute_us: float = 120.0,
        expert_load_us: float = 250.0,
        max_concurrent_io: int = 16,
        eviction_policy: str = "lru",  # "lru" or "predictive_lfru"
    ):
        self.capacity_per_layer = capacity_per_layer
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.expert_size_bytes = expert_size_bytes
        self.layer_compute_us = layer_compute_us
        self.expert_load_us = expert_load_us
        self.max_concurrent_io = max_concurrent_io
        self.eviction_policy = eviction_policy.lower()

        # [layer] -> OrderedDict of expert_id -> CacheEntry
        self.cache: List[collections.OrderedDict[int, CacheEntry]] = [
            collections.OrderedDict() for _ in range(num_layers)
        ]

        # In-flight prefetch jobs: (layer, expert_id) -> finish_time_us
        self.in_flight_io: Dict[Tuple[int, int], float] = {}
        # Timeline state
        self.current_time_us: float = 0.0
        # Priority protected experts from recent prediction: set of (layer, expert_id)
        self.protected_experts: Set[Tuple[int, int]] = set()

        self.stats = SimulationStats()

    def set_protected_experts(self, keys: List[Tuple[int, int]]) -> None:
        """Register upcoming predicted experts to protect against premature eviction (Predictive LFRU)."""
        self.protected_experts = set(keys)

    def is_resident(self, layer: int, expert_id: int) -> bool:
        """Returns True if expert slab is in cache and I/O load is finished."""
        if not (0 <= layer < self.num_layers):
            return False
        entry = self.cache[layer].get(expert_id)
        if entry is None:
            return False
        return self.current_time_us >= entry.ready_time_us

    def is_in_cache_or_inflight(self, layer: int, expert_id: int) -> bool:
        """Returns True if expert is either cached or currently loading."""
        if not (0 <= layer < self.num_layers):
            return False
        if expert_id in self.cache[layer]:
            return True
        return (layer, expert_id) in self.in_flight_io

    def _evict_one(self, layer: int) -> Optional[int]:
        """Evict one expert from layer cache according to configured policy."""
        layer_cache = self.cache[layer]
        if not layer_cache:
            return None

        evict_id = None
        if self.eviction_policy == "predictive_lfru":
            # Find candidate with lowest utility, favoring unprotected and low-frequency slabs
            best_candidate = None
            lowest_score = float("inf")

            for eid, entry in layer_cache.items():
                is_protected = (layer, eid) in self.protected_experts
                # Utility score combines recency, frequency, and predictive protection
                recency_delta = max(1.0, self.current_time_us - entry.last_accessed_time_us)
                freq_score = float(entry.access_count) / recency_delta
                # Heavily boost protected upcoming experts to avoid evicting them before arrival
                protection_boost = 1000.0 if is_protected else 0.0
                score = freq_score + protection_boost

                if score < lowest_score:
                    lowest_score = score
                    best_candidate = eid

            evict_id = best_candidate if best_candidate is not None else next(iter(layer_cache))
        else:
            # Standard LRU: evict oldest entry
            evict_id = next(iter(layer_cache))

        # Check if evicted entry was a wasted prefetch
        evicted_entry = layer_cache.pop(evict_id)
        if evicted_entry.is_prefetched and not evicted_entry.was_accessed:
            self.stats.wasted_prefetches += 1

        return evict_id

    def dispatch_prefetch(self, layer: int, expert_id: int) -> bool:
        """Asynchronously prefetch an expert slab into cache if not already present."""
        if not (0 <= layer < self.num_layers) or not (0 <= expert_id < self.num_experts):
            return False

        # Redundancy check: if already resident or in-flight, suppress duplicate I/O
        if self.is_in_cache_or_inflight(layer, expert_id):
            self.stats.redundant_prefetches_filtered += 1
            return False

        # Check if layer cache is at capacity; evict if needed to reserve slot
        if len(self.cache[layer]) >= self.capacity_per_layer:
            self._evict_one(layer)

        # Calculate I/O completion time considering concurrent background I/O queue
        active_io_count = len(self.in_flight_io)
        queue_delay = 0.0
        if active_io_count >= self.max_concurrent_io:
            queue_delay = (active_io_count - self.max_concurrent_io + 1) * (self.expert_load_us * 0.25)

        ready_time = self.current_time_us + self.expert_load_us + queue_delay
        self.in_flight_io[(layer, expert_id)] = ready_time

        entry = CacheEntry(
            layer=layer,
            expert_id=expert_id,
            admitted_time_us=self.current_time_us,
            last_accessed_time_us=self.current_time_us,
            access_count=0,
            is_prefetched=True,
            was_accessed=False,
            ready_time_us=ready_time,
        )
        self.cache[layer][expert_id] = entry

        self.stats.prefetch_dispatches += 1
        self.stats.disk_bytes_read += self.expert_size_bytes

        # Update peak resident count
        total_resident = sum(len(c) for c in self.cache)
        if total_resident > self.stats.peak_resident_experts:
            self.stats.peak_resident_experts = total_resident

        return True

    def access_expert(self, layer: int, expert_id: int) -> float:
        """Simulate demand access for an expert at current layer.
        
        Returns stall time incurred (0.0 if hit, >0 if late prefetch or demand miss).
        """
        self.stats.total_expert_requests += 1
        stall_us = 0.0

        entry = self.cache[layer].get(expert_id)

        if entry is not None:
            # Entry is in cache; check if ready or late
            if self.current_time_us >= entry.ready_time_us:
                # Timely hit!
                self.stats.cache_hits += 1
                if entry.is_prefetched:
                    if not entry.was_accessed:
                        self.stats.useful_prefetches += 1
                    self.stats.timely_prefetches += 1
                entry.was_accessed = True
                entry.access_count += 1
                entry.last_accessed_time_us = self.current_time_us
                # Move to end for LRU recency
                self.cache[layer].move_to_end(expert_id)
            else:
                # Late prefetch hit: I/O was initiated but not yet finished
                remaining_load = entry.ready_time_us - self.current_time_us
                stall_us = remaining_load
                self.current_time_us += stall_us
                self.stats.cache_hits += 1
                if entry.is_prefetched:
                    if not entry.was_accessed:
                        self.stats.useful_prefetches += 1
                    self.stats.late_prefetches += 1
                entry.was_accessed = True
                entry.access_count += 1
                entry.last_accessed_time_us = self.current_time_us
                self.cache[layer].move_to_end(expert_id)
        else:
            # Demand miss: expert is not in cache at all. Incur full synchronous load.
            self.stats.cache_misses += 1
            self.stats.demand_loads += 1
            self.stats.disk_bytes_read += self.expert_size_bytes

            stall_us = self.expert_load_us
            self.current_time_us += stall_us

            # Evict if full
            if len(self.cache[layer]) >= self.capacity_per_layer:
                self._evict_one(layer)

            new_entry = CacheEntry(
                layer=layer,
                expert_id=expert_id,
                admitted_time_us=self.current_time_us,
                last_accessed_time_us=self.current_time_us,
                access_count=1,
                is_prefetched=False,
                was_accessed=True,
                ready_time_us=self.current_time_us,
            )
            self.cache[layer][expert_id] = new_entry

        # Cleanup completed in-flight jobs
        completed = [k for k, t in self.in_flight_io.items() if t <= self.current_time_us]
        for k in completed:
            del self.in_flight_io[k]

        total_resident = sum(len(c) for c in self.cache)
        if total_resident > self.stats.peak_resident_experts:
            self.stats.peak_resident_experts = total_resident

        self.stats.simulated_stall_time_us += stall_us
        return stall_us

    def step_layer_compute(self) -> None:
        """Advance time by layer computation latency."""
        self.current_time_us += self.layer_compute_us
        self.stats.simulated_compute_time_us += self.layer_compute_us

        # Cleanup completed I/O jobs
        completed = [k for k, t in self.in_flight_io.items() if t <= self.current_time_us]
        for k in completed:
            del self.in_flight_io[k]

    def finalize(self) -> SimulationStats:
        """Finalize simulation, accounting for any remaining unhit prefetches in cache."""
        for layer in range(self.num_layers):
            for eid, entry in self.cache[layer].items():
                if entry.is_prefetched and not entry.was_accessed:
                    self.stats.wasted_prefetches += 1

        self.stats.simulated_total_time_us = (
            self.stats.simulated_compute_time_us + self.stats.simulated_stall_time_us
        )
        return self.stats


# Alias for intuitive API naming
PrefetchCacheSimulator = ExpertCacheSimulator
