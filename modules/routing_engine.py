"""
GraphRoutingEngine
==================
Traffic-aware emergency vehicle routing using a synthetic municipal
road network built with NetworkX.

Design:
  - 10 × 10 grid of intersections (100 nodes, ~180 directed edges).
  - Each edge carries: base_distance (km), road_type, traffic_level.
  - Composite edge weight = base_dist × road_multiplier × traffic × vehicle_penalty.
  - Dijkstra shortest path (nx.shortest_path) used for routing.
  - Different emergency vehicle types penalise different road types.
"""

from __future__ import annotations

import math
import random
import logging
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_SIZE = 10          # 10×10 grid → 100 nodes

ROAD_TYPES = ["highway", "arterial", "local", "service_lane"]

ROAD_TYPE_MULTIPLIER: dict[str, float] = {
    "highway":      0.6,   # fastest
    "arterial":     1.0,
    "local":        1.4,
    "service_lane": 2.2,   # slowest / narrowest
}

# Traffic congestion levels
TRAFFIC_LEVELS = ["clear", "moderate", "heavy", "gridlock"]
TRAFFIC_MULTIPLIER: dict[str, float] = {
    "clear":    1.0,
    "moderate": 1.5,
    "heavy":    2.5,
    "gridlock": 4.0,
}

# Vehicle types and which road types they are penalised on
VEHICLE_PROFILES: dict[str, dict] = {
    "fire_engine": {
        "label":   "🚒 Fire Engine",
        "penalty_roads": {"service_lane": 3.5, "local": 1.8},
        "color":   "#FF6B35",
    },
    "ambulance": {
        "label":   "🚑 Ambulance",
        "penalty_roads": {"service_lane": 1.3},
        "color":   "#4ECDC4",
    },
    "police_car": {
        "label":   "👮 Police Car",
        "penalty_roads": {},
        "color":   "#45B7D1",
    },
    "motorbike_medic": {
        "label":   "🏍️ Motorbike Medic",
        "penalty_roads": {},   # smallest — no penalty
        "color":   "#96CEB4",
    },
    "hazmat_unit": {
        "label":   "☣️ HazMat Unit",
        "penalty_roads": {"service_lane": 4.0, "local": 2.0},
        "color":   "#FFEAA7",
    },
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class RouteResult:
    vehicle_type: str
    source_node: int
    target_node: int
    path: list[int]
    path_coords: list[tuple[float, float]]   # (lat, lon) for each node
    total_cost: float
    estimated_time_min: float
    road_segments: list[dict]               # per-edge details
    congestion_zones: list[int]             # node indices with heavy traffic ahead


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

class GraphRoutingEngine:
    """
    Builds and maintains a synthetic municipal road network graph.
    Provides Dijkstra-based routing for any emergency vehicle type.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.G: nx.DiGraph = nx.DiGraph()
        self._node_coords: dict[int, tuple[float, float]] = {}
        self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _node_id(self, r: int, c: int) -> int:
        return r * GRID_SIZE + c

    def _build_graph(self) -> None:
        """Construct a 10×10 directed grid graph with realistic road attributes."""
        logger.info("Building municipal road network graph …")

        # Base coordinates: a fictional city near 28.6°N 77.2°E (Delhi region)
        base_lat, base_lon = 28.60, 77.20
        cell_deg = 0.005   # ~550 m per cell

        # Add nodes
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                nid = self._node_id(r, c)
                lat = base_lat + r * cell_deg + self._rng.uniform(-0.0005, 0.0005)
                lon = base_lon + c * cell_deg + self._rng.uniform(-0.0005, 0.0005)
                self._node_coords[nid] = (lat, lon)
                self.G.add_node(nid, lat=lat, lon=lon, row=r, col=c)

        # Add directed edges (bidirectional — two directed edges per pair)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                nid = self._node_id(r, c)
                neighbors = []
                if c + 1 < GRID_SIZE:
                    neighbors.append(self._node_id(r, c + 1))
                if r + 1 < GRID_SIZE:
                    neighbors.append(self._node_id(r + 1, c))

                for nb in neighbors:
                    road_type = self._assign_road_type(r, c)
                    traffic   = self._assign_traffic()
                    dist      = self._haversine(
                        self._node_coords[nid],
                        self._node_coords[nb],
                    )
                    self.G.add_edge(
                        nid, nb,
                        base_distance=dist,
                        road_type=road_type,
                        traffic_level=traffic,
                    )
                    # Reverse edge (may have different traffic)
                    self.G.add_edge(
                        nb, nid,
                        base_distance=dist,
                        road_type=road_type,
                        traffic_level=self._assign_traffic(),
                    )

        logger.info(
            "Graph built: %d nodes, %d edges", self.G.number_of_nodes(), self.G.number_of_edges()
        )

    def _assign_road_type(self, r: int, c: int) -> str:
        """
        Heuristic: edges along the outer ring and centre diagonals are
        highways/arterials; inner roads are local/service lanes.
        """
        if r == 0 or r == GRID_SIZE - 1 or c == 0 or c == GRID_SIZE - 1:
            return self._rng.choice(["highway", "arterial"])
        if r in (4, 5) or c in (4, 5):
            return "arterial"
        return self._rng.choice(["local", "service_lane", "local"])

    def _assign_traffic(self) -> str:
        weights = [0.35, 0.35, 0.20, 0.10]
        return self._rng.choices(TRAFFIC_LEVELS, weights=weights)[0]

    @staticmethod
    def _haversine(coord_a: tuple[float, float], coord_b: tuple[float, float]) -> float:
        """Return great-circle distance in km."""
        lat1, lon1 = math.radians(coord_a[0]), math.radians(coord_a[1])
        lat2, lon2 = math.radians(coord_b[0]), math.radians(coord_b[1])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 6371.0 * 2 * math.asin(math.sqrt(a))

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(
        self,
        source_node: Optional[int] = None,
        target_node: Optional[int] = None,
        vehicle_type: str = "ambulance",
    ) -> RouteResult:
        """
        Compute the optimal route for the given vehicle type using Dijkstra.

        Parameters
        ----------
        source_node : int, optional — defaults to depot (node 0)
        target_node : int, optional — defaults to a random node
        vehicle_type : str — key into VEHICLE_PROFILES
        """
        if vehicle_type not in VEHICLE_PROFILES:
            vehicle_type = "ambulance"

        if source_node is None:
            source_node = 0
        if target_node is None:
            target_node = self._rng.randint(50, 99)

        profile = VEHICLE_PROFILES[vehicle_type]
        penalty_roads = profile["penalty_roads"]

        # Assign composite weights
        for u, v, data in self.G.edges(data=True):
            road_type    = data["road_type"]
            traffic      = data["traffic_level"]
            base_dist    = data["base_distance"]
            road_mult    = ROAD_TYPE_MULTIPLIER[road_type]
            traffic_mult = TRAFFIC_MULTIPLIER[traffic]
            veh_penalty  = penalty_roads.get(road_type, 1.0)

            cost = base_dist * road_mult * traffic_mult * veh_penalty
            self.G[u][v]["cost"] = cost

        try:
            path = nx.shortest_path(self.G, source=source_node, target=target_node, weight="cost")
        except nx.NetworkXNoPath:
            # Fallback to nearest reachable node
            logger.warning("No path found; using fallback target.")
            reachable = list(nx.single_source_dijkstra_path_length(self.G, source_node).keys())
            target_node = reachable[len(reachable) // 2]
            path = nx.shortest_path(self.G, source=source_node, target=target_node, weight="cost")

        # Compute total cost and per-segment details
        total_cost = 0.0
        segments: list[dict] = []
        congestion_zones: list[int] = []

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge = self.G[u][v]
            seg_cost = edge["cost"]
            total_cost += seg_cost
            segments.append({
                "from": u,
                "to":   v,
                "road_type":     edge["road_type"],
                "traffic_level": edge["traffic_level"],
                "distance_km":   round(edge["base_distance"], 3),
                "cost":          round(seg_cost, 4),
            })
            if edge["traffic_level"] in ("heavy", "gridlock"):
                congestion_zones.append(v)

        # Estimated time (minutes): cost ≈ effective km; avg emergency speed 60 km/h
        estimated_time = (total_cost / 60.0) * 60.0  # minutes
        # Clamp to realistic range for a ~5 km city grid
        estimated_time = max(1.5, min(estimated_time, 45.0))

        path_coords = [self._node_coords[n] for n in path]

        return RouteResult(
            vehicle_type=vehicle_type,
            source_node=source_node,
            target_node=target_node,
            path=path,
            path_coords=path_coords,
            total_cost=round(total_cost, 3),
            estimated_time_min=round(estimated_time, 1),
            road_segments=segments,
            congestion_zones=list(set(congestion_zones)),
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def refresh_traffic(self) -> None:
        """Randomise traffic levels to simulate real-time conditions."""
        for u, v in self.G.edges():
            self.G[u][v]["traffic_level"] = self._assign_traffic()

    def node_coords(self, node: int) -> tuple[float, float]:
        return self._node_coords[node]

    @property
    def all_node_coords(self) -> dict[int, tuple[float, float]]:
        return self._node_coords.copy()

    @property
    def vehicle_profiles(self) -> dict[str, dict]:
        return VEHICLE_PROFILES

    @property
    def graph(self) -> nx.DiGraph:
        return self.G
