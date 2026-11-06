"""
Geospatial Catchment, Trade Area Gravity & Store Cannibalization Engine.

Provides:
1. Huff Gravity Retail Trade Area Model:
   - Computes customer shopping attraction probability:
     P_ij = (S_j^lambda / d_ij^beta) / sum_k (S_k^lambda / d_ik^beta)
   - Calibrates distance decay friction exponent (beta) and store attractiveness elasticity (lambda).
   - Accurately computes geographic Haversine distances between customer demand tracts and store nodes.
2. Catchment Trade Area Zoning:
   - Profiles Primary Trade Area (60% patronage contour), Secondary (80%), and Tertiary (95%).
   - Calculates customer population reach, average drive distance, and demographic spending capture.
3. Multi-Store Spatial Cannibalization & Market Overlap Forecaster:
   - Simulates adding a new candidate store into the metropolitan retail network.
   - Quantifies gross revenue of new store vs cannibalization leakage drained from sister stores.
   - Computes true Net Incremental Market Expansion Revenue and Cannibalization Ratio.
4. Candidate Expansion Site Selection Multi-Criteria Frontier:
   - Ranks expansion sites by spatial efficiency, cannibalization resistance, and net ROI.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import math
import numpy as np
import pandas as pd

from src.analytics import RetailAnalyticsEngine


class GeospatialGravityEngine:
    """
    Enterprise retail geospatial market analytics, Huff gravity trade catchment,
    and sister-store cannibalization forecasting engine.
    """

    # Existing Retail Store Network Nodes
    STORE_NETWORK: Dict[str, Dict[str, Any]] = {
        "STORE_A_MANHATTAN": {
            "store_id": "STORE_A_MANHATTAN",
            "name": "Store A - Flagship Downtown (Manhattan)",
            "lat": 40.7580,
            "lon": -73.9855,
            "sqft": 45000,
            "attractiveness_score": 92.0,
            "annual_sales_baseline": 185000.0,
            "merchandise_depth_tier": "Full Departmental",
        },
        "STORE_B_WHITE_PLAINS": {
            "store_id": "STORE_B_WHITE_PLAINS",
            "name": "Store B - Suburban North (White Plains)",
            "lat": 41.0339,
            "lon": -73.7629,
            "sqft": 28000,
            "attractiveness_score": 78.0,
            "annual_sales_baseline": 142000.0,
            "merchandise_depth_tier": "Standard Retail",
        },
        "STORE_C_JERSEY_CITY": {
            "store_id": "STORE_C_JERSEY_CITY",
            "name": "Store C - Metro West (Jersey City)",
            "lat": 40.7178,
            "lon": -74.0431,
            "sqft": 32000,
            "attractiveness_score": 81.0,
            "annual_sales_baseline": 129000.0,
            "merchandise_depth_tier": "Premium Focus",
        },
    }

    # Representative Census Demand Tracts across Metropolitan Trade Area
    DEMAND_TRACTS: List[Dict[str, Any]] = [
        {"tract_id": "TRACT_01", "name": "Midtown East", "lat": 40.7549, "lon": -73.9722, "population": 45000, "annual_retail_spend_millions": 18.5},
        {"tract_id": "TRACT_02", "name": "Upper West Side", "lat": 40.7870, "lon": -73.9754, "population": 62000, "annual_retail_spend_millions": 24.0},
        {"tract_id": "TRACT_03", "name": "Lower Manhattan / Wall St", "lat": 40.7075, "lon": -74.0090, "population": 38000, "annual_retail_spend_millions": 19.2},
        {"tract_id": "TRACT_04", "name": "Brooklyn Heights / DUMBO", "lat": 40.6960, "lon": -73.9933, "population": 55000, "annual_retail_spend_millions": 21.4},
        {"tract_id": "TRACT_05", "name": "Queens - Astoria", "lat": 40.7644, "lon": -73.9235, "population": 70000, "annual_retail_spend_millions": 22.8},
        {"tract_id": "TRACT_06", "name": "South Bronx", "lat": 40.8160, "lon": -73.9180, "population": 65000, "annual_retail_spend_millions": 16.5},
        {"tract_id": "TRACT_07", "name": "Westchester Central", "lat": 41.0100, "lon": -73.7800, "population": 48000, "annual_retail_spend_millions": 25.0},
        {"tract_id": "TRACT_08", "name": "Bergen County NJ", "lat": 40.9100, "lon": -74.0700, "population": 52000, "annual_retail_spend_millions": 26.5},
        {"tract_id": "TRACT_09", "name": "Hoboken / Newport", "lat": 40.7440, "lon": -74.0324, "population": 40000, "annual_retail_spend_millions": 19.0},
        {"tract_id": "TRACT_10", "name": "Staten Island North", "lat": 40.6400, "lon": -74.0800, "population": 35000, "annual_retail_spend_millions": 14.0},
    ]

    def __init__(self, data_path: Optional[Path] = None, seed: int = 42):
        self.analytics = RetailAnalyticsEngine(data_path=data_path)
        self.df = self.analytics.df.copy()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Computes great-circle distance in kilometers between two latitude/longitude coordinates."""
        r = 6371.0  # Earth radius in km
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2)
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return float(r * c)

    def get_network_catalog(self) -> Dict[str, Any]:
        """Returns metadata on active store network nodes and demand tract demographics."""
        return {
            "total_active_stores": len(self.STORE_NETWORK),
            "store_nodes": self.STORE_NETWORK,
            "total_demand_tracts": len(self.DEMAND_TRACTS),
            "demand_tracts": self.DEMAND_TRACTS,
            "total_catchment_population": sum(t["population"] for t in self.DEMAND_TRACTS),
            "total_market_retail_expenditure_millions": round(sum(t["annual_retail_spend_millions"] for t in self.DEMAND_TRACTS), 2),
        }

    def compute_huff_probabilities(
        self,
        stores: Dict[str, Dict[str, Any]],
        beta: float = 1.75,
        lambda_param: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Computes Huff gravity attraction probabilities P_ij across all demand tracts and store nodes.
        Formula: P_ij = (S_j^lambda / d_ij^beta) / sum_k (S_k^lambda / d_ik^beta)
        """
        if beta <= 0.0:
            raise ValueError("Distance decay exponent (beta) must be strictly positive.")
        if lambda_param <= 0.0:
            raise ValueError("Attractiveness elasticity (lambda) must be strictly positive.")

        results = []
        for tract in self.DEMAND_TRACTS:
            t_lat, t_lon = tract["lat"], tract["lon"]
            utilities = {}
            distances = {}

            for s_id, s_info in stores.items():
                d_km = max(0.5, self.haversine_distance_km(t_lat, t_lon, s_info["lat"], s_info["lon"]))
                attractiveness = float(s_info.get("attractiveness_score", 80.0))
                # Huff utility numerator: S^lambda / d^beta
                u = (attractiveness ** lambda_param) / (d_km ** beta)
                utilities[s_id] = u
                distances[s_id] = round(d_km, 2)

            # Outside competitor retail options in metropolitan trade area
            competitor_utility = 1.25
            total_utility = sum(utilities.values()) + competitor_utility
            probs = {s_id: round(u / total_utility, 4) for s_id, u in utilities.items()}
            probs["COMPETITORS"] = round(competitor_utility / total_utility, 4)

            # Calculate tract dollars captured per store
            spend_m = tract["annual_retail_spend_millions"]
            captured_spend = {s_id: round(probs[s_id] * spend_m, 3) for s_id in stores.keys()}

            results.append({
                "tract_id": tract["tract_id"],
                "tract_name": tract["name"],
                "population": tract["population"],
                "annual_spend_millions": spend_m,
                "distances_km": distances,
                "huff_patronage_probabilities": probs,
                "captured_spend_millions": captured_spend,
            })

        return results

    def compute_huff_market_shares(
        self,
        beta: float = 1.75,
        lambda_param: float = 1.0,
    ) -> Dict[str, Any]:
        """Computes aggregate baseline store market shares and revenue capture under Huff Gravity."""
        tract_probs = self.compute_huff_probabilities(self.STORE_NETWORK, beta=beta, lambda_param=lambda_param)

        total_market_spend = sum(t["annual_retail_spend_millions"] for t in self.DEMAND_TRACTS)
        store_captured = {s_id: 0.0 for s_id in self.STORE_NETWORK.keys()}
        store_pop_reach = {s_id: 0.0 for s_id in self.STORE_NETWORK.keys()}

        for item in tract_probs:
            for s_id, spend in item["captured_spend_millions"].items():
                if s_id in store_captured:
                    store_captured[s_id] += spend
            for s_id, prob in item["huff_patronage_probabilities"].items():
                if s_id in store_pop_reach:
                    store_pop_reach[s_id] += prob * item["population"]

        summary = {}
        for s_id, cap_spend in store_captured.items():
            s_name = self.STORE_NETWORK[s_id]["name"]
            share_pct = (cap_spend / total_market_spend) * 100.0
            summary[s_id] = {
                "store_name": s_name,
                "captured_market_spend_millions": round(cap_spend, 2),
                "market_share_pct": round(share_pct, 2),
                "effective_catchment_population": int(round(store_pop_reach[s_id])),
                "attractiveness_score": self.STORE_NETWORK[s_id]["attractiveness_score"],
            }

        return {
            "parameters": {"beta_distance_decay": beta, "lambda_elasticity": lambda_param},
            "store_market_shares": summary,
            "total_market_spend_millions": round(total_market_spend, 2),
            "demand_tracts_evaluation": tract_probs,
        }

    def calculate_catchment_trade_areas(
        self,
        store_id: str,
        beta: float = 1.75,
    ) -> Dict[str, Any]:
        """
        Profiles Primary (60%), Secondary (80%), and Tertiary (95%) trade area catchment boundaries,
        population reach, and average travel distance for a specific retail store.
        """
        if store_id not in self.STORE_NETWORK:
            raise ValueError(f"Store '{store_id}' not found. Available stores: {list(self.STORE_NETWORK.keys())}")

        tract_probs = self.compute_huff_probabilities(self.STORE_NETWORK, beta=beta)

        # Sort tracts by descending patronage probability for this store
        sorted_tracts = sorted(
            tract_probs,
            key=lambda t: t["huff_patronage_probabilities"][store_id],
            reverse=True,
        )

        total_captured = sum(t["captured_spend_millions"][store_id] for t in sorted_tracts)
        cum_captured = 0.0

        primary_tracts = []
        secondary_tracts = []
        tertiary_tracts = []

        for t in sorted_tracts:
            tract_spend = t["captured_spend_millions"][store_id]
            cum_captured += tract_spend
            share = cum_captured / max(0.001, total_captured)
            tract_entry = {
                "tract_id": t["tract_id"],
                "name": t["tract_name"],
                "probability": t["huff_patronage_probabilities"][store_id],
                "distance_km": t["distances_km"][store_id],
                "captured_spend_millions": tract_spend,
            }

            if share <= 0.60 or len(primary_tracts) == 0:
                primary_tracts.append(tract_entry)
            elif share <= 0.85 or len(secondary_tracts) == 0:
                secondary_tracts.append(tract_entry)
            else:
                tertiary_tracts.append(tract_entry)

        def _calc_stats(tract_list):
            if not tract_list:
                return {"tract_count": 0, "avg_distance_km": 0.0, "total_spend_millions": 0.0}
            avg_dist = sum(x["distance_km"] for x in tract_list) / len(tract_list)
            tot_spend = sum(x["captured_spend_millions"] for x in tract_list)
            return {
                "tract_count": len(tract_list),
                "avg_distance_km": round(avg_dist, 2),
                "total_spend_millions": round(tot_spend, 2),
                "tracts": [x["name"] for x in tract_list],
            }

        return {
            "store_id": store_id,
            "store_name": self.STORE_NETWORK[store_id]["name"],
            "total_captured_revenue_millions": round(total_captured, 2),
            "primary_trade_area_60pct": _calc_stats(primary_tracts),
            "secondary_trade_area_80pct": _calc_stats(secondary_tracts),
            "tertiary_trade_area_95pct": _calc_stats(tertiary_tracts),
        }

    def analyze_store_cannibalization(
        self,
        new_store_name: str,
        new_store_lat: float,
        new_store_lon: float,
        new_store_sqft: float,
        new_store_attractiveness: float,
        beta: float = 1.75,
    ) -> Dict[str, Any]:
        """
        Simulates the introduction of a new candidate store into the metropolitan retail network.
        Quantifies new store gross revenue, cannibalization leakage from sister stores,
        net incremental revenue unlocked, and cannibalization ratio.
        """
        if new_store_sqft <= 0:
            raise ValueError("Square footage must be positive.")
        if new_store_attractiveness <= 0:
            raise ValueError("Attractiveness score must be positive.")

        # Baseline market share before expansion
        base_huff = self.compute_huff_market_shares(beta=beta)
        base_shares = base_huff["store_market_shares"]

        # Expanded network with candidate store
        new_store_id = f"STORE_NEW_{new_store_name.upper().replace(' ', '_')}"
        expanded_network = dict(self.STORE_NETWORK)
        expanded_network[new_store_id] = {
            "store_id": new_store_id,
            "name": new_store_name,
            "lat": new_store_lat,
            "lon": new_store_lon,
            "sqft": new_store_sqft,
            "attractiveness_score": new_store_attractiveness,
            "merchandise_depth_tier": "Candidate Expansion",
        }

        # Re-compute Huff probabilities with new store included
        expanded_probs = self.compute_huff_probabilities(expanded_network, beta=beta)

        post_captured = {s_id: 0.0 for s_id in expanded_network.keys()}
        for item in expanded_probs:
            for s_id, spend in item["captured_spend_millions"].items():
                post_captured[s_id] += spend

        new_store_gross_rev = post_captured[new_store_id]

        cannibalization_breakdown = {}
        total_cannibalized_spend = 0.0

        for s_id in self.STORE_NETWORK.keys():
            base_rev = base_shares[s_id]["captured_market_spend_millions"]
            post_rev = post_captured[s_id]
            delta = base_rev - post_rev
            leakage = max(0.0, delta)
            total_cannibalized_spend += leakage
            leakage_pct = (leakage / max(0.001, base_rev)) * 100.0

            cannibalization_breakdown[s_id] = {
                "store_name": self.STORE_NETWORK[s_id]["name"],
                "baseline_revenue_millions": round(base_rev, 2),
                "post_expansion_revenue_millions": round(post_rev, 2),
                "cannibalized_revenue_millions": round(leakage, 2),
                "cannibalization_rate_pct": round(leakage_pct, 2),
            }

        net_incremental_revenue = max(0.0, new_store_gross_rev - total_cannibalized_spend)
        cannibalization_ratio_pct = (total_cannibalized_spend / max(0.001, new_store_gross_rev)) * 100.0

        # Recommendation triage
        if cannibalization_ratio_pct > 50.0:
            viability = "High Risk / Excessive Sister Store Cannibalization"
            rec_action = "Reject or Relocate: Over 50% of revenue siphoned from existing locations."
        elif cannibalization_ratio_pct > 30.0:
            viability = "Moderate Viability / Acceptable Overlap"
            rec_action = "Proceed with Differentiation: Emphasize unique category assortment to minimize overlap."
        else:
            viability = "Highly Favorable / Greenfield Expansion"
            rec_action = "Strong Approve: Unlocks predominantly new incremental territory with minimal cannibalization."

        return {
            "candidate_store": {
                "store_id": new_store_id,
                "name": new_store_name,
                "lat": new_store_lat,
                "lon": new_store_lon,
                "sqft": new_store_sqft,
                "attractiveness_score": new_store_attractiveness,
            },
            "financial_impact_summary": {
                "new_store_gross_projected_millions": round(new_store_gross_rev, 2),
                "sister_stores_cannibalized_millions": round(total_cannibalized_spend, 2),
                "net_incremental_expansion_revenue_millions": round(net_incremental_revenue, 2),
                "cannibalization_ratio_pct": round(cannibalization_ratio_pct, 2),
                "expansion_viability_rating": viability,
                "strategic_recommendation": rec_action,
            },
            "pairwise_cannibalization_breakdown": cannibalization_breakdown,
        }

    def evaluate_expansion_candidates(
        self,
        candidates: Optional[List[Dict[str, Any]]] = None,
        beta: float = 1.75,
    ) -> Dict[str, Any]:
        """Evaluates and ranks multiple candidate retail expansion locations by net ROI and spatial efficiency."""
        default_candidates = [
            {"name": "Site 1 - Brooklyn Heights", "lat": 40.6960, "lon": -73.9933, "sqft": 30000, "attractiveness": 84.0},
            {"name": "Site 2 - Stamford Waterfront CT", "lat": 41.0534, "lon": -73.5387, "sqft": 35000, "attractiveness": 80.0},
            {"name": "Site 3 - Hoboken PATH Terminal NJ", "lat": 40.7350, "lon": -74.0290, "sqft": 26000, "attractiveness": 82.0},
        ]
        sites = candidates or default_candidates

        evaluated = []
        for site in sites:
            res = self.analyze_store_cannibalization(
                new_store_name=site["name"],
                new_store_lat=site["lat"],
                new_store_lon=site["lon"],
                new_store_sqft=site["sqft"],
                new_store_attractiveness=site["attractiveness"],
                beta=beta,
            )
            fin = res["financial_impact_summary"]
            evaluated.append({
                "candidate_name": site["name"],
                "gross_revenue_millions": fin["new_store_gross_projected_millions"],
                "cannibalized_millions": fin["sister_stores_cannibalized_millions"],
                "net_incremental_revenue_millions": fin["net_incremental_expansion_revenue_millions"],
                "cannibalization_ratio_pct": fin["cannibalization_ratio_pct"],
                "viability": fin["expansion_viability_rating"],
            })

        # Rank by net incremental revenue descending
        ranked = sorted(evaluated, key=lambda x: x["net_incremental_revenue_millions"], reverse=True)
        for rank, item in enumerate(ranked, 1):
            item["rank"] = rank

        return {
            "ranked_expansion_sites": ranked,
            "top_recommended_site": ranked[0]["candidate_name"] if ranked else None,
            "strategic_rationale": "Prioritizes candidate locations maximizing net incremental demand capture while suppressing sister store cannibalization drag.",
        }
