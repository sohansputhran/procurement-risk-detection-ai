# src/procurement_risk_detection_ai/graph/graph_utils.py
"""
Supplier graph prototype:
- Build a bipartite graph (buyers ↔ suppliers) from OCDS curated parquet files.
- Compute supplier metrics: degree, betweenness (optional), distance_to_sanctioned.
- Write metrics to data/graph/metrics.parquet
- (Optional) Save an ego PNG around a given supplier.

Usage:
  python -m procurement_risk_detection_ai.graph.graph_utils \
    --awards data/curated/ocds/awards.parquet \
    --tenders data/curated/ocds/tenders.parquet \
    --sanctions data/curated/worldbank/ineligible.parquet \
    --out-dir data \
    --ego-supplier-id S1
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt  # only needed if --ego-supplier-id is used

    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

# Try to import multi-source shortest path length from algorithms submodule
try:
    from networkx.algorithms.shortest_paths.unweighted import (
        multi_source_shortest_path_length as _ms_spl,
    )
except Exception:
    _ms_spl = None


def _norm_name(x: object) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).casefold()
    s = re.sub(r"[\W_]+", " ", s, flags=re.UNICODE)  # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _supplier_key(supplier_id: Optional[str], supplier_name: Optional[str]) -> str:
    # Prefer supplier_id; fall back to hashed normalized name
    if supplier_id and str(supplier_id).strip():
        return str(supplier_id)
    nm = _norm_name(supplier_name) or "unknown"
    return "sup:" + str(abs(hash(nm)))  # stable within a run; good enough for prototype


def _buyer_key(buyer_id: Optional[str], buyer_name: Optional[str]) -> str:
    # Prefer buyer_id; else normalized name
    if buyer_id and str(buyer_id).strip():
        return "buyer:" + str(buyer_id)
    nm = _norm_name(buyer_name) or "unknown-buyer"
    return "buyer:" + nm


def build_bipartite_graph(
    tenders: pd.DataFrame, awards: pd.DataFrame
) -> Tuple[nx.Graph, Dict[str, Dict]]:
    """
    Returns:
      G: bipartite graph (buyers ↔ suppliers)
      meta: dict with 'buyer_nodes' and 'supplier_nodes' sets
    """
    # Ensure required columns exist
    for col in ["tender_id"]:
        if col not in tenders.columns:
            tenders[col] = None
    for col in ["tender_id", "supplier_id", "supplier_name", "award_id"]:
        if col not in awards.columns:
            awards[col] = None

    # Join awards→tenders for buyer info
    buyers_df = (
        tenders[["tender_id", "buyer_id", "buyer_name"]].copy()
        if "buyer_id" in tenders.columns
        else tenders[["tender_id", "buyer_name"]].copy()
    )
    buyers_df["buyer_id"] = buyers_df.get("buyer_id")
    buyers_df["buyer_name"] = buyers_df.get("buyer_name")
    df = awards.merge(buyers_df, on="tender_id", how="left")

    G = nx.Graph()
    buyer_nodes = set()
    supplier_nodes = set()

    # Aggregate multiple awards between same buyer-supplier into edge weight
    df["amount"] = pd.to_numeric(df.get("amount", np.nan), errors="coerce")
    agg = (
        df.groupby(
            ["buyer_id", "buyer_name", "supplier_id", "supplier_name"], dropna=False
        )
        .agg(awards_count=("award_id", "count"), total_amount=("amount", "sum"))
        .reset_index()
    )

    for _, r in agg.iterrows():
        bkey = _buyer_key(r.get("buyer_id"), r.get("buyer_name"))
        skey = _supplier_key(r.get("supplier_id"), r.get("supplier_name"))
        G.add_node(
            bkey,
            bipartite="buyer",
            name=r.get("buyer_name"),
            buyer_id=r.get("buyer_id"),
        )
        G.add_node(
            skey,
            bipartite="supplier",
            name=r.get("supplier_name"),
            supplier_id=r.get("supplier_id"),
        )
        buyer_nodes.add(bkey)
        supplier_nodes.add(skey)
        # Edge with weights
        G.add_edge(
            bkey,
            skey,
            awards_count=int(r["awards_count"]),
            total_amount=(
                float(r["total_amount"]) if pd.notna(r["total_amount"]) else 0.0
            ),
        )

    meta = {"buyer_nodes": buyer_nodes, "supplier_nodes": supplier_nodes}
    return G, meta


def _match_sanctions(suppliers: pd.DataFrame, sanctions: pd.DataFrame) -> pd.Series:
    """
    Return a boolean Series aligned to suppliers['supplier_name'] indicating sanctioned status by simple normalized-name match.
    """
    # Sanctions list may have 'normalized_name' (our pipeline) or only 'name'
    sanc_names = (
        sanctions.get("normalized_name")
        if "normalized_name" in sanctions.columns
        else sanctions.get("name")
    )
    sanc_set = {_norm_name(x) for x in sanc_names.dropna().astype(str).tolist()}
    supl_norm = suppliers["supplier_name"].map(_norm_name)
    return supl_norm.isin(sanc_set)


def compute_metrics(
    tenders_path: str,
    awards_path: str,
    sanctions_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    ego_supplier_id: Optional[str] = None,
    compute_betweenness: bool = True,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Build graph and compute supplier metrics.
    Returns (metrics_df, ego_png_path_or_None).
    """
    tenders = pd.read_parquet(tenders_path)
    awards = pd.read_parquet(awards_path)

    # Build graph
    G, meta = build_bipartite_graph(tenders, awards)

    # Prepare supplier reference (id/name) mapping
    suppliers_ref = []
    for node in meta["supplier_nodes"]:
        data = G.nodes[node]
        suppliers_ref.append(
            {
                "node": node,
                "supplier_id": data.get("supplier_id"),
                "supplier_name": data.get("name"),
            }
        )
    suppliers_ref = pd.DataFrame(suppliers_ref)

    # Simple metrics on suppliers
    deg = {n: G.degree(n) for n in meta["supplier_nodes"]}
    deg_df = pd.DataFrame({"node": list(deg.keys()), "degree": list(deg.values())})

    btw_df = pd.DataFrame({"node": [], "betweenness": []})
    if compute_betweenness:
        # Betweenness can be heavy; for prototype graphs it's fine.
        btw = nx.betweenness_centrality(G, normalized=True)
        btw_df = pd.DataFrame(
            {"node": list(btw.keys()), "betweenness": list(btw.values())}
        )
        btw_df = btw_df[btw_df["node"].isin(meta["supplier_nodes"])]

    metrics = suppliers_ref.merge(deg_df, on="node", how="left").merge(
        btw_df, on="node", how="left"
    )
    metrics["betweenness"] = metrics["betweenness"].fillna(0.0)

    # Distance to sanctioned suppliers
    metrics["distance_to_sanctioned"] = np.nan
    if sanctions_path and os.path.exists(sanctions_path):
        sanctions = pd.read_parquet(sanctions_path)
        suppliers = suppliers_ref.copy()
        suppliers["is_sanctioned"] = _match_sanctions(suppliers, sanctions)
        sanctioned_nodes = set(
            suppliers.loc[suppliers["is_sanctioned"], "node"].tolist()
        )

        if sanctioned_nodes:
            # Multi-source shortest paths (prefer fast impl if available)
            if _ms_spl is not None:
                lengths = _ms_spl(G, sources=sanctioned_nodes)
            else:
                # Fallback: min over single-source BFS from each sanctioned node
                lengths = {}
                for s in sanctioned_nodes:
                    for n, d in nx.single_source_shortest_path_length(G, s).items():
                        if (n not in lengths) or (d < lengths[n]):
                            lengths[n] = d
            # Assign distance for supplier nodes only
            metrics["distance_to_sanctioned"] = metrics["node"].map(
                lambda n: lengths.get(n, np.nan)
            )
        else:
            metrics["distance_to_sanctioned"] = np.nan

    # Save outputs
    out_png = None
    if out_dir:
        os.makedirs(os.path.join(out_dir, "graph"), exist_ok=True)
        out_parquet = os.path.join(out_dir, "graph", "metrics.parquet")
        metrics_out = metrics.drop(columns=["node"])
        metrics_out.to_parquet(out_parquet, index=False)

        # Ego graph (optional)
        if ego_supplier_id and HAVE_PLT:
            # find node by supplier_id
            row = metrics.loc[
                metrics["supplier_id"].astype(str) == str(ego_supplier_id)
            ]
            if not row.empty:
                node = row.iloc[0]["node"]
                H = nx.ego_graph(G, node, radius=2)
                plt.figure(figsize=(6, 6))
                pos = nx.spring_layout(H, seed=42)
                # Color buyers vs suppliers
                colors = [
                    (
                        "#1f77b4"
                        if H.nodes[n].get("bipartite") == "supplier"
                        else "#2ca02c"
                    )
                    for n in H.nodes
                ]
                nx.draw(
                    H,
                    pos,
                    node_color=colors,
                    with_labels=False,
                    node_size=120,
                    edge_color="#cccccc",
                )
                plt.title(f"Ego graph for supplier_id={ego_supplier_id}")
                out_png = os.path.join(
                    out_dir,
                    "graph",
                    f"ego_{re.sub(r'[^A-Za-z0-9_-]', '_', str(ego_supplier_id))}.png",
                )
                plt.savefig(out_png, dpi=150, bbox_inches="tight")
                plt.close()

    return metrics, out_png


def main():
    ap = argparse.ArgumentParser(
        description="Build buyer↔supplier graph metrics from OCDS curated parquet."
    )
    ap.add_argument("--tenders", type=str, default="data/curated/ocds/tenders.parquet")
    ap.add_argument("--awards", type=str, default="data/curated/ocds/awards.parquet")
    ap.add_argument(
        "--sanctions", type=str, default="data/curated/worldbank/ineligible.parquet"
    )
    ap.add_argument("--out-dir", type=str, default="data")
    ap.add_argument(
        "--ego-supplier-id",
        type=str,
        default=None,
        help="Optional supplier_id to render an ego PNG (radius=2)",
    )
    ap.add_argument(
        "--no-betweenness",
        action="store_true",
        help="Skip betweenness (faster on large graphs)",
    )
    args = ap.parse_args()

    metrics, out_png = compute_metrics(
        tenders_path=args.tenders,
        awards_path=args.awards,
        sanctions_path=args.sanctions if args.sanctions else None,
        out_dir=args.out_dir,
        ego_supplier_id=args.ego_supplier_id,
        compute_betweenness=not args.no_betweenness,
    )

    print(
        f"Wrote: {os.path.join(args.out_dir, 'graph', 'metrics.parquet')} (rows={len(metrics)})"
    )
    if out_png:
        print(f"Wrote ego PNG: {out_png}")


if __name__ == "__main__":
    main()
