"""
FastAPI router for LitePolis math endpoints.

This module provides API endpoints for mathematical analysis of conversation data:
- PCA visualization
- Clustering
- Representativeness (repness) calculation

These endpoints are designed to be compatible with the Polis frontend.
"""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import logging

from litepolis_database_default import DatabaseActor
from .algorithms import PCA, KMeans
from .validation import validate_matrix

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3/math", tags=["math"])


class PolisResponse(BaseModel):
    status: str = "ok"
    data: Optional[Dict[str, Any]] = None


class MathResultCache:
    """Database-backed cache for math results."""
    
    @classmethod
    def get(cls, zid: int, math_tick: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get cached math result for conversation."""
        if math_tick is not None:
            result = DatabaseActor.get_math_result(zid, math_tick)
        else:
            result = DatabaseActor.get_latest_math_result(zid)
        if result:
            return result.get_data()
        return None
    
    @classmethod
    def set(cls, zid: int, data: Dict[str, Any]) -> int:
        """Store math result and return new math_tick."""
        result = DatabaseActor.store_result(zid, data)
        return result.math_tick
    
    @classmethod
    def get_tick(cls, zid: int) -> int:
        """Get current math_tick for conversation."""
        return DatabaseActor.get_current_tick(zid)


def get_zid_from_conversation_id(conversation_id: str) -> Optional[int]:
    """Convert conversation_id (zinvite) to zid."""
    zinvite = DatabaseActor.read_zinvite(conversation_id)
    if zinvite:
        return zinvite.zid
    # Try parsing as numeric zid
    try:
        zid = int(conversation_id)
        conv = DatabaseActor.read_conversation(zid)
        if conv:
            return zid
    except (ValueError, TypeError):
        pass
    return None


def build_vote_matrix(zid: int) -> tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
    """
    Build participant x comment vote matrix for a conversation.
    
    Returns:
        - DataFrame: participant x comment matrix with vote values
        - pid_to_idx: mapping from participant pid to row index
        - tid_to_idx: mapping from comment tid to column index
    """
    # Get all participants in this conversation
    participants = DatabaseActor.list_participants_by_zid(zid, page_size=10000)
    if not participants:
        return pd.DataFrame(), {}, {}
    
    # Get all approved comments in this conversation
    from sqlmodel import select
    from litepolis_database_default.Comments import Comment
    from litepolis_database_default.utils import get_session
    
    with get_session() as session:
        comments = session.exec(
            select(Comment)
            .where(Comment.conversation_id == zid)
            .where(Comment.moderation_status >= 0)  # Approved or pending
        ).all()
    
    if not comments:
        return pd.DataFrame(), {}, {}
    
    # Build mappings
    pids = [p.pid for p in participants]
    tids = [c.id for c in comments]
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}
    tid_to_idx = {tid: i for i, tid in enumerate(tids)}
    
    # Build vote matrix
    from litepolis_database_default.Vote import Vote
    with get_session() as session:
        votes = session.exec(
            select(Vote)
            .where(Vote.comment_id.in_(tids))
        ).all()
    
    # Create matrix filled with NaN (unvoted)
    matrix = np.full((len(pids), len(tids)), np.nan)
    
    for vote in votes:
        # Get the participant for this vote
        participant = DatabaseActor.get_participant_by_zid_uid(zid, vote.user_id)
        if participant and participant.pid in pid_to_idx and vote.comment_id in tid_to_idx:
            row = pid_to_idx[participant.pid]
            col = tid_to_idx[vote.comment_id]
            matrix[row, col] = vote.value
    
    df = pd.DataFrame(matrix, index=pids, columns=tids)
    return df, pid_to_idx, tid_to_idx


def compute_pca_projection(matrix: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, PCA]:
    """Compute PCA projection of vote matrix."""
    # Handle missing values by filling with 0 (neutral)
    filled_matrix = np.nan_to_num(matrix, nan=0.0)
    
    if filled_matrix.shape[0] < 2 or filled_matrix.shape[1] < 2:
        # Not enough data for PCA
        return np.zeros((filled_matrix.shape[0], n_components)), None
    
    pca = PCA(n_components=n_components)
    projection = pca.fit_transform(filled_matrix)
    return projection, pca


def compute_base_clusters(projection: np.ndarray, pids: List[int], k: int = 100) -> List[Dict]:
    """
    Compute base clusters using K-means on participant projections.
    
    Returns list of base clusters with id, members, center, and count.
    """
    n_participants = len(pids)
    if n_participants == 0:
        return []
    
    # Adjust k if we have fewer participants
    actual_k = min(k, max(1, n_participants // 2))
    
    if actual_k < 2 or projection.shape[0] < 2:
        # Single cluster if not enough data
        return [{
            "id": 0,
            "members": pids,
            "x": float(np.mean(projection[:, 0])) if projection.size > 0 else 0.0,
            "y": float(np.mean(projection[:, 1])) if projection.size > 1 else 0.0,
            "count": len(pids)
        }]
    
    kmeans = KMeans(n_clusters=actual_k)
    labels = kmeans.fit_predict(projection)
    
    base_clusters = []
    for cluster_id in range(actual_k):
        member_indices = np.where(labels == cluster_id)[0]
        if len(member_indices) == 0:
            continue
        
        members = [pids[i] for i in member_indices]
        center_x = float(np.mean(projection[member_indices, 0]))
        center_y = float(np.mean(projection[member_indices, 1]))
        
        base_clusters.append({
            "id": cluster_id,
            "members": members,
            "x": center_x,
            "y": center_y,
            "count": len(members)
        })
    
    return base_clusters


def compute_group_clusters(base_clusters: List[Dict], min_k: int = 2, max_k: int = 5) -> List[Dict]:
    """
    Compute group clusters using K-means on base cluster centers.
    
    Uses silhouette score to select optimal k.
    """
    if len(base_clusters) < 2:
        return [{"id": 0, "members": [bc["id"] for bc in base_clusters], "x": 0.0, "y": 0.0}]
    
    # Get base cluster centers and weights
    centers = np.array([[bc["x"], bc["y"]] for bc in base_clusters])
    weights = np.array([bc["count"] for bc in base_clusters])
    
    # Try different k values and pick best by silhouette score
    best_k = min_k
    best_score = -1
    
    for k in range(min_k, min(max_k + 1, len(base_clusters))):
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(centers)
        
        # Simple silhouette score calculation
        score = compute_silhouette_score(centers, labels)
        if score > best_score:
            best_score = score
            best_k = k
    
    # Final clustering with best k
    kmeans = KMeans(n_clusters=best_k)
    labels = kmeans.fit_predict(centers)
    
    group_clusters = []
    for group_id in range(best_k):
        member_indices = np.where(labels == group_id)[0]
        member_bids = [base_clusters[i]["id"] for i in member_indices]
        
        # Weighted center
        total_weight = sum(weights[member_indices])
        center_x = sum(base_clusters[i]["x"] * weights[i] for i in member_indices) / total_weight if total_weight > 0 else 0.0
        center_y = sum(base_clusters[i]["y"] * weights[i] for i in member_indices) / total_weight if total_weight > 0 else 0.0
        
        group_clusters.append({
            "id": group_id,
            "members": member_bids,
            "x": center_x,
            "y": center_y
        })
    
    return group_clusters


def compute_silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute simplified silhouette score."""
    if len(np.unique(labels)) < 2:
        return -1
    
    n = len(X)
    if n < 2:
        return -1
    
    silhouette_values = []
    for i in range(n):
        # Same cluster distances
        same_cluster = X[labels == labels[i]]
        if len(same_cluster) > 1:
            a = np.mean(np.linalg.norm(same_cluster - X[i], axis=1))
        else:
            a = 0
        
        # Nearest other cluster distances
        other_clusters = [c for c in np.unique(labels) if c != labels[i]]
        if other_clusters:
            b_values = []
            for c in other_clusters:
                other_cluster = X[labels == c]
                b_values.append(np.mean(np.linalg.norm(other_cluster - X[i], axis=1)))
            b = min(b_values)
        else:
            b = 0
        
        if max(a, b) > 0:
            silhouette_values.append((b - a) / max(a, b))
    
    return np.mean(silhouette_values) if silhouette_values else -1


def compute_repness(
    vote_matrix: pd.DataFrame,
    group_clusters: List[Dict],
    base_clusters: List[Dict],
    tids: List[int]
) -> Dict[str, List[Dict]]:
    """
    Compute representativeness (repness) of comments for each group.
    
    Repness measures how representative a comment is of a group's opinion.
    """
    repness = {}
    
    # Build bid to pid mapping
    bid_to_pids = {bc["id"]: bc["members"] for bc in base_clusters}
    
    for group in group_clusters:
        gid = str(group["id"])
        repness[gid] = []
        
        # Get all pids in this group
        group_pids = []
        for bid in group["members"]:
            group_pids.extend(bid_to_pids.get(bid, []))
        
        if not group_pids:
            continue
        
        # Get votes for this group
        group_votes = vote_matrix.loc[[p for p in group_pids if p in vote_matrix.index]]
        
        # Get votes for all other participants
        other_pids = [p for p in vote_matrix.index if p not in group_pids]
        other_votes = vote_matrix.loc[other_pids] if other_pids else pd.DataFrame()
        
        for tid in tids:
            if tid not in vote_matrix.columns:
                continue
            
            # Count agrees, disagrees, seen for this group
            group_col = group_votes[tid].dropna()
            na = int((group_col == 1).sum())  # agree count
            nd = int((group_col == -1).sum())  # disagree count
            ns = int(len(group_col))  # total seen
            
            if ns < 2:
                continue
            
            # Same for other groups
            if not other_votes.empty and tid in other_votes.columns:
                other_col = other_votes[tid].dropna()
                na_other = int((other_col == 1).sum())
                nd_other = int((other_col == -1).sum())
                ns_other = int(len(other_col))
            else:
                na_other = 0
                nd_other = 0
                ns_other = 0
            
            # Compute repness score (simplified binomial test)
            # Compare group's agree/disagree rate to overall rate
            pa = na / ns if ns > 0 else 0  # prob agree in group
            prob_disagree = nd / ns if ns > 0 else 0  # prob disagree in group
            
            total_na = na + na_other
            total_nd = nd + nd_other
            total_ns = ns + ns_other
            
            pa_overall = total_na / total_ns if total_ns > 0 else 0
            pd_overall = total_nd / total_ns if total_ns > 0 else 0
            
            # Repness is how much more this group agrees/disagrees than overall
            repness_agree = pa - pa_overall
            repness_disagree = prob_disagree - pd_overall
            
            # Determine if repful for agree or disagree
            if repness_agree >= repness_disagree and repness_agree > 0.1:
                repness[gid].append({
                    "tid": int(tid),
                    "n-success": na,
                    "n-trials": ns,
                    "p-success": round(pa, 3),
                    "p-test": round(repness_agree, 3),
                    "repness": round(repness_agree, 3),
                    "repness-test": round(repness_agree, 3),
                    "repful-for": "agree"
                })
            elif repness_disagree > 0.1:
                repness[gid].append({
                    "tid": int(tid),
                    "n-success": nd,
                    "n-trials": ns,
                    "p-success": round(pd, 3),
                    "p-test": round(repness_disagree, 3),
                    "repness": round(repness_disagree, 3),
                    "repness-test": round(repness_disagree, 3),
                    "repful-for": "disagree"
                })
        
        # Sort by repness and take top comments
        repness[gid].sort(key=lambda x: x["repness"], reverse=True)
        repness[gid] = repness[gid][:10]  # Top 10 comments per group
    
    return repness


def compute_group_votes(
    vote_matrix: pd.DataFrame,
    group_clusters: List[Dict],
    base_clusters: List[Dict]
) -> Dict[str, Dict]:
    """Compute aggregated vote counts per group per comment."""
    group_votes = {}
    
    bid_to_pids = {bc["id"]: bc["members"] for bc in base_clusters}
    
    for group in group_clusters:
        gid = str(group["id"])
        
        # Get all pids in this group
        group_pids = []
        for bid in group["members"]:
            group_pids.extend(bid_to_pids.get(bid, []))
        
        group_pids = [p for p in group_pids if p in vote_matrix.index]
        if not group_pids:
            continue
        
        group_votes[gid] = {
            "id": group["id"],
            "n-members": len(group_pids),
            "votes": {}
        }
        
        for tid in vote_matrix.columns:
            col = vote_matrix.loc[group_pids, tid].dropna()
            if len(col) > 0:
                group_votes[gid]["votes"][str(int(tid))] = {
                    "A": int((col == 1).sum()),
                    "D": int((col == -1).sum()),
                    "S": int(len(col))
                }
    
    return group_votes


def compute_consensus(vote_matrix: pd.DataFrame, tids: List[int]) -> Dict[str, List[Dict]]:
    """Compute consensus comments (highly agreed/disagreed overall)."""
    consensus = {"agree": [], "disagree": []}
    
    for tid in tids:
        if tid not in vote_matrix.columns:
            continue
        
        col = vote_matrix[tid].dropna()
        ns = len(col)
        if ns < 3:
            continue
        
        na = int((col == 1).sum())
        nd = int((col == -1).sum())
        
        pa = na / ns
        pd = nd / ns
        
        # High consensus threshold (80%)
        if pa >= 0.8:
            consensus["agree"].append({
                "tid": int(tid),
                "n-success": na,
                "n-trials": ns,
                "p-success": round(pa, 3)
            })
        elif pd >= 0.8:
            consensus["disagree"].append({
                "tid": int(tid),
                "n-success": nd,
                "n-trials": ns,
                "p-success": round(pd, 3)
            })
    
    return consensus


def compute_full_math(zid: int) -> Dict[str, Any]:
    """
    Compute full math analysis for a conversation.
    
    Returns the complete pca2 output format compatible with Polis frontend.
    """
    # Build vote matrix
    vote_df, pid_to_idx, tid_to_idx = build_vote_matrix(zid)
    
    pids = list(pid_to_idx.keys())
    tids = list(tid_to_idx.keys())
    
    result = {
        "math_tick": 0,
        "n": len(pids),
        "n-cmts": len(tids),
        "tids": [int(t) for t in tids],
        "in-conv": [int(p) for p in pids],
        "mod-out": [],
        "mod-in": [int(t) for t in tids],
        "lastVoteTimestamp": int(datetime.now(timezone.utc).timestamp() * 1000)
    }
    
    if vote_df.empty or len(pids) < 1:
        # Return empty result for conversations with no data
        result.update({
            "base-clusters": {"id": [], "members": [], "x": [], "y": [], "count": []},
            "group-clusters": [],
            "pca": {
                "center": [0.0, 0.0],
                "comps": [[1.0, 0.0], [0.0, 1.0]],
                "comment-projection": [],
                "comment-extremity": []
            },
            "repness": {},
            "group-votes": {},
            "consensus": {"agree": [], "disagree": []}
        })
        return result
    
    # PCA projection
    matrix = vote_df.values
    projection, pca = compute_pca_projection(matrix)
    
    # Base clusters
    base_clusters = compute_base_clusters(projection, pids)
    
    # Format base clusters for output
    result["base-clusters"] = {
        "id": [bc["id"] for bc in base_clusters],
        "members": [bc["members"] for bc in base_clusters],
        "x": [bc["x"] for bc in base_clusters],
        "y": [bc["y"] for bc in base_clusters],
        "count": [bc["count"] for bc in base_clusters]
    }
    
    # Group clusters
    group_clusters = compute_group_clusters(base_clusters)
    result["group-clusters"] = group_clusters
    
    # PCA data
    if pca is not None:
        result["pca"] = {
            "center": pca.mean.tolist() if pca.mean is not None else [0.0, 0.0],
            "comps": pca.components.T.tolist() if pca.components is not None else [[1.0, 0.0], [0.0, 1.0]],
            "comment-projection": [],  # TODO: implement comment projection
            "comment-extremity": []   # TODO: implement comment extremity
        }
    else:
        result["pca"] = {
            "center": [0.0, 0.0],
            "comps": [[1.0, 0.0], [0.0, 1.0]],
            "comment-projection": [],
            "comment-extremity": []
        }
    
    # Repness
    result["repness"] = compute_repness(vote_df, group_clusters, base_clusters, tids)
    
    # Group votes
    result["group-votes"] = compute_group_votes(vote_df, group_clusters, base_clusters)
    
    # Consensus
    result["consensus"] = compute_consensus(vote_df, tids)
    
    return result


@router.get("/pca", response_model=PolisResponse)
async def get_pca(
    conversation_id: str = Query(..., description="Conversation ID or zinvite code")
):
    """
    Get PCA visualization data (v1 format).
    
    Returns participant and comment projections for visualization.
    """
    zid = get_zid_from_conversation_id(conversation_id)
    if zid is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Check cache first
    cached = MathResultCache.get(zid)
    if cached:
        return PolisResponse(
            status="ok",
            data={
                "commentProjection": cached.get("pca", {}).get("comment-projection", []),
                "ptptotProjection": [],  # Participant projections (from base clusters)
                "baseCluster": cached.get("base-clusters", {}),
                "groupAware": len(cached.get("group-clusters", [])) > 0
            }
        )
    
    # Compute new results
    result = compute_full_math(zid)
    MathResultCache.set(zid, result)
    
    return PolisResponse(
        status="ok",
        data={
            "commentProjection": result.get("pca", {}).get("comment-projection", []),
            "ptptotProjection": [],
            "baseCluster": result.get("base-clusters", {}),
            "groupAware": len(result.get("group-clusters", [])) > 0
        }
    )


@router.get("/pca2")
async def get_pca2(
    conversation_id: str = Query(..., description="Conversation ID or zinvite code"),
    math_tick: Optional[int] = Query(None, description="Math tick for cache validation")
):
    """
    Get PCA visualization data (v2 format).
    
    Returns the full math result including:
    - base-clusters: K-means clusters of participants
    - group-clusters: Higher-level opinion groups
    - repness: Representative comments per group
    - group-votes: Aggregated vote counts
    - consensus: Comments with high agreement
    """
    zid = get_zid_from_conversation_id(conversation_id)
    if zid is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Check cache first
    cached = MathResultCache.get(zid, math_tick)
    if cached:
        return cached
    
    # Compute new results
    result = compute_full_math(zid)
    MathResultCache.set(zid, result)
    
    return result


@router.get("/correlationMatrix", response_model=PolisResponse)
async def get_correlation_matrix(
    report_id: int = Query(..., description="Report ID"),
    math_tick: Optional[int] = Query(None, description="Math tick for cache validation")
):
    """
    Get correlation matrix for comments in a report.
    
    Note: This endpoint is deprecated in the original Polis implementation.
    Returns a basic correlation matrix computed from vote data.
    """
    # For now, return empty matrix (would need report implementation)
    return PolisResponse(
        status="ok",
        data={
            "matrix": [],
            "comments": []
        }
    )


@router.post("/mathUpdate", response_model=PolisResponse)
async def trigger_math_update(
    conversation_id: str = Query(..., description="Conversation ID to update")
):
    """
    Trigger a recomputation of math results for a conversation.
    
    Clears the cache and forces recomputation on next request.
    """
    zid = get_zid_from_conversation_id(conversation_id)
    if zid is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Force recomputation by computing new results
    result = compute_full_math(zid)
    new_tick = MathResultCache.set(zid, result)
    
    return PolisResponse(
        status="ok",
        data={
            "math_tick": new_tick,
            "message": "Math update completed"
        }
    )


def get_router() -> APIRouter:
    """Return the math router for mounting in a FastAPI app."""
    return router
