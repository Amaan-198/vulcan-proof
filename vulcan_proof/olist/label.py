"""Leak-free Olist fulfillment-complaint labels."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import pandas as pd

from ..errors import InvariantError, SchemaError
from ..params import P, Params
from ..schemas import OLIST_LABELS, cast, require_columns


EXCLUDED_STATUSES = frozenset({"canceled", "unavailable", "created"})
LABEL_REASONS = frozenset({"late_low_score", "comment_match", "not_delivered", "none"})


def _review_first(reviews: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        reviews,
        {"order_id": object(), "review_score": object(), "review_comment_message": object(), "review_creation_date": object()},
        "reviews",
    )
    current = reviews.copy()
    current["review_creation_date"] = pd.to_datetime(current["review_creation_date"], errors="coerce")
    current = current.sort_values(
        ["order_id", "review_creation_date"], kind="mergesort", na_position="last"
    )
    return current.drop_duplicates("order_id", keep="first")


def build_labels(
    orders: pd.DataFrame,
    reviews: pd.DataFrame,
    params: Params = P,
) -> pd.DataFrame:
    """Build one exact label row per non-excluded Olist order."""
    require_columns(
        orders,
        {
            "order_id": object(),
            "order_status": object(),
            "order_purchase_timestamp": object(),
            "order_delivered_customer_date": object(),
            "order_estimated_delivery_date": object(),
        },
        "orders",
    )
    frame = orders.copy()
    frame["purchase_ts"] = pd.to_datetime(frame["order_purchase_timestamp"], errors="raise")
    frame["delivered_ts"] = pd.to_datetime(frame["order_delivered_customer_date"], errors="coerce")
    frame["estimated_ts"] = pd.to_datetime(frame["order_estimated_delivery_date"], errors="coerce")
    excluded = frame["order_status"].isin(EXCLUDED_STATUSES)
    frame = frame.loc[~excluded].copy()
    first_reviews = _review_first(reviews)
    frame = frame.merge(
        first_reviews[["order_id", "review_score", "review_comment_message"]],
        on="order_id",
        how="left",
        validate="one_to_one",
    )
    score = pd.to_numeric(frame["review_score"], errors="coerce")
    low_score = score <= int(params["olist.label.low_score_max"])
    delivered_status = frame["order_status"].eq("delivered")
    late = delivered_status & frame["delivered_ts"].notna() & frame["estimated_ts"].notna() & (frame["delivered_ts"] > frame["estimated_ts"])
    comment = frame["review_comment_message"].fillna("").astype("string")
    complaint = re.compile(str(params["olist.label.complaint_regex"]), flags=re.IGNORECASE)
    comment_match = delivered_status & comment.str.contains(complaint, na=False)
    dataset_end = pd.Timestamp(params["olist.label.dataset_end"])
    not_delivered_status = frame["order_status"].isin(
        frozenset({"shipped", "approved", "processing", "invoiced"})
    )
    not_delivered = not_delivered_status & (
        frame["purchase_ts"] + pd.to_timedelta(int(params["olist.label.not_delivered_days"]), unit="D")
        < dataset_end
    )
    reason = pd.Series("none", index=frame.index, dtype="string")
    reason.loc[late & low_score] = "late_low_score"
    reason.loc[~(late & low_score) & comment_match & low_score] = "comment_match"
    reason.loc[~(late & low_score) & ~(comment_match & low_score) & not_delivered] = "not_delivered"
    label = reason.ne("none").astype("int8")
    result = pd.DataFrame(
        {
            "order_id": frame["order_id"].astype("string"),
            "purchase_ts": frame["purchase_ts"],
            "label": label,
            "label_reason": reason,
        }
    )
    result = cast(result, "OLIST_LABELS")
    if not set(result["label_reason"].astype(str)).issubset(LABEL_REASONS):
        raise InvariantError("unknown Olist label reason")
    return result


def label_statistics(labels: pd.DataFrame, excluded_count: int = 0) -> dict[str, Any]:
    """Return finite label-rate and reason-mix values for reporting."""
    if list(labels.columns) != list(OLIST_LABELS):
        from ..schemas import check

        check(labels, "OLIST_LABELS")
    rate = float(labels["label"].mean())
    reasons = labels["label_reason"].astype(str).value_counts().to_dict()
    return {
        "label_rate": rate,
        "reason_mix": {str(key): int(value) for key, value in sorted(reasons.items())},
        "excluded_count": int(excluded_count),
    }
