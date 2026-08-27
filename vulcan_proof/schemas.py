"""Named DataFrame schemas and strict validation."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import pandas as pd
from pandas.api import types as ptypes

from .errors import SchemaError
from .params import P


Schema = OrderedDict[str, str]


def _olist_feature_schema() -> Schema:
    dtypes = {
        "price_total": "float32",
        "freight_total": "float32",
        "n_items": "int16",
        "n_sellers": "int16",
        "product_category_en": "category",
        "product_weight_g": "float32",
        "product_volume_cm3": "float32",
        "product_photos_qty": "int16",
        "product_description_length": "int32",
        "payment_type": "category",
        "payment_installments": "int16",
        "customer_state": "category",
        "seller_state": "category",
        "same_state": "int8",
        "purchase_month": "int8",
        "purchase_dow": "int8",
        "purchase_hour": "int8",
        "seller_prior_orders": "int32",
        "seller_prior_complaint_rate": "float32",
        "customer_prior_orders": "int32",
    }
    return OrderedDict((column, dtypes[column]) for column in P["olist.features.permitted"])


OLIST_ORDER_FEATURES = _olist_feature_schema()
OLIST_LABELS: Schema = OrderedDict(
    [
        ("order_id", "string"),
        ("purchase_ts", "datetime64[ns]"),
        ("label", "int8"),
        ("label_reason", "string"),
    ]
)


def _sim_observed_schema() -> Schema:
    """Build the Phase-1 observed-order schema from the parameter allowlist."""
    dtypes: dict[str, str] = {
        "order_id": "string",
        "merchant_id": "string",
        "customer_id": "string",
        "category": "category",
        "order_value": "float32",
        "payment_method": "category",
        "network": "category",
        "issuer_family": "category",
        "new_customer": "int8",
        "address_mismatch": "int8",
        "prior_disputes": "int16",
        "account_age_days": "float32",
        "cart_items": "int16",
        "hour_of_day": "int8",
        "month": "int8",
        "merchant_order_count": "int32",
        "merchant_dispute_rate_hist": "float32:nullable",
        "merchant_contest_rate_hist": "float32:nullable",
        "merchant_compliance_hist": "float32:nullable",
        "verified_contact_available": "int8",
        "ack_optin": "int8",
        "eligible_tier": "category",
        "decision_date": "int32",
        "split": "category",
        "censored": "int8",
        "order_day": "int32",
    }
    return OrderedDict((column, dtypes[column]) for column in P["features.permitted"] + ["split", "censored", "order_day"])


ORDER_OBSERVED = _sim_observed_schema()
ORDER_HIDDEN: Schema = OrderedDict(
    [
        ("order_id", "string"),
        ("hidden_truth", "category"),
        ("hidden_z_risk", "float32"),
        ("hidden_z_type", "float32"),
        ("hidden_risk_mult", "float32"),
        ("hidden_quality", "float32"),
        ("hidden_carrier_reliability", "float32"),
        ("hidden_archetype", "category"),
        ("hidden_compliance", "float32"),
        ("hidden_contest_base", "float32"),
        ("hidden_requested_bitmask", "uint16"),
        ("hidden_random_stratum", "int8"),
        ("hidden_dispute_potential", "int8"),
        ("hidden_dispute_type", "category:nullable"),
        ("hidden_dispute_open_day", "int32:nullable"),
        ("hidden_resolution_day", "int32:nullable"),
        ("hidden_u_response", "float32"),
    ]
)


PLAN: Schema = OrderedDict(
    [
        ("order_id", "string"),
        ("requested_bitmask", "uint16"),
    ]
)


OUTCOME: Schema = OrderedDict(
    [
        ("order_id", "string"),
        ("arm_id", "string"),
        ("requested_bitmask", "uint16"),
        ("complied", "int8"),
        ("materialised_bitmask", "uint16"),
        ("wrong_recipient", "int8"),
        ("cash_cost", "float32"),
        ("time_cost", "float32"),
        ("ack_sent", "int8"),
        ("response", "category"),
        ("prevented", "int8"),
        ("prevention_mode", "category:nullable"),
        ("dispute_opened", "int8"),
        ("dispute_type", "category:nullable"),
        ("contested", "int8"),
        ("won", "int8"),
        ("value", "float32"),
        ("net", "float32"),
        ("censored", "int8"),
        ("split", "category"),
        ("claim_class", "category"),
    ]
)


def _is_nullable(dtype: str) -> bool:
    return dtype.endswith(":nullable")


def _base_dtype(dtype: str) -> str:
    return dtype.removesuffix(":nullable")


def _dtype_matches(series: pd.Series, expected: str) -> bool:
    if expected == "category":
        return isinstance(series.dtype, pd.CategoricalDtype)
    if expected == "string":
        return ptypes.is_string_dtype(series.dtype) and not isinstance(series.dtype, pd.CategoricalDtype)
    if expected == "datetime64[ns]":
        return ptypes.is_datetime64_ns_dtype(series.dtype)
    if expected == "float32":
        return ptypes.is_float_dtype(series.dtype) and series.dtype == "float32"
    if expected == "int8":
        return series.dtype == "int8"
    if expected == "int16":
        return series.dtype == "int16"
    if expected == "int32":
        return str(series.dtype) in {"int32", "Int32"}
    if expected == "uint16":
        return str(series.dtype) == "uint16"
    return False


def check(df: pd.DataFrame, name: str, allow_extra: bool = False) -> pd.DataFrame:
    """Validate exact columns, dtypes, and nullability for a named schema."""
    schemas: dict[str, Schema] = {
        "OLIST_ORDER_FEATURES": OLIST_ORDER_FEATURES,
        "OLIST_LABELS": OLIST_LABELS,
        "ORDER_OBSERVED": ORDER_OBSERVED,
        "ORDER_HIDDEN": ORDER_HIDDEN,
        "PLAN": PLAN,
        "OUTCOME": OUTCOME,
    }
    if name not in schemas:
        raise KeyError(name)
    schema = schemas[name]
    expected = set(schema)
    actual = set(df.columns)
    missing = expected - actual
    extra = actual - expected
    if tuple(df.columns) != tuple(schema) and not allow_extra:
        raise SchemaError(f"{name} columns are not in declared order")
    if missing:
        raise SchemaError(f"{name} is missing columns: {sorted(missing)}")
    if extra and not allow_extra:
        raise SchemaError(f"{name} has extra columns: {sorted(extra)}")
    for column, declared in schema.items():
        nullable = _is_nullable(declared)
        dtype = _base_dtype(declared)
        if not _dtype_matches(df[column], dtype):
            raise SchemaError(
                f"{name}.{column} has dtype {df[column].dtype}, expected {declared}"
            )
        if not nullable and bool(df[column].isna().any()):
            raise SchemaError(f"{name}.{column} contains nulls")
    return df


def cast(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Cast a frame to a named schema and validate it."""
    schemas: dict[str, Schema] = {
        "OLIST_ORDER_FEATURES": OLIST_ORDER_FEATURES,
        "OLIST_LABELS": OLIST_LABELS,
        "ORDER_OBSERVED": ORDER_OBSERVED,
        "ORDER_HIDDEN": ORDER_HIDDEN,
        "PLAN": PLAN,
        "OUTCOME": OUTCOME,
    }
    if name not in schemas:
        raise KeyError(name)
    schema = schemas[name]
    result = df.copy()
    for column, declared in schema.items():
        dtype = _base_dtype(declared)
        if dtype == "category":
            result[column] = result[column].astype("category")
        elif dtype == "string":
            result[column] = result[column].astype("string")
        else:
            result[column] = result[column].astype(dtype)
    result = result.loc[:, list(schema)]
    return check(result, name)


def require_columns(df: pd.DataFrame, columns: Mapping[str, Any], context: str) -> None:
    """Raise a schema error when a raw input is missing a required column."""
    missing = set(columns) - set(df.columns)
    if missing:
        raise SchemaError(f"{context} is missing columns: {sorted(missing)}")
