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
        return series.dtype == "int32"
    return False


def check(df: pd.DataFrame, name: str, allow_extra: bool = False) -> pd.DataFrame:
    """Validate exact columns, dtypes, and nullability for a named schema."""
    schemas: dict[str, Schema] = {"OLIST_ORDER_FEATURES": OLIST_ORDER_FEATURES, "OLIST_LABELS": OLIST_LABELS}
    if name not in schemas:
        raise KeyError(name)
    schema = schemas[name]
    expected = set(schema)
    actual = set(df.columns)
    missing = expected - actual
    extra = actual - expected
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
    schemas: dict[str, Schema] = {"OLIST_ORDER_FEATURES": OLIST_ORDER_FEATURES, "OLIST_LABELS": OLIST_LABELS}
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
