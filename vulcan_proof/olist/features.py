"""Purchase-time Olist features with strictly historical seller statistics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..errors import InvariantError
from ..params import P, Params
from ..schemas import OLIST_ORDER_FEATURES, check, require_columns


TABLE_NAMES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "translation": "product_category_name_translation.csv",
}


def _raw(tables: Mapping[str, pd.DataFrame], name: str) -> pd.DataFrame:
    filename = TABLE_NAMES[name]
    if filename not in tables:
        raise KeyError(filename)
    return tables[filename]


def _unique_order_item(items: pd.DataFrame) -> pd.DataFrame:
    current = items.copy()
    current["price"] = pd.to_numeric(current["price"], errors="raise")
    current["order_item_id"] = pd.to_numeric(current["order_item_id"], errors="raise")
    return current.sort_values(
        ["order_id", "price", "order_item_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).drop_duplicates("order_id", keep="first")


def _history_features(
    labels: pd.DataFrame,
    items: pd.DataFrame,
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    params: Params,
) -> pd.DataFrame:
    """Return order-keyed seller and customer history features."""
    label_rows = labels[["order_id", "purchase_ts", "label"]].copy()
    label_rows["order_id"] = label_rows["order_id"].astype("string")
    orders_key = orders[["order_id", "customer_id"]].copy()
    orders_key["order_id"] = orders_key["order_id"].astype("string")
    customers_key = customers[["customer_id", "customer_unique_id"]].copy()
    order_customer = orders_key.merge(customers_key, on="customer_id", how="left", validate="many_to_one")
    label_rows = label_rows.merge(order_customer, on="order_id", how="left", validate="one_to_one")
    if label_rows["customer_unique_id"].isna().any():
        raise InvariantError("customer_unique_id is missing for an eligible Olist order")

    history_items = items[["order_id", "seller_id"]].drop_duplicates().copy()
    history_items["order_id"] = history_items["order_id"].astype("string")
    seller_history = history_items.merge(
        label_rows[["order_id", "purchase_ts", "label"]],
        on="order_id",
        how="inner",
        validate="many_to_one",
    ).sort_values(["seller_id", "purchase_ts", "order_id"], kind="mergesort")
    train_end = pd.Timestamp(params["olist.split.train_end"]) + pd.Timedelta(days=1)
    train_rate = float(label_rows.loc[label_rows["purchase_ts"] < train_end, "label"].mean())
    shrinkage = float(params["olist.prior_shrinkage_n"])
    maturity = int(params["olist.split.maturity_days"])
    current = label_rows[["order_id", "purchase_ts", "customer_unique_id"]].copy()
    highest = _unique_order_item(items)[["order_id", "seller_id"]]
    current = current.merge(highest, on="order_id", how="left", validate="one_to_one")
    current["cutoff"] = current["purchase_ts"] - pd.to_timedelta(maturity, unit="D")
    seller_groups: dict[Any, tuple[np.ndarray, np.ndarray]] = {}
    for seller_id, group in seller_history.groupby("seller_id", sort=False):
        seller_groups[seller_id] = (
            group["purchase_ts"].to_numpy(dtype="datetime64[ns]"),
            group["label"].to_numpy(dtype="int64"),
        )
    seller_counts: list[int] = []
    seller_complaints: list[int] = []
    for seller_id, cutoff in zip(current["seller_id"], current["cutoff"], strict=True):
        dates, outcomes = seller_groups.get(
            seller_id,
            (np.array([], dtype="datetime64[ns]"), np.array([], dtype="int64")),
        )
        count = int(np.searchsorted(dates, np.datetime64(cutoff), side="left"))
        seller_counts.append(count)
        seller_complaints.append(int(outcomes[:count].sum()))
    current["seller_prior_orders"] = seller_counts
    current["seller_prior_complaint_rate"] = (
        np.asarray(seller_complaints, dtype="float64") + shrinkage * train_rate
    ) / (np.asarray(seller_counts, dtype="float64") + shrinkage)

    customer_history = label_rows.sort_values(
        ["customer_unique_id", "purchase_ts", "order_id"], kind="mergesort"
    )
    customer_groups: dict[Any, np.ndarray] = {
        customer_id: group["purchase_ts"].to_numpy(dtype="datetime64[ns]")
        for customer_id, group in customer_history.groupby("customer_unique_id", sort=False)
    }
    customer_counts: list[int] = []
    for customer_id, purchase_ts in zip(current["customer_unique_id"], current["purchase_ts"], strict=True):
        dates = customer_groups[customer_id]
        customer_counts.append(int(np.searchsorted(dates, np.datetime64(purchase_ts), side="left")))
    current["customer_prior_orders"] = customer_counts
    return current.set_index("order_id")[["seller_prior_orders", "seller_prior_complaint_rate", "customer_prior_orders"]]


def build_features(
    tables: Mapping[str, pd.DataFrame],
    labels: pd.DataFrame,
    params: Params = P,
) -> pd.DataFrame:
    """Build exactly the permitted Olist purchase-time feature columns."""
    from ..schemas import check as schema_check

    schema_check(labels, "OLIST_LABELS")
    orders = _raw(tables, "orders")
    items = _raw(tables, "items")
    payments = _raw(tables, "payments")
    products = _raw(tables, "products")
    customers = _raw(tables, "customers")
    sellers = _raw(tables, "sellers")
    translation = _raw(tables, "translation")
    # The public Olist CSV preserves the source typo ``lenght``; the phase
    # contract uses the canonical feature name ``length``.
    if "product_description_length" not in products.columns and "product_description_lenght" in products.columns:
        products = products.rename(columns={"product_description_lenght": "product_description_length"})
    require_columns(
        orders,
        {"order_id": object(), "customer_id": object(), "order_purchase_timestamp": object()},
        "orders",
    )
    require_columns(
        items,
        {"order_id": object(), "order_item_id": object(), "product_id": object(), "seller_id": object(), "price": object(), "freight_value": object()},
        "items",
    )
    require_columns(payments, {"order_id": object(), "payment_value": object(), "payment_type": object(), "payment_installments": object(), "payment_sequential": object()}, "payments")
    require_columns(products, {"product_id": object(), "product_category_name": object(), "product_weight_g": object(), "product_length_cm": object(), "product_height_cm": object(), "product_width_cm": object(), "product_photos_qty": object(), "product_description_length": object()}, "products")
    require_columns(customers, {"customer_id": object(), "customer_unique_id": object(), "customer_state": object()}, "customers")
    require_columns(sellers, {"seller_id": object(), "seller_state": object()}, "sellers")
    require_columns(translation, {"product_category_name": object(), "product_category_name_english": object()}, "translation")

    order_ids = labels["order_id"].astype("string")
    order_info = orders[["order_id", "customer_id", "order_purchase_timestamp"]].copy()
    order_info["order_id"] = order_info["order_id"].astype("string")
    order_info["purchase_ts"] = pd.to_datetime(order_info.pop("order_purchase_timestamp"), errors="raise")
    order_info = order_info.set_index("order_id").reindex(order_ids).reset_index()
    if order_info["customer_id"].isna().any():
        raise InvariantError("an eligible label has no matching order")
    item_current = items.copy()
    item_current["order_id"] = item_current["order_id"].astype("string")
    item_current["price"] = pd.to_numeric(item_current["price"], errors="raise")
    item_current["freight_value"] = pd.to_numeric(item_current["freight_value"], errors="raise")
    selected_items = _unique_order_item(item_current)
    totals = item_current.groupby("order_id", sort=False).agg(
        price_total=("price", "sum"),
        freight_total=("freight_value", "sum"),
        n_items=("order_item_id", "count"),
        n_sellers=("seller_id", "nunique"),
    )
    totals.index = totals.index.astype("string")
    result = order_info.set_index("order_id")[["customer_id", "purchase_ts"]].join(totals, how="left")
    result["price_total"] = result["price_total"].fillna(0.0)
    result["freight_total"] = result["freight_total"].fillna(0.0)
    result["n_items"] = result["n_items"].fillna(0)
    result["n_sellers"] = result["n_sellers"].fillna(0)

    product_info = products.merge(translation, on="product_category_name", how="left", validate="many_to_one")
    item_enriched = selected_items.merge(product_info, on="product_id", how="left", validate="many_to_one")
    item_enriched = item_enriched.merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left", validate="many_to_one")
    item_enriched["product_category_name_english"] = item_enriched["product_category_name_english"].fillna("unknown")
    item_enriched["seller_state"] = item_enriched["seller_state"].fillna("unknown")
    item_enriched = item_enriched.set_index(item_enriched["order_id"].astype("string"))
    result["product_category_en"] = item_enriched.reindex(result.index)["product_category_name_english"].fillna("unknown")
    result["product_weight_g"] = pd.to_numeric(item_enriched.reindex(result.index)["product_weight_g"], errors="coerce").fillna(0.0)
    length = pd.to_numeric(item_enriched.reindex(result.index)["product_length_cm"], errors="coerce").fillna(0.0)
    height = pd.to_numeric(item_enriched.reindex(result.index)["product_height_cm"], errors="coerce").fillna(0.0)
    width = pd.to_numeric(item_enriched.reindex(result.index)["product_width_cm"], errors="coerce").fillna(0.0)
    result["product_volume_cm3"] = length * height * width
    result["product_photos_qty"] = pd.to_numeric(item_enriched.reindex(result.index)["product_photos_qty"], errors="coerce").fillna(0)
    result["product_description_length"] = pd.to_numeric(item_enriched.reindex(result.index)["product_description_length"], errors="coerce").fillna(0)
    result["seller_state"] = item_enriched.reindex(result.index)["seller_state"].fillna("unknown")

    payments_current = payments.copy()
    payments_current["order_id"] = payments_current["order_id"].astype("string")
    payments_current["payment_value"] = pd.to_numeric(payments_current["payment_value"], errors="raise")
    payments_current["payment_sequential"] = pd.to_numeric(payments_current["payment_sequential"], errors="raise")
    largest_payment = payments_current.sort_values(
        ["order_id", "payment_value", "payment_sequential"], ascending=[True, False, True], kind="mergesort"
    ).drop_duplicates("order_id", keep="first").set_index("order_id")
    result["payment_type"] = largest_payment.reindex(result.index)["payment_type"].fillna("unknown")
    result["payment_installments"] = pd.to_numeric(largest_payment.reindex(result.index)["payment_installments"], errors="coerce").fillna(0)

    customer_states = customers.set_index(customers["customer_id"].astype("string"))["customer_state"]
    result["customer_state"] = customer_states.reindex(result["customer_id"].astype("string")).fillna("unknown").to_numpy()
    result["same_state"] = (result["customer_state"].astype("string") == result["seller_state"].astype("string")).astype("int8")
    result["purchase_month"] = result["purchase_ts"].dt.month
    result["purchase_dow"] = result["purchase_ts"].dt.dayofweek
    result["purchase_hour"] = result["purchase_ts"].dt.hour
    history = _history_features(labels, items, orders, customers, params).reindex(result.index)
    result = result.join(history)
    result = result.drop(columns=["customer_id", "purchase_ts"])
    result["product_category_en"] = result["product_category_en"].astype("category")
    result["payment_type"] = result["payment_type"].astype("category")
    result["customer_state"] = result["customer_state"].astype("category")
    result["seller_state"] = result["seller_state"].astype("category")
    result = result.reset_index(drop=True)
    result = result.loc[:, list(OLIST_ORDER_FEATURES)]
    result["price_total"] = result["price_total"].astype("float32")
    result["freight_total"] = result["freight_total"].astype("float32")
    result["n_items"] = result["n_items"].astype("int16")
    result["n_sellers"] = result["n_sellers"].astype("int16")
    result["product_weight_g"] = result["product_weight_g"].astype("float32")
    result["product_volume_cm3"] = result["product_volume_cm3"].astype("float32")
    result["product_photos_qty"] = result["product_photos_qty"].astype("int16")
    result["product_description_length"] = result["product_description_length"].astype("int32")
    result["payment_installments"] = result["payment_installments"].astype("int16")
    for column in ("same_state", "purchase_month", "purchase_dow", "purchase_hour"):
        result[column] = result[column].astype("int8")
    result["seller_prior_orders"] = result["seller_prior_orders"].astype("int32")
    result["seller_prior_complaint_rate"] = result["seller_prior_complaint_rate"].astype("float32")
    result["customer_prior_orders"] = result["customer_prior_orders"].astype("int32")
    check(result, "OLIST_ORDER_FEATURES")
    return result
