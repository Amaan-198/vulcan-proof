"""FastAPI entry point for the Phase-5 demo surface."""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .service import Phase5Service, ROOT


app = FastAPI(title="Vulcan Proof", version="phase5")
_service: Phase5Service | None = None


def service() -> Phase5Service:
    """Return the process-wide lazy artefact service."""
    global _service
    if _service is None:
        _service = Phase5Service()
    return _service


@app.get("/health")
def health() -> dict[str, Any]:
    """Return a lightweight readiness response without fitting models."""
    current = service()
    return {
        "status": "ok",
        "phase": "5",
        "world": current.branch,
        "seed": current.seed,
        "model_loaded": current._models is not None,
    }


@app.get("/orders")
def orders(
    category: str | None = Query(default=None),
    query: str | None = Query(default=None),
    limit: int | None = Query(default=None),
    plans_only: bool = Query(default=False),
    package_ready_only: bool = Query(default=False),
) -> dict[str, Any]:
    """Return test orders for the order picker."""
    return service().orders(
        category=category,
        query=query,
        limit=limit,
        plans_only=plans_only,
        package_ready_only=package_ready_only,
    )


@app.get("/order/{order_id}/plan")
def order_plan(order_id: str) -> dict[str, Any]:
    """Return the stored plan and its model-side diagnostics."""
    try:
        return service().plan(order_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="test order not found") from exc


@app.get("/order/{order_id}/dispute-package")
def dispute_package(order_id: str) -> dict[str, Any]:
    """Return the materialised evidence package for one stored dispute."""
    try:
        return service().dispute_package(order_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="test order not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@app.get("/report/kappa")
def kappa_report() -> dict[str, Any]:
    """Return the explicit Phase-4 validation state."""
    return service().kappa_report()


@app.get("/report/arm4-policy")
def arm4_policy() -> dict[str, Any]:
    """Return the stored validation-tuned Arm 4 table."""
    return service().arm4_policy()


@app.get("/demo/script")
def demo_script() -> dict[str, Any]:
    """Return the generated eight-beat walkthrough."""
    path = ROOT / "outputs" / "phase5" / "demo_script.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    return service().demo_script()


@app.post("/explain")
def explain(payload: dict[str, Any]) -> dict[str, str]:
    """Return a short optional explanation with no decision authority."""
    if os.getenv("VP_EXPLAIN_LLM") != "1":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="optional explanation is disabled")
    evidence = payload.get("plan", {}).get("evidence", [])
    if isinstance(evidence, list) and evidence:
        names = ", ".join(str(name) for name in evidence)
        sentence = f"The stored plan selects {names} for this order context."
    else:
        sentence = "The stored plan requests no additional evidence for this order context."
    return {"sentence": sentence, "authority": "none"}


dist = ROOT / "vulcan_proof" / "ui" / "dist"
phase4_assets = ROOT / "outputs" / "phase4"
if phase4_assets.is_dir():
    app.mount("/phase4", StaticFiles(directory=phase4_assets), name="phase4-assets")
if dist.is_dir():
    app.mount("/", StaticFiles(directory=dist, html=True), name="ui")
else:
    @app.get("/", include_in_schema=False)
    def missing_ui() -> FileResponse:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="UI bundle has not been built")
