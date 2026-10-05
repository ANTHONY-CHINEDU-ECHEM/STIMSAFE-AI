"""FastAPI backend for the StimSafe dashboard."""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from stimsafe import __version__
from stimsafe.config import ROOT
from stimsafe.inference.recommender import InvalidPatientError, ProtocolRecommender, get_recommender

app = FastAPI(title="StimSafe AI", version=__version__,
              description="Safety constrained ovarian stimulation protocol recommendation. Research prototype. Not a medical device.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])


class Patient(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "age": 31, "amh_ng_ml": 4.8, "afc": 24, "fsh_iu_l": 5.6, "bmi": 22.5, "pcos": 1, "first_cycle": 1}})

    age: float = Field(..., ge=18, le=46)
    amh_ng_ml: float = Field(..., ge=0.01, le=25)
    afc: float = Field(..., ge=0, le=60)
    fsh_iu_l: float = Field(..., ge=0.5, le=40)
    bmi: float = Field(..., ge=15, le=50)
    pcos: int = Field(0, ge=0, le=1)
    first_cycle: int = Field(1, ge=0, le=1)
    previous_oocytes: Optional[float] = Field(None, ge=0, le=60)
    previous_ohss: int = Field(0, ge=0, le=1)


def recommender_dependency() -> ProtocolRecommender:
    try:
        return get_recommender()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/api/recommend")
def recommend(patient: Patient, recommender: ProtocolRecommender = Depends(recommender_dependency)) -> dict:
    try:
        return recommender.recommend(patient.model_dump())
    except InvalidPatientError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/similar")
def similar(patient: Patient, recommender: ProtocolRecommender = Depends(recommender_dependency)) -> dict:
    try:
        return recommender.similar_patients(patient.model_dump())
    except InvalidPatientError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/policies")
def policies(recommender: ProtocolRecommender = Depends(recommender_dependency)) -> dict:
    return {"policies": recommender.policy_table()}


# Serve the built React dashboard when it exists (run `make frontend` first).
_dist = ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="dashboard")
