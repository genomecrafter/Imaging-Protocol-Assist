# agents/fhir_models.py
from pydantic import BaseModel
from typing import List, Optional, Any


class RelatedArtifact(BaseModel):
    type: str
    display: Optional[str] = None
    url: Optional[str] = None


class Action(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None


class PlanDefinition(BaseModel):
    resourceType: str = "PlanDefinition"
    id: str
    title: str
    status: str = "active"
    description: Optional[str] = None
    useContext: Optional[List[Any]] = None
    action: Optional[List[Action]] = None
    relatedArtifact: Optional[List[RelatedArtifact]] = None


class ActivityDetail(BaseModel):
    description: str


class Activity(BaseModel):
    detail: ActivityDetail


class Note(BaseModel):
    text: str


class CarePlan(BaseModel):
    resourceType: str = "CarePlan"
    id: str = "protocol-recommendations"
    status: str = "active"
    intent: str = "plan"
    title: str = "CT Thoracic Aorta Protocol Recommendations"
    description: str = "Recommendations for imaging protocol selection and supportive care."
    activity: List[Activity]
    note: List[Note]


class BundleEntry(BaseModel):
    resource: dict


class FHIRBundle(BaseModel):
    resourceType: str = "Bundle"
    type: str = "collection"
    entry: List[BundleEntry]
