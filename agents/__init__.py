from agents.acfql import ACFQLAgent
from agents.drift import DriftAgent
from agents.meanflow import MeanflowAgent

agents = dict(
    acfql=ACFQLAgent,
    drift=DriftAgent,
    meanflow=MeanflowAgent,
)
