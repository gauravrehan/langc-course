"""Dependencies passed to graph nodes through LangGraph's runtime context.

Context is supplied on every invoke and is never checkpointed, so it can hold
live objects such as the repository and the prediction model.
"""

from dataclasses import dataclass

from pat.notify import Notifier
from pat.predict import PredictionModel
from pat.repository import PatRepository


@dataclass
class PatContext:
    repo: PatRepository
    model: PredictionModel
    notifier: Notifier
