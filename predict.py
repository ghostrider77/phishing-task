import urllib.parse
from enum import StrEnum
from typing import Any

import joblib
from pydantic import BaseModel, field_serializer
from sklearn.pipeline import Pipeline


class Label(StrEnum):
    CLEAN = "clean"
    PHISHING = "phishing"


class ResponseModel(BaseModel):
    probability: float
    label: Label

    @field_serializer("label")
    def _serialize_label(self, label: Label) -> str:
        return label.value


def load_classifier(path: str) -> tuple[Pipeline, float]:
    try:
        artifact = joblib.load(path)
        return artifact["model"], artifact["threshold"]

    except FileNotFoundError as exc:
        raise RuntimeError(f"File does not exist: {path}") from exc

    except KeyError as exc:
        raise RuntimeError(f"Invalid model export: missing key {exc.args[0]!r}.") from exc

    except Exception as exc:
        raise RuntimeError(f"Failed to load model: {exc}") from exc


MODEL, THRESHOLD = load_classifier("model/url_phishing_classifier.joblib")


def transform_url(url: str) -> str:
    url = url.rstrip("/ ")
    try:
        parsed = urllib.parse.urlsplit(url)
        url = parsed.netloc + parsed.path
        if parsed.query:
            url = url + "?" + parsed.query

    except ValueError:
        pass

    url = urllib.parse.unquote(url)
    return url.lower()


def predict(url: str) -> dict[str, Any]:
    if not isinstance(url, str):
        raise ValueError(f"The input for the classification model is not a string: {url}")

    cleaned_url = transform_url(url)
    probability = MODEL.predict_proba([cleaned_url])[:, 1].item()
    prediction = Label.PHISHING if probability >= THRESHOLD else Label.CLEAN
    response = ResponseModel(probability=float(probability), label=prediction)
    return response.model_dump()
