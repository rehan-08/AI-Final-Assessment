"""
Task 1 helpers - text cleaning and sentiment labeling.

Approach
--------
Primary labeler : `cardiffnlp/twitter-roberta-base-sentiment-latest`, a RoBERTa
                  transformer (PyTorch) fine-tuned for 3-class sentiment
                  (negative / neutral / positive). Its label is the arg-max of
                  the softmax probabilities, so the output is deterministic.
Cross-checks    : VADER (lexicon/rule based) and TextBlob (pattern lexicon,
                  polarity score). Neither is used to set the label; they are
                  used only to measure agreement with the transformer as a
                  sanity check on the labels.

All three methods read the same cleaned text: "<subject>. <body>".
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
_SOFT_BREAK = re.compile(r"=\r?\n")                     # quoted-printable line wraps
_QP_CODE = re.compile(r"=(?:20|01|09|3D|0A|0D)")         # common QP escape codes
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+")
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Remove e-mail encoding artefacts, URLs and addresses; collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = _SOFT_BREAK.sub("", text)
    text = _QP_CODE.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = text.replace("*~*~*~*~*~*~*~*~*~*", " ")
    return _WS.sub(" ", text).strip()


def build_model_input(subject: str, body: str) -> str:
    """Subject carries useful tone ("Congrats!", "Problem with ..."), so prepend it."""
    subject = clean_text(subject)
    if subject.lower() in {"(no subject)", ""}:
        subject = ""
    body = clean_text(body)
    return f"{subject}. {body}".strip(". ").strip() if subject else body


# --------------------------------------------------------------------------- #
# Transformer labeler
# --------------------------------------------------------------------------- #
def transformer_sentiment(texts: list[str], batch_size: int = 32) -> pd.DataFrame:
    """Return P(neg), P(neu), P(pos) and the arg-max label for each text."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [t if t else "." for t in texts[i:i + batch_size]]
            enc = tok(batch, padding=True, truncation=True, max_length=512,
                      return_tensors="pt")
            probs.append(torch.softmax(model(**enc).logits, dim=-1).numpy())
    probs = np.vstack(probs)

    out = pd.DataFrame(probs, columns=["p_negative", "p_neutral", "p_positive"])
    out["sentiment"] = [LABEL_MAP[i] for i in probs.argmax(axis=1)]
    out["confidence"] = probs.max(axis=1)
    return out


# --------------------------------------------------------------------------- #
# VADER cross-check
# --------------------------------------------------------------------------- #
def vader_sentiment(texts: list[str]) -> pd.Series:
    """Standard VADER thresholds: compound >= 0.05 Positive, <= -0.05 Negative."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    compound = np.array([sia.polarity_scores(t)["compound"] for t in texts])
    return pd.Series(np.where(compound >= 0.05, "Positive",
                              np.where(compound <= -0.05, "Negative", "Neutral")))


# --------------------------------------------------------------------------- #
# TextBlob cross-check
# --------------------------------------------------------------------------- #
def textblob_sentiment(texts: list[str], threshold: float = 0.05) -> pd.DataFrame:
    """
    TextBlob (pattern lexicon) polarity in [-1, 1]. Same +/-0.05 neutral band
    as VADER so the two lexicon methods are comparable.
    """
    from textblob import TextBlob

    polarity = np.array([TextBlob(t).sentiment.polarity for t in texts])
    label = np.where(polarity > threshold, "Positive",
                     np.where(polarity < -threshold, "Negative", "Neutral"))
    return pd.DataFrame({"textblob_polarity": polarity, "textblob_sentiment": label})
