"""
Tasks 3-6 helpers - monthly scoring, ranking, flight risk and regression.
All functions are pure (DataFrame in, DataFrame out) so each step can be
unit-checked in the notebook.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SCORE_MAP = {"Positive": 1, "Negative": -1, "Neutral": 0}


# --------------------------------------------------------------------------- #
# Task 3 - monthly score
# --------------------------------------------------------------------------- #
def monthly_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (employee, calendar month).  The score is the sum of message
    scores inside that month only, so it implicitly resets to 0 at the start of
    every month.  Employees with no messages in a month get no row.
    """
    d = df.assign(score=df["sentiment"].map(SCORE_MAP),
                  month=df["date"].dt.to_period("M"))
    g = d.groupby(["employee", "month"])
    out = g.agg(
        score=("score", "sum"),
        n_messages=("score", "size"),
        n_positive=("sentiment", lambda s: (s == "Positive").sum()),
        n_negative=("sentiment", lambda s: (s == "Negative").sum()),
        n_neutral=("sentiment", lambda s: (s == "Neutral").sum()),
    ).reset_index()
    return out.sort_values(["month", "employee"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Task 4 - rankings
# --------------------------------------------------------------------------- #
def monthly_rankings(scores: pd.DataFrame, k: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Top-k positive : score descending, ties broken alphabetically by employee.
    Top-k negative : score ascending (most negative first), ties alphabetical.
    """
    pos = (scores.sort_values(["month", "score", "employee"], ascending=[True, False, True])
                 .groupby("month").head(k))
    neg = (scores.sort_values(["month", "score", "employee"], ascending=[True, True, True])
                 .groupby("month").head(k))
    for t in (pos, neg):
        t.insert(1, "rank", t.groupby("month").cumcount() + 1)
    cols = ["month", "rank", "employee", "score", "n_messages"]
    return pos[cols].reset_index(drop=True), neg[cols].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Task 5 - flight risk (rolling 30 days)
# --------------------------------------------------------------------------- #
def flight_risk(df: pd.DataFrame, threshold: int = 4, window_days: int = 30) -> pd.DataFrame:
    """
    Flag an employee if ANY 30-day rolling window contains >= `threshold`
    Negative messages, independent of calendar months and of the score.

    Window definition: the 30 calendar days [d, d+29] inclusive. Using a
    two-pointer scan over each employee's sorted negative-message dates, the
    window ending at message j starts at the first message i with
    date_j - date_i <= window_days - 1.  Every message counts (two negatives
    on the same day count as two).
    """
    rows = []
    neg = df[df["sentiment"] == "Negative"].sort_values("date")
    for emp, g in neg.groupby("employee"):
        dates = g["date"].to_numpy(dtype="datetime64[D]")
        best, best_start, best_end, first_hit = 0, None, None, None
        i = 0
        for j in range(len(dates)):
            while (dates[j] - dates[i]).astype(int) > window_days - 1:
                i += 1
            cnt = j - i + 1
            if cnt > best:
                best, best_start, best_end = cnt, dates[i], dates[j]
            if cnt >= threshold and first_hit is None:
                first_hit = dates[j]
        rows.append({
            "employee": emp,
            "total_negative": len(dates),
            "max_negatives_in_30d": best,
            "peak_window_start": pd.Timestamp(best_start),
            "peak_window_end": pd.Timestamp(best_end),
            "first_flagged_on": pd.Timestamp(first_hit) if first_hit is not None else pd.NaT,
            "flight_risk": best >= threshold,
        })
    out = pd.DataFrame(rows)
    # employees with zero negative messages are explicitly not at risk
    missing = sorted(set(df["employee"]) - set(out["employee"]))
    if missing:
        out = pd.concat([out, pd.DataFrame({"employee": missing, "total_negative": 0,
                                            "max_negatives_in_30d": 0, "flight_risk": False})])
    return out.sort_values(["flight_risk", "max_negatives_in_30d", "employee"],
                           ascending=[False, False, True]).reset_index(drop=True)


def flight_risk_bruteforce(df: pd.DataFrame, threshold: int = 4, window_days: int = 30) -> set:
    """Independent O(n^2) re-implementation used only to validate `flight_risk`."""
    flagged = set()
    neg = df[df["sentiment"] == "Negative"]
    for emp, g in neg.groupby("employee"):
        for start in g["date"].unique():
            end = start + pd.Timedelta(days=window_days - 1)
            if ((g["date"] >= start) & (g["date"] <= end)).sum() >= threshold:
                flagged.add(emp)
                break
    return flagged


# --------------------------------------------------------------------------- #
# Task 6 - feature table for regression
# --------------------------------------------------------------------------- #
def monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Employee-month features that do NOT use the sentiment label itself
    (avoids target leakage): volume, length and style of the messages.
    """
    d = df.assign(month=df["date"].dt.to_period("M"),
                  char_len=df["clean_text"].str.len(),
                  word_count=df["clean_text"].str.split().str.len(),
                  subj_len=df["Subject"].str.len(),
                  is_reply=df["Subject"].str.match(r"(?i)^\s*re:").astype(int),
                  is_fwd=df["Subject"].str.match(r"(?i)^\s*(fw|fwd):").astype(int),
                  n_exclaim=df["clean_text"].str.count("!"),
                  n_question=df["clean_text"].str.count(r"\?"),
                  weekend=(df["date"].dt.dayofweek >= 5).astype(int))
    g = d.groupby(["employee", "month"])
    return g.agg(
        n_messages=("char_len", "size"),
        avg_char_len=("char_len", "mean"),
        avg_word_count=("word_count", "mean"),
        total_words=("word_count", "sum"),
        avg_subject_len=("subj_len", "mean"),
        reply_ratio=("is_reply", "mean"),
        fwd_ratio=("is_fwd", "mean"),
        exclaim_per_msg=("n_exclaim", "mean"),
        question_per_msg=("n_question", "mean"),
        weekend_ratio=("weekend", "mean"),
    ).reset_index()


def ols_summary(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Plain OLS coefficients with standard errors, t-stats and p-values."""
    from scipy import stats

    Xc = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(Xc, y.to_numpy(dtype=float), rcond=None)
    resid = y.to_numpy() - Xc @ beta
    dof = len(y) - Xc.shape[1]
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(Xc.T @ Xc)))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    return pd.DataFrame({"coef": beta, "std_err": se, "t": t, "p_value": p},
                        index=["intercept"] + list(X.columns))
