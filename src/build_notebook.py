"""Generates Employee_Sentiment_Analysis.ipynb (the main deliverable) from the cells below."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OBS = {}  # observation text, filled in by name from observations.py
try:
    from observations import OBS  # noqa: F401
except ImportError:
    pass

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
obs = lambda key: md(OBS.get(key, "**Observations:** _see report_"))

# --------------------------------------------------------------------------- #
md("""
# Employee Sentiment Analysis

**Goal:** label an unlabeled set of employee e-mails with sentiment, explore the data, score and
rank employees month by month, flag flight risks and model monthly sentiment with linear regression.

| Section | Task |
|---|---|
| 0 | Setup & data loading |
| 1 | Sentiment labeling (RoBERTa transformer + VADER cross-check) |
| 2 | Exploratory data analysis |
| 3 | Monthly employee score |
| 4 | Employee ranking |
| 5 | Flight-risk identification (rolling 30 days) |
| 6 | Predictive modeling (linear regression) |
| 7 | Summary |

Supporting code lives in `src/` (`sentiment.py`, `analysis.py`); this notebook calls it and shows every step.
Figures go to `visualization/`, tables to `outputs/`.
""")

# --------------------------------------------------------------------------- #
md("## 0. Setup & data loading")
code("""
# Standard libraries and plotting
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

# Make the helper modules in src/ importable
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from sentiment import (build_model_input, transformer_sentiment, vader_sentiment,
                       textblob_sentiment, MODEL_NAME)
from analysis import (monthly_scores, monthly_rankings, flight_risk, flight_risk_bruteforce,
                      monthly_features, ols_summary, SCORE_MAP)

# Output folders for figures and tables
VIS, OUT = ROOT / "visualization", ROOT / "outputs"
VIS.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)

# Display settings, one colour per sentiment class, and a fixed random seed
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"Positive": "#2a9d8f", "Neutral": "#adb5bd", "Negative": "#e76f51"}
SEED = 42
pd.set_option("display.width", 140)

def save(fig, name):
    \"\"\"Save a figure into visualization/ and show it inline.\"\"\"
    fig.tight_layout()
    fig.savefig(VIS / name, dpi=150, bbox_inches="tight")
    plt.show()
""")
code("""
# Load the raw, unlabeled dataset and take a first look
raw = pd.read_csv(ROOT / "data" / "test.csv")
print("shape:", raw.shape)
raw.head()
""")

# --------------------------------------------------------------------------- #
md("""
## 1. Sentiment labeling

**Approach**

1. **Cleaning** – the bodies are raw Enron e-mails with quoted-printable artefacts (`=20`, `=01`, soft line
   breaks `=\\n`), URLs and addresses. These are stripped and whitespace collapsed. The subject is prepended
   to the body because it often carries tone ("Congratulations!", "RE: problem with ...").
2. **Primary labeler – transformer (LLM-style encoder)**: `cardiffnlp/twitter-roberta-base-sentiment-latest`,
   a RoBERTa model (PyTorch) fine-tuned for 3-class sentiment. Each message gets P(neg), P(neu), P(pos) and
   the label is the **arg-max** – no hand-tuned thresholds, fully deterministic (`model.eval()`, no sampling).
   Input is truncated at 512 tokens (bodies are ≤ 999 characters, so in practice nothing is cut).
3. **Validation – VADER and TextBlob**, two lexicon-based methods, label the same text independently.
   VADER uses its standard compound-score cut-offs (±0.05). TextBlob gives a polarity in [−1, 1], and the same
   ±0.05 band is used for Neutral so the two lexicon methods are comparable. Agreement and Cohen's κ between
   all three methods act as a sanity check, and disagreements are inspected manually. Neither lexicon method
   changes the final label.

**Why this choice:** a transformer reads context ("not a problem", "unfortunately … will not work") that
word-count lexicons miss, while being small enough to run on CPU in ~1–2 minutes, so results are
reproducible without an API key. Labels are cached to `outputs/labeled_messages.csv`; delete that file to
recompute from scratch.
""")
code("""
# Rename the sender column, parse dates (month/day/year) and build the cleaned model input
df = raw.rename(columns={"from": "employee"}).copy()
df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
df["clean_text"] = [build_model_input(s, b) for s, b in zip(df["Subject"], df["body"])]

# The transformer is the slow step, so its output is cached; delete the CSV to recompute
cache = OUT / "labeled_messages.csv"
lab = pd.read_csv(cache) if cache.exists() else None
if lab is not None:
    # Guard against a cache built from different data
    assert len(lab) == len(df) and (lab["clean_text"].fillna("") == df["clean_text"]).all(), "stale cache"
    for c in ["p_negative", "p_neutral", "p_positive", "sentiment", "confidence", "vader_sentiment"]:
        df[c] = lab[c].values
    print("loaded cached transformer + VADER labels")
else:
    # Primary label: RoBERTa transformer (arg-max of the 3 class probabilities)
    df = pd.concat([df, transformer_sentiment(df["clean_text"].tolist())], axis=1)
    # Cross-check 1: VADER
    df["vader_sentiment"] = vader_sentiment(df["clean_text"].tolist()).values
    print("labels computed with", MODEL_NAME)

# Cross-check 2: TextBlob (fast, so it is always recomputed)
df = pd.concat([df.drop(columns=["textblob_polarity", "textblob_sentiment"], errors="ignore"),
                textblob_sentiment(df["clean_text"].tolist())], axis=1)
df.to_csv(cache, index=False)

df[["Subject", "clean_text", "sentiment", "confidence", "vader_sentiment", "textblob_sentiment"]].head(8)
""")
code("""
# Compare the three methods: label counts, pairwise agreement and Cohen's kappa
from itertools import combinations
from sklearn.metrics import cohen_kappa_score

METHODS = {"Transformer (final)": "sentiment", "VADER": "vader_sentiment", "TextBlob": "textblob_sentiment"}
counts = pd.DataFrame({m: df[c].value_counts() for m, c in METHODS.items()}).loc[["Positive", "Neutral", "Negative"]]
print("Label counts per method:"); print(counts, "\\n")

pairs = pd.DataFrame([{"pair": f"{a} vs {b}",
                       "agreement": (df[METHODS[a]] == df[METHODS[b]]).mean(),
                       "cohen_kappa": cohen_kappa_score(df[METHODS[a]], df[METHODS[b]])}
                      for a, b in combinations(METHODS, 2)]).set_index("pair").round(3)
pairs.to_csv(OUT / "labeler_agreement.csv")

# Share of messages where all three methods give the same label
all_agree = ((df.sentiment == df.vader_sentiment) & (df.sentiment == df.textblob_sentiment)).mean()
print(f"All three agree: {all_agree:.1%}   Mean transformer confidence: {df['confidence'].mean():.3f}")
pairs
""")
code("""
# Cross-tables of the final label against each lexicon method
for name, col in [("VADER", "vader_sentiment"), ("TextBlob", "textblob_sentiment")]:
    display(pd.crosstab(df["sentiment"], df[col], margins=True, rownames=["Transformer"], colnames=[name]))
""")
code("""
# Highest-confidence examples of each class, to check that the labels make sense
for lab_ in ["Positive", "Negative", "Neutral"]:
    print(f"==== {lab_} (highest confidence) ====")
    for t in df[df.sentiment == lab_].nlargest(3, "confidence")["clean_text"]:
        print(" -", t[:160])
""")
code("""
# Opposite-polarity disagreements (Positive vs Negative) between the transformer and each lexicon method
for name, col in [("VADER", "vader_sentiment"), ("TextBlob", "textblob_sentiment")]:
    opp = df[((df.sentiment == "Positive") & (df[col] == "Negative")) |
             ((df.sentiment == "Negative") & (df[col] == "Positive"))]
    print(f"\\n{len(opp)} opposite-polarity disagreements with {name}")
    # Show a few so they can be judged by eye
    for _, r in opp.head(4).iterrows():
        print(f"  [RoBERTa={r.sentiment:8s} {name}={r[col]:8s}]", r.clean_text[:140])
""")
code("""
# Figure: confusion heatmaps against VADER and TextBlob, plus transformer confidence per label
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
order = ["Negative", "Neutral", "Positive"]
for a, (name, col) in zip(ax[:2], [("VADER", "vader_sentiment"), ("TextBlob", "textblob_sentiment")]):
    cm = pd.crosstab(df["sentiment"], df[col]).reindex(index=order, columns=order, fill_value=0)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=a, cbar=False)
    a.set(title=f"Transformer vs {name}", xlabel=name, ylabel="Transformer (final)")
# Confidence distribution of the transformer for each assigned label
for lab_ in order:
    sns.kdeplot(df.loc[df.sentiment == lab_, "confidence"], ax=ax[2], label=lab_, color=PALETTE[lab_], fill=True, alpha=.3)
ax[2].set(title="Transformer confidence by assigned label", xlabel="max class probability"); ax[2].legend()
save(fig, "01_label_validation.png")
""")
obs("labeling")

# --------------------------------------------------------------------------- #
md("## 2. Exploratory data analysis\n### 2.1 Structure, types, missing values, duplicates")
code("""
# Column types, missing values and duplicates in the raw data
print(raw.dtypes, "\\n")
print("missing values:\\n", raw.isna().sum(), "\\n")
print("exact duplicate rows:", raw.duplicated().sum())
print("duplicate (employee, date, body):", df.duplicated(["employee", "date", "body"]).sum())
# Time span and number of senders
print("date range:", df.date.min().date(), "->", df.date.max().date(), f"({df.date.dt.to_period('M').nunique()} months)")
print("employees:", df.employee.nunique())
# Message length in characters and words (on the cleaned text)
df["char_len"] = df["clean_text"].str.len()
df["word_count"] = df["clean_text"].str.split().str.len()
df[["char_len", "word_count"]].describe().round(1)
""")
code("""
# Per-employee profile: message volume, sentiment mix and average length
emp_tbl = (df.groupby("employee")
             .agg(messages=("sentiment", "size"),
                  positive=("sentiment", lambda s: (s=="Positive").mean()),
                  neutral=("sentiment", lambda s: (s=="Neutral").mean()),
                  negative=("sentiment", lambda s: (s=="Negative").mean()),
                  avg_words=("word_count", "mean"))
             .sort_values("messages", ascending=False))
emp_tbl.style.format({"positive": "{:.1%}", "neutral": "{:.1%}", "negative": "{:.1%}", "avg_words": "{:.0f}"})
""")
md("### 2.2 Sentiment distribution")
code("""
# Counts and shares of each sentiment label
dist = df["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"])
print(pd.DataFrame({"count": dist, "share": (dist / len(df)).round(3)}))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
# Left: overall bar chart with counts and percentages
ax[0].bar(dist.index, dist.values, color=[PALETTE[k] for k in dist.index])
for i, v in enumerate(dist.values): ax[0].text(i, v, f"{v}\\n({v/len(df):.0%})", ha="center", va="bottom")
ax[0].set(title="Overall sentiment distribution", ylabel="messages"); ax[0].margins(y=.15)
# Right: stacked share of each label per employee
share = pd.crosstab(df.employee, df.sentiment, normalize="index")[["Negative", "Neutral", "Positive"]].sort_values("Positive")
share.plot.barh(stacked=True, ax=ax[1], color=[PALETTE[c] for c in share.columns], width=.8)
ax[1].set(title="Sentiment mix per employee", xlabel="share of messages", ylabel=""); ax[1].legend(loc="lower right", fontsize=8)
save(fig, "02_sentiment_distribution.png")
""")
md("### 2.3 Trends over time")
code("""
# Count of each label per calendar month, and net sentiment (positive minus negative)
df["month"] = df["date"].dt.to_period("M")
m = pd.crosstab(df.month, df.sentiment)[["Positive", "Neutral", "Negative"]]
m.index = m.index.to_timestamp()
net = (m["Positive"] - m["Negative"])
fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
# Top: stacked monthly volume; bottom: net sentiment and positive / negative percentages
ax[0].stackplot(m.index, m.T.values, labels=m.columns, colors=[PALETTE[c] for c in m.columns], alpha=.85)
ax[0].set(title="Monthly message volume by sentiment", ylabel="messages"); ax[0].legend(loc="lower right", fontsize=8)
ax[1].plot(m.index, net, marker="o", color="#264653", label="net sentiment (pos - neg)")
ax[1].plot(m.index, (m["Positive"] / m.sum(axis=1)) * 100, marker="s", color=PALETTE["Positive"], label="% positive")
ax[1].plot(m.index, (m["Negative"] / m.sum(axis=1)) * 100, marker="^", color=PALETTE["Negative"], label="% negative")
ax[1].set(title="Net sentiment and positive / negative share per month", ylabel="value"); ax[1].legend()
save(fig, "03_sentiment_trend_monthly.png")
m.assign(total=m.sum(axis=1), net=net).T
""")
code("""
# Sentiment share by day of week (left) and by year (right)
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
dow = pd.crosstab(df.date.dt.day_name(), df.sentiment, normalize="index")[["Negative","Neutral","Positive"]]
dow = dow.reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
dow.plot.bar(stacked=True, ax=ax[0], color=[PALETTE[c] for c in dow.columns], rot=30)
ax[0].set(title="Sentiment share by weekday", ylabel="share", xlabel=""); ax[0].legend(fontsize=8)
yr = pd.crosstab(df.date.dt.year, df.sentiment, normalize="index")[["Negative","Neutral","Positive"]]
yr.plot.bar(stacked=True, ax=ax[1], color=[PALETTE[c] for c in yr.columns], rot=0)
ax[1].set(title="Sentiment share by year", ylabel="share", xlabel=""); ax[1].legend(fontsize=8)
save(fig, "04_weekday_year_patterns.png")
print("messages per weekday:\\n", df.date.dt.day_name().value_counts())
""")
md("### 2.4 Message length vs sentiment, and other patterns / anomalies")
code("""
# Left: word count per label (log scale, lengths are skewed)
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
sns.boxplot(data=df, x="sentiment", y="word_count", order=["Positive","Neutral","Negative"], palette=PALETTE, ax=ax[0])
ax[0].set(title="Word count by sentiment", yscale="log")
# Right: net sentiment per message for each employee and quarter
heat = df.pivot_table(index="employee", columns=df.date.dt.to_period("Q").astype(str),
                      values="sentiment", aggfunc=lambda s: ((s=="Positive").sum()-(s=="Negative").sum())/len(s))
sns.heatmap(heat, cmap="RdYlGn", center=0, ax=ax[1], cbar_kws={"label": "net sentiment / msg"})
ax[1].set(title="Net sentiment per message, employee x quarter", xlabel="", ylabel="")
save(fig, "05_length_and_heatmap.png")
print(df.groupby("sentiment")[["char_len", "word_count"]].median())
""")
code("""
# Anomalies: very short messages, subjects, repeated bodies
print("bodies under 20 characters:", (df.body.str.strip().str.len() < 20).sum())
print("'(No Subject)':", (df.Subject == "(No Subject)").sum())
print("replies (RE:):", df.Subject.str.match(r"(?i)^\\s*re:").sum(), "| forwards (FW:):", df.Subject.str.match(r"(?i)^\\s*fwd?:").sum())
print("\\nmost repeated bodies:")
print(df.body.str.strip().str[:70].value_counts().head(5))
print("\\nsentiment by reply/forward/new:")
# Classify each message by its subject prefix
kind = np.select([df.Subject.str.match(r"(?i)^\\s*re:"), df.Subject.str.match(r"(?i)^\\s*fwd?:")], ["reply", "forward"], "new")
pd.crosstab(kind, df.sentiment, normalize="index").round(3)
""")
obs("eda")

# --------------------------------------------------------------------------- #
md("""
## 3. Monthly employee score

**Method**

* Each message scores **+1** (Positive), **−1** (Negative) or **0** (Neutral).
* Messages are grouped by `employee` and **calendar month** (`date.to_period("M")`, e.g. `2010-03`).
* The monthly score is the **sum of message scores inside that month only**. Because each month is its own
  group, the score automatically **resets to 0 at the start of every month** – nothing is carried over.
* The running (cumulative) score *within* a month is also computed per message so the reset can be verified.
""")
code("""
# Message score: +1 Positive, -1 Negative, 0 Neutral
df["score"] = df["sentiment"].map(SCORE_MAP)
# Running total inside each employee-month (restarts every month)
df = df.sort_values(["employee", "date"]).reset_index(drop=True)
df["cum_score_in_month"] = df.groupby(["employee", "month"])["score"].cumsum()

# One row per employee and month with the monthly score and label counts
scores = monthly_scores(df)
scores.to_csv(OUT / "monthly_scores.csv", index=False)
print(scores.shape, "employee-month rows")
scores.head(10)
""")
code("""
# --- validation of the aggregation ---
# 1) monthly score equals last cumulative value of the month
last_cum = df.groupby(["employee", "month"])["cum_score_in_month"].last().reset_index(name="last_cum")
chk = scores.merge(last_cum, on=["employee", "month"])
assert (chk.score == chk.last_cum).all()
# 2) score == positives - negatives, and counts add up
assert (scores.score == scores.n_positive - scores.n_negative).all()
assert scores.n_messages.sum() == len(df)
# 3) reset: the first message of each month has cumulative score equal to its own score
first = df.groupby(["employee", "month"]).head(1)
assert (first.cum_score_in_month == first.score).all()
print("all aggregation checks passed")

# Heatmap of every employee's monthly score
pivot = scores.pivot(index="employee", columns="month", values="score")
pivot.columns = pivot.columns.astype(str)
fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(pivot, cmap="RdYlGn", center=0, annot=True, fmt=".0f", annot_kws={"size": 7}, ax=ax, cbar_kws={"label": "monthly score"})
ax.set(title="Monthly sentiment score per employee (resets every month)", xlabel="", ylabel="")
save(fig, "06_monthly_score_heatmap.png")
""")
obs("scores")

# --------------------------------------------------------------------------- #
md("""
## 4. Employee ranking

For **every month** (rankings are derived directly from the Task 3 table):

* **Top 3 positive** – sort by monthly score **descending**, ties broken **alphabetically** by employee.
* **Top 3 negative** – sort by monthly score **ascending** (most negative first), ties broken **alphabetically**.

Only employees who sent at least one message that month are ranked.
""")
code("""
# Top 3 positive and top 3 negative employees for every month, saved as CSV
top_pos, top_neg = monthly_rankings(scores, k=3)
top_pos.to_csv(OUT / "top3_positive_by_month.csv", index=False)
top_neg.to_csv(OUT / "top3_negative_by_month.csv", index=False)

# Side-by-side table: one row per month, "name (score)" in each rank column
wide = lambda t: t.assign(cell=t.employee.str.replace("@enron.com", "").str.replace("@ipgdirect.com", "") + " (" + t.score.astype(str) + ")") \\
                  .pivot(index="month", columns="rank", values="cell")
side = pd.concat({"Top 3 positive": wide(top_pos), "Top 3 negative": wide(top_neg)}, axis=1)
side
""")
code("""
# Check, for every month, that rows are ordered by score and then alphabetically
for t, asc in [(top_pos, False), (top_neg, True)]:
    for _, g in t.groupby("month"):
        s = list(zip(g.score if asc else -g.score, g.employee))
        assert s == sorted(s)
print("ordering & alphabetical tie-break verified for all months")

# Show the rankings for the most recent month
latest = scores.month.max()
print("Latest month:", latest)
display(top_pos[top_pos.month == latest]); display(top_neg[top_neg.month == latest])
""")
code("""
# Overall view: total score across the whole period + how often each employee lands in a monthly top 3
overall = (scores.groupby("employee")
                 .agg(total_score=("score", "sum"), avg_monthly_score=("score", "mean"), months_active=("score", "size"))
                 .join(top_pos.employee.value_counts().rename("times_top3_positive"))
                 .join(top_neg.employee.value_counts().rename("times_top3_negative"))
                 .fillna(0).astype({"times_top3_positive": int, "times_top3_negative": int}))
overall = overall.reset_index().sort_values(["total_score", "employee"], ascending=[False, True]).reset_index(drop=True)
overall.to_csv(OUT / "overall_employee_ranking.csv", index=False)

# Left: total score per employee; right: number of months in each top-3 list
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
o = overall.sort_values("total_score")
ax[0].barh(o.employee, o.total_score, color=["#2a9d8f" if v >= o.total_score.median() else "#e9c46a" for v in o.total_score])
ax[0].set(title="Total sentiment score, Jan 2010 – Dec 2011", xlabel="sum of monthly scores")
cnt = overall.set_index("employee")[["times_top3_positive", "times_top3_negative"]].sort_values("times_top3_positive")
cnt.plot.barh(ax=ax[1], color=[PALETTE["Positive"], PALETTE["Negative"]])
ax[1].set(title="Months in which the employee ranked in a top 3", xlabel="number of months (of 24)", ylabel="")
save(fig, "07_employee_rankings.png")
overall
""")
obs("ranking")

# --------------------------------------------------------------------------- #
md("""
## 5. Flight-risk identification

**Rule:** an employee is a flight risk if they sent **4 or more Negative messages within any rolling
30-day window**, regardless of calendar months and regardless of their score.

**Implementation:** each employee's Negative messages are sorted by date and scanned with a two-pointer
sliding window covering 30 calendar days inclusive (`date_j − date_i ≤ 29 days`). The maximum count in any
window is recorded together with the window dates. Every message counts (two on the same day = 2).

**Robustness:** the result is cross-checked against an independent brute-force implementation, and a
sensitivity check shows how the list changes with the threshold.
""")
code("""
# Peak count of negatives in any rolling 30-day window, per employee
risk = flight_risk(df, threshold=4, window_days=30)
risk.to_csv(OUT / "flight_risk.csv", index=False)
flagged = risk.loc[risk.flight_risk, "employee"].tolist()
# The independent brute-force method must flag exactly the same employees
assert set(flagged) == flight_risk_bruteforce(df), "fast and brute-force results differ"
print("brute-force cross-check passed")
print(f"\\nFLIGHT RISK employees ({len(flagged)}):", *flagged, sep="\\n  ")
risk
""")
code("""
# Sensitivity: how many employees would be flagged with other thresholds / window sizes
sens = pd.DataFrame({w: {t: flight_risk(df, t, w).flight_risk.sum() for t in [3, 4, 5, 6]} for w in [14, 30, 60]})
sens.index.name = "threshold (negatives)"; sens.columns.name = "window (days)"
sens
""")
code("""
# Rolling 30-day negative count per employee over time
neg = df[df.sentiment == "Negative"].groupby(["employee", "date"]).size().rename("n").reset_index()
full_days = pd.date_range(df.date.min(), df.date.max(), freq="D")
fig, ax = plt.subplots(figsize=(14, 5))
for emp, g in neg.groupby("employee"):
    # Daily negatives filled to every calendar day, then summed over the last 30 days
    s =g.set_index("date")["n"].reindex(full_days, fill_value=0).rolling(30, min_periods=1).sum()
    ax.plot(s.index, s.values, lw=2.2 if emp in flagged else 0.8, alpha=1 if emp in flagged else .35,
            label=emp.split("@")[0] + (" (RISK)" if emp in flagged else ""))
ax.axhline(4, color="red", ls="--", lw=1.2); ax.text(full_days[-1], 4.1, "flight-risk threshold = 4", color="red", ha="right")
ax.set(title="Rolling 30-day count of negative messages", ylabel="negatives in last 30 days", ylim=(-0.2, 4.8))
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
save(fig, "08_flight_risk_rolling.png")
""")
code("""
# Peak 30-day negative count per employee; flagged employees in red
fig, ax = plt.subplots(figsize=(9, 4))
r = risk.sort_values("max_negatives_in_30d")
ax.barh(r.employee, r.max_negatives_in_30d, color=[PALETTE["Negative"] if f else "#adb5bd" for f in r.flight_risk])
ax.axvline(4, color="red", ls="--"); ax.set(title="Peak negatives in any 30-day window (red = flight risk)", xlabel="negative messages")
save(fig, "09_flight_risk_peak.png")
""")
obs("flight")

# --------------------------------------------------------------------------- #
md("""
## 6. Predictive modeling – linear regression

**Target:** monthly sentiment score of an employee (Task 3).
**Unit of observation:** employee-month (rows from Task 3).

**Features** – deliberately *not* derived from the sentiment labels (that would leak the target):

| feature | meaning |
|---|---|
| `n_messages` | messages sent in the month (frequency) |
| `avg_char_len`, `avg_word_count`, `total_words` | message length |
| `avg_subject_len` | subject length |
| `reply_ratio`, `fwd_ratio` | share of replies / forwards |
| `exclaim_per_msg`, `question_per_msg` | punctuation style |
| `weekend_ratio` | share sent on weekends |

**Evaluation:** 80/20 random train/test split (seed 42), 5-fold cross-validation, and a time-based split
(train 2010 → test 2011). Metrics: R², MAE, RMSE, compared with a mean-only baseline.
""")
code("""
# Modeling tools from scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.dummy import DummyRegressor

# Employee-month features joined to the Task 3 score (the target)
feat = monthly_features(df).merge(scores[["employee", "month", "score"]], on=["employee", "month"])
FEATURES = ["n_messages", "avg_char_len", "avg_word_count", "total_words", "avg_subject_len",
            "reply_ratio", "fwd_ratio", "exclaim_per_msg", "question_per_msg", "weekend_ratio"]
X, y = feat[FEATURES], feat["score"]
print(feat.shape); feat[FEATURES + ["score"]].describe().T.round(2)
""")
code("""
# Correlation between all features and the score, to spot drivers and collinearity
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(feat[FEATURES + ["score"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 8})
ax.set_title("Correlation of features with monthly score")
save(fig, "10_feature_correlation.png")
""")
code("""
def metrics(y_true, y_pred):
    \"\"\"R2, MAE and RMSE for one set of predictions.\"\"\"
    return {"R2": r2_score(y_true, y_pred), "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred) ** .5}

# 80/20 random split; scaled linear regression vs a predict-the-mean baseline
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
model = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
base = DummyRegressor().fit(X_tr, y_tr)

# time-based split: learn on 2010, predict 2011
is_2010 = feat.month.dt.year == 2010
model_t = make_pipeline(StandardScaler(), LinearRegression()).fit(X[is_2010], y[is_2010])

# 5-fold cross-validation on all rows
cv = cross_validate(make_pipeline(StandardScaler(), LinearRegression()), X, y,
                    cv=KFold(5, shuffle=True, random_state=SEED),
                    scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"])

# Collect all evaluations in one table
results = pd.DataFrame({
    "Linear reg - train": metrics(y_tr, model.predict(X_tr)),
    "Linear reg - test (random 20%)": metrics(y_te, model.predict(X_te)),
    "Baseline (mean) - test": metrics(y_te, base.predict(X_te)),
    "Linear reg - 5-fold CV (mean)": {"R2": cv["test_r2"].mean(), "MAE": -cv["test_neg_mean_absolute_error"].mean(),
                                      "RMSE": -cv["test_neg_root_mean_squared_error"].mean()},
    "Linear reg - time split (2011)": metrics(y[~is_2010], model_t.predict(X[~is_2010])),
}).T.round(3)
results.to_csv(OUT / "model_metrics.csv")
print("CV R2 per fold:", cv["test_r2"].round(3))
results
""")
code("""
# OLS with standard errors and p-values, to see which features matter
coefs = ols_summary(pd.DataFrame(StandardScaler().fit_transform(X), columns=FEATURES), y)
coefs.to_csv(OUT / "model_coefficients.csv")
print("OLS on all rows, standardized features (coef = change in score per 1 SD of the feature)")
coefs.round(4)
""")
code("""
# Parsimonious model: only the frequency feature, to show how much it alone explains
simple = LinearRegression().fit(X_tr[["n_messages"]], y_tr)
print("n_messages only  -> test R2 = %.3f, slope = %.3f per message" %
      (r2_score(y_te, simple.predict(X_te[["n_messages"]])), simple.coef_[0]))
""")
code("""
# Test-set predictions and residuals
pred = model.predict(X_te); resid = y_te - pred
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
# Actual vs predicted (points on the red line are perfect predictions)
ax[0].scatter(y_te, pred, alpha=.7, color="#264653"); lim = [min(y.min(), pred.min()) - 1, max(y.max(), pred.max()) + 1]
ax[0].plot(lim, lim, "r--"); ax[0].set(title=f"Actual vs predicted (test)  R²={r2_score(y_te, pred):.2f}", xlabel="actual score", ylabel="predicted score")
# Residuals should scatter randomly around zero
ax[1].scatter(pred, resid, alpha=.7, color="#e76f51"); ax[1].axhline(0, color="k", lw=1)
ax[1].set(title="Residuals vs predicted", xlabel="predicted", ylabel="residual")
# Standardized coefficients: effect size of each feature
c = coefs.drop("intercept")["coef"].sort_values()
ax[2].barh(c.index, c.values, color=["#e76f51" if v < 0 else "#2a9d8f" for v in c.values])
ax[2].set(title="Standardized coefficients", xlabel="effect on score per 1 SD")
save(fig, "11_model_performance.png")

# Bar chart comparing metrics across the evaluation setups
fig, ax = plt.subplots(figsize=(8, 4))
results[["R2", "MAE", "RMSE"]].plot.bar(ax=ax, rot=20, color=["#264653", "#2a9d8f", "#e9c46a"])
ax.set(title="Model evaluation metrics"); ax.axhline(0, color="k", lw=.8)
save(fig, "12_model_metrics.png")
""")
obs("model")

# --------------------------------------------------------------------------- #
md("## 7. Summary")
code("""
# Final answers to the project questions
latest_pos = top_pos[top_pos.month == latest].employee.tolist()
latest_neg = top_neg[top_neg.month == latest].employee.tolist()
print("Overall top 3 positive (total score):", overall.head(3).employee.tolist())
print("Overall top 3 negative (total score):", overall.sort_values(["total_score","employee"]).head(3).employee.tolist())
print(f"Latest month {latest} top 3 positive:", latest_pos)
print(f"Latest month {latest} top 3 negative:", latest_neg)
print("Flight risks:", flagged)
print("Labeled data with sentiment column -> outputs/labeled_messages.csv")
""")
obs("summary")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, ROOT / "Employee_Sentiment_Analysis.ipynb")
print("notebook written")
