"""
Narrative text for the notebook (OBS) and the Word report (REPORT).
Written after reviewing the executed results. Every number here comes from the
notebook outputs / outputs/*.csv.
"""

OBS = {
"labeling": """
**Observations – labeling**

* Final label distribution: **Neutral 1,566 (71.5%)**, **Positive 487 (22.2%)**, **Negative 138 (6.3%)**. That fits
  corporate e-mail, which is mostly scheduling, documents and FYIs.
* Mean transformer confidence is **0.81**. The highest-confidence examples make sense: promotion
  congratulations are Positive, "comed sucks!" is Negative, and logistics notes are Neutral.
* **Three methods compared:**

  | | Positive | Neutral | Negative | Agreement with transformer | Cohen's κ |
  |---|---|---|---|---|---|
  | Transformer (final) | 487 | 1,566 | 138 | – | – |
  | VADER | 1,574 | 444 | 173 | 42.9% | 0.17 |
  | TextBlob | 1,165 | 758 | 268 | 52.2% | 0.24 |

  VADER and TextBlob agree with each other more (65.3%, κ 0.36) than with the transformer. All three agree on 33.9% of messages.
* Both lexicon methods label far more messages Positive. They count polite words ("please", "thanks", "good",
  "great") as positive, so routine business e-mails come out positive. Most disagreements are therefore
  *Neutral (transformer) vs Positive (lexicon)*: 1,059 with VADER and 710 with TextBlob.
* Opposite-polarity disagreements (Positive vs Negative) are rare: **69 with VADER (3.1%)** and **51 with TextBlob (2.3%)**.
  In these the transformer is usually right. It labels *"Unfortunately, today is not going to work…"* and
  *"Please remove me from your email list"* as Negative where VADER says Positive. It also reads
  *"it wouldn't hurt to have another guy"* as Positive, where TextBlob sees "hurt" and says Negative.
* For this reason the transformer is kept as the source of truth, and VADER and TextBlob are used only as sanity checks.
  The transformer is the most conservative of the three, which suits a flight-risk use case where false
  "negative" flags have a cost.
""",
"eda": """
**Observations – EDA**

* **Structure:** 2,191 messages, 4 text columns (`Subject`, `body`, `date`, `from`), **no missing values and no
  duplicate rows**. The data covers 10 employees and 24 months (Jan 2010 – Dec 2011).
* **Anomaly – evenly spread dates:** every month has 91–92 messages and weekends hold about 31% of the traffic
  (Sat 335, Sun 334, the busiest days next to Friday). The body text refers to 2000–2001 events while the dates
  are 2010–2011. The dates were most likely re-sampled when the dataset was built, so day-of-week and seasonal
  effects are not meaningful. Weekday and weekend sentiment are almost identical (6.3% negative for both).
* **Trend:** sentiment is stable over time. Net (pos − neg) per month ranges from **+6 (Feb 2010)** to
  **+22 (Oct 2011)**. 2011 is a little more positive than 2010 (23.3% vs 21.1% positive, 5.7% vs 6.9% negative).
  Oct 2010 has the most negative messages (10).
* **Employees:** johnny.palmer (29% positive) and sally.beck (27%) are the most positive. bobette.riner (8.8%)
  and john.arnold (8.2%) have the highest negative share. don.baughman and rhonda.denton are the most neutral (~78%).
* **Length:** negative messages are shorter (median 24 words) than neutral (30) and positive (38) ones. Complaints
  tend to be short and direct.
* **Message type:** replies (`RE:`) carry the most emotion: 9.6% negative and 33.4% positive, against 1.4% and
  16.7% for forwards. Back-and-forth conversation is where engagement, good or bad, shows up.
* **Other anomalies:** 30 empty bodies, 100 bodies under 20 characters, 141 "(No Subject)", and one automated
  notice ("We have received the executed EEI Master…") repeated 26 times. These are kept, because they are real
  messages an employee sent, and they are labeled Neutral as expected.
""",
"scores": """
**Observations – scores**

* There are 240 employee-month rows: every employee is active in all 24 months. All aggregation checks pass:
  score = positives − negatives, message counts add up to 2,191, and the first message of each month
  starts a new running total (the reset is verified).
* Monthly scores range from **−3** (patti.thompson, Oct 2010) to **+8** (sally.beck, Aug 2010). The median is 1,
  and only 26 of 240 employee-months (11%) are negative.
* Because each message adds at most ±1, the score depends on volume. Busy employees can reach larger
  absolute scores. This is examined in Task 6.
""",
"ranking": """
**Observations – ranking**

* The ranking is computed from the Task 3 table for each of the 24 months. Ordering and alphabetical
  tie-breaks are checked by assertions.
* **Overall (sum of monthly scores):** johnny.palmer (51), sally.beck (49) and lydia.delgado (48) lead.
  rhonda.denton (16), john.arnold (22) and don.baughman (25) trail.
* **Dec 2011 (latest month):** positive – kayne.coulter (4), eric.bass (3), patti.thompson (3);
  negative – bobette.riner (0), don.baughman (1), john.arnold (1).
* Bottom-3 lists often contain scores of 0 or even positive values, because negative months are rare. A place in
  a "negative" list therefore means *least positive*, not necessarily unhappy.
* Volatility: bobette.riner is in the top 3 positive 13 times and the top 3 negative 8 times. don.baughman and
  kayne.coulter appear in the negative list most often (12 and 11 months).
""",
"flight": """
**Observations – flight risk**

* **3 employees are flagged:** **john.arnold** (4 negatives between 2010-06-06 and 06-21), **patti.thompson**
  (4 between 2010-10-04 and 10-29) and **rhonda.denton** (4 between 2010-12-24 and 2011-01-01).
* rhonda.denton's window crosses a year boundary. A per-calendar-month count would have missed it,
  which shows why a rolling window is needed.
* The two-pointer implementation and an independent brute-force check return the same set.
* bobette.riner (19 negatives in total, peak 3) and lydia.delgado (17, peak 3) sit one message below the threshold.
  They are the next people to watch. With a threshold of 3, 5 employees would be flagged.
* The flags are driven by short bursts. John Arnold has the most negatives overall (21) and is flagged,
  but rhonda.denton has only 11 in total and is still flagged because 4 of them fall within 9 days.
""",
"model": """
**Observations – model**

* The linear model beats the mean baseline clearly: on the test set, **MAE 1.12 vs 1.44** and **R² 0.18 vs −0.15**.
  Results are consistent across 5-fold CV (mean R² 0.20, fold range 0.06–0.36) and the time-based split
  (train 2010 → test 2011, R² 0.25). Train R² is 0.27, so overfitting is small.
* **Message frequency is the dominant and significant driver** (β = +0.83 per SD, p = 0.015). The single-feature
  model gives test R² 0.19 with a slope of **+0.14 score per extra message**. That is almost exactly the average
  net sentiment per message (22.2% pos − 6.3% neg = 0.159). In other words, the score partly measures
  activity: more messages give more chances to score +1.
* **Reply ratio (+0.24, p = 0.03)** and **exclamation marks per message (+0.23, p = 0.03)** are also significant
  and positive. Conversational, expressive writing goes with higher sentiment.
* Length features (`avg_char_len`, `avg_word_count`, `total_words`) are not significant and have very large
  standard errors because they are strongly collinear (r ≈ 0.99). The model could drop them without losing accuracy.
* An R² of about 0.2 means message metadata explains only a fifth of the variation in sentiment. The rest
  depends on what is said, so this model is useful for explaining patterns, not for predicting individual scores.
  The residuals widen for high predicted values (heteroscedasticity), which comes from scores being bounded counts.
""",
"summary": """
**Summary of findings**

| | Result |
|---|---|
| Labels | 71.5% Neutral, 22.2% Positive, 6.3% Negative (RoBERTa transformer; VADER and TextBlob cross-checks) |
| Top 3 positive (overall) | johnny.palmer, sally.beck, lydia.delgado |
| Top 3 negative (overall) | rhonda.denton, john.arnold, don.baughman |
| Flight risks | **john.arnold, patti.thompson, rhonda.denton** |
| Watch list (3 negatives in 30 days) | bobette.riner, lydia.delgado |
| Model | Linear regression R² ≈ 0.2 (test / CV / time split); message frequency, reply ratio and exclamation use are significant |

**Recommendations:** reach out to the three flagged employees, starting with John Arnold (the most negatives
overall, and often in the bottom 3). Monitor the two employees on the watch list. Consider normalizing the score by
message volume (e.g. net sentiment per message) so that quiet employees are not ranked as "negative" just for
sending fewer e-mails.
""",
}

REPORT = {
"subtitle": "Unlabeled employee e-mail dataset (test.csv) – sentiment labeling, EDA, monthly scoring, ranking, "
            "flight-risk detection and linear-regression modeling. Code: Employee_Sentiment_Analysis.ipynb.",
"exec": [
    "2,191 e-mails from 10 employees (Jan 2010 – Dec 2011) were labeled with a RoBERTa transformer: 71.5% Neutral, "
    "22.2% Positive, 6.3% Negative.",
    "Overall top 3 positive employees: johnny.palmer (total 51), sally.beck (49), lydia.delgado (48). "
    "Overall top 3 negative: rhonda.denton (16), john.arnold (22), don.baughman (25). Monthly top-3 tables for all 24 months are in Section 4.",
    "Flight risks (≥ 4 negative messages within a rolling 30-day window): john.arnold, patti.thompson and rhonda.denton. "
    "bobette.riner and lydia.delgado peak at 3 and form a watch list.",
    "Sentiment is stable over time and slightly better in 2011. Replies carry the most emotion, and negative messages are the shortest.",
    "A linear regression on message metadata explains about 20% of the variance in monthly score (test R² 0.18, CV 0.20, "
    "time split 0.25) and clearly beats the baseline. Message volume is the main driver.",
],
"data": "The dataset has 2,191 rows and 4 string columns (Subject, body, date, from) with no missing values and no "
        "duplicates. There are 10 senders, each active in all 24 months. Dates are parsed as month/day/year. "
        "The bodies are raw Enron e-mails containing quoted-printable encoding artefacts, which are removed before modeling.",
"labeling": [
    "Cleaning: soft line breaks (=\\n) and QP codes (=20, =01, …) are removed, URLs and e-mail addresses are stripped, "
    "and whitespace is collapsed. The subject is prepended to the body because it often carries tone.",
    "Primary labeler: cardiffnlp/twitter-roberta-base-sentiment-latest, a RoBERTa transformer (PyTorch / Hugging Face) "
    "fine-tuned for 3-class sentiment. For every message the model outputs P(negative), P(neutral) and P(positive), "
    "and the label is the arg-max. The model runs in eval mode with no sampling, so labels are fully reproducible. "
    "Probabilities and confidence are stored next to the label in outputs/labeled_messages.csv.",
    "Why: a transformer understands context and negation (\"unfortunately … will not work\", \"not a problem\") "
    "that lexicon methods miss, and it runs on a CPU in a couple of minutes without an API key.",
    "Validation: two lexicon-based methods, VADER and TextBlob, labeled the same text independently. Both use a ±0.05 "
    "neutral band around a polarity score. Neither changes the final label.",
    "Agreement with the transformer: VADER 42.9% (Cohen's κ 0.17), TextBlob 52.2% (κ 0.24). VADER and TextBlob agree "
    "with each other 65.3% of the time (κ 0.36), and all three agree on 33.9% of messages. The lexicon methods label "
    "far more messages Positive (VADER 1,574, TextBlob 1,165, transformer 487) because they treat polite business "
    "wording as positive. Most disagreements are transformer-Neutral vs lexicon-Positive (1,059 with VADER, 710 with TextBlob).",
    "Opposite polarities (Positive vs Negative) are rare: 69 messages with VADER (3.1%) and 51 with TextBlob (2.3%). "
    "A manual review of those favours the transformer. Examples: \"Unfortunately, today is not going to work…\" (transformer: "
    "Negative, VADER: Positive) and \"it wouldn't hurt to have another guy\" (transformer: Positive, TextBlob: Negative). "
    "Mean transformer confidence is 0.81.",
],
"eda": [
    "Sentiment mix: Neutral 1,566 (71.5%), Positive 487 (22.2%), Negative 138 (6.3%).",
    "Time trend: there are 91–92 messages every month. Monthly net sentiment ranges from +6 (Feb 2010) to +22 (Oct 2011). "
    "2011 is slightly more positive than 2010 (23.3% vs 21.1% positive; 5.7% vs 6.9% negative).",
    "Anomaly: the e-mail content refers to 2000–2001 but the dates are 2010–2011, they are spread evenly across months, "
    "and about 31% of messages fall on weekends. The dates were probably re-sampled, so day-of-week effects should not be read as behaviour.",
    "Employees: johnny.palmer (29% positive) and sally.beck (27%) are the most positive. bobette.riner (8.8% negative) and "
    "john.arnold (8.2%) are the most negative. don.baughman and rhonda.denton are the most neutral (~78%).",
    "Length: negative messages are the shortest (median 24 words vs 30 neutral and 38 positive).",
    "Message type: replies are the most emotional (9.6% negative, 33.4% positive). Forwards are almost all neutral (1.4% negative).",
    "Data quality: 30 empty bodies, 100 bodies under 20 characters, 141 \"(No Subject)\", and one automated notice repeated 26 times. "
    "They are kept because they are real messages.",
],
"scoring": [
    "Each message scores +1 (Positive), −1 (Negative) or 0 (Neutral).",
    "Messages are grouped by employee and calendar month (pandas Period 'M'). The monthly score is the sum inside that "
    "month only, so it resets to 0 at the start of each month. A running total within the month is also kept for each message.",
    "Validation (assertions in the notebook): score = #positive − #negative, the last running total of each month equals "
    "the monthly score, the first message of each month starts a fresh total, and message counts add up to 2,191.",
    "Result: 240 employee-months. Scores range from −3 (patti.thompson, Oct 2010) to +8 (sally.beck, Aug 2010), "
    "the median is 1, and only 26 employee-months (11%) are negative.",
],
"ranking": [
    "For each month, the Task 3 scores are sorted. Top 3 positive: score descending, then employee name A→Z. "
    "Top 3 negative: score ascending, then name A→Z. Only employees active in that month are ranked "
    "(all 10 are active every month). The ordering and tie-breaks are verified by assertions.",
    "Because negative months are rare, the bottom-3 list often contains scores of 0 or above. It identifies the "
    "least positive employees of the month.",
    "Overall view: summing all monthly scores gives johnny.palmer (51), sally.beck (49) and lydia.delgado (48) as the most positive, "
    "and rhonda.denton (16), john.arnold (22) and don.baughman (25) as the most negative.",
    "Dec 2011 (latest month): positive – kayne.coulter (4), eric.bass (3), patti.thompson (3); "
    "negative – bobette.riner (0), don.baughman (1), john.arnold (1).",
],
"flight": [
    "Criterion: 4 or more Negative messages within any rolling 30-day window (30 calendar days inclusive, i.e. "
    "date_last − date_first ≤ 29 days). The window ignores calendar months and the score. Every message counts.",
    "Method: a two-pointer sliding window over each employee's sorted negative-message dates. It records the peak "
    "count, the peak window and the first date the threshold was reached. An independent brute-force check gives the same result.",
    "Flagged: john.arnold (peak window 2010-06-06 → 06-21), patti.thompson (2010-10-04 → 10-29), "
    "rhonda.denton (2010-12-24 → 2011-01-01). Rhonda Denton's window crosses a month and year boundary, "
    "so a calendar-month rule would have missed her.",
    "Sensitivity: a threshold of 3 would flag 5 employees (adding bobette.riner and lydia.delgado). A threshold of 5 flags none "
    "at 30 days. With 14 days, only 1 employee reaches 4.",
],
"model_setup": [
    "Target: the monthly sentiment score (Task 3). Unit: employee-month (240 rows).",
    "Features, chosen to avoid leakage (none uses the sentiment label): message count, average character length, "
    "average word count, total words, average subject length, reply ratio, forward ratio, exclamation marks and "
    "question marks per message, and weekend ratio.",
    "Model: StandardScaler + scikit-learn LinearRegression. Evaluation: 80/20 random split (seed 42), "
    "5-fold cross-validation, a time-based split (train 2010 → test 2011) and a mean-only baseline. "
    "Metrics: R², MAE, RMSE. p-values come from OLS on standardized features.",
],
"model_findings": [
    "The model beats the baseline on every split (test MAE 1.12 vs 1.44). R² is 0.18 on the test set, 0.20 in CV and "
    "0.25 on the time split, against 0.27 on training data, so the model generalizes with little overfitting.",
    "Message frequency is the strongest and statistically significant predictor (+0.83 per SD, p = 0.015). On its own it "
    "gives +0.14 score per message, which matches the dataset's average net sentiment per message (0.159). "
    "The monthly score therefore partly measures activity.",
    "Reply ratio (+0.24, p = 0.03) and exclamation marks (+0.23, p = 0.03) also raise the score: conversational, "
    "expressive e-mail is more positive. Question marks and weekend share have small negative, non-significant effects.",
    "The length features are collinear (r ≈ 0.99), so their individual coefficients are unstable and not significant.",
    "An R² of about 0.2 means most of the sentiment comes from content rather than metadata. The model is useful for "
    "explaining patterns and is a weak forecaster of individual scores. The residuals widen for higher predictions.",
],
"recommendations": [
    "Contact the three flight-risk employees (john.arnold, patti.thompson, rhonda.denton) through their managers or HR. "
    "Prioritise john.arnold, who has the most negative messages overall (21) and often appears in the monthly bottom 3.",
    "Keep bobette.riner and lydia.delgado on a watch list, since each reached 3 negatives within 30 days.",
    "Report a volume-normalized score (net sentiment per message) next to the raw score, so that low-volume employees "
    "are not ranked as negative only because they send fewer e-mails.",
    "Run the rolling-window check continuously (e.g. weekly) instead of once a month, because bursts that cross month ends would otherwise be missed.",
    "Look at the content of short reply chains, where most negative sentiment appears.",
],
"limitations": [
    "Labels come from a model with no human-annotated ground truth. The transformer was trained on social-media text, "
    "and some business wording may be mislabeled. Spot checks and the VADER and TextBlob comparisons support the labels but do not prove them.",
    "Quoted or forwarded text inside a body is attributed to the sender.",
    "Dates appear to be synthetic (evenly spread, ~31% on weekends), so time-of-week findings are not interpreted.",
    "The regression has only 240 rows, and the target is a bounded count. A Poisson or ordinal model, or text "
    "embeddings as features, could perform better but are outside the linear-regression scope.",
],
"repro": "pip install -r requirements.txt (the CPU build of torch is enough), then run Employee_Sentiment_Analysis.ipynb "
         "from top to bottom. Transformer labels are cached in outputs/labeled_messages.csv; delete that file to "
         "recompute them (about 5–10 minutes on a CPU, including the model download). "
         "Then run python src/build_report.py to rebuild this document. The random seed is 42 for all splits.",
}
