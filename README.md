# Employee Sentiment Analysis

This project labels 2,191 unlabeled employee e-mails (10 employees, Jan 2010 – Dec 2011) as Positive, Negative or
Neutral. It then explores the data, scores and ranks employees month by month, flags flight risks and models
monthly sentiment with linear regression.

* **Main notebook:** [`Employee_Sentiment_Analysis.ipynb`](Employee_Sentiment_Analysis.ipynb) runs every task, with commentary and observations
* **Report:** [`report/Final_Report.docx`](report/Final_Report.docx)
* **Figures:** [`visualization/`](visualization)
* **Tables:** [`outputs/`](outputs) – `labeled_messages.csv` is the dataset with the added `sentiment` column

## Summary of results

### Sentiment labels
The final labels come from a RoBERTa transformer (`cardiffnlp/twitter-roberta-base-sentiment-latest`, PyTorch). Each label is
the model's most likely class. VADER and TextBlob, two lexicon-based methods, are used only as cross-checks:

| Method | Positive | Neutral | Negative | Agreement with transformer |
|---|---|---|---|---|
| **Transformer (final)** | 487 (22.2%) | 1,566 (71.5%) | 138 (6.3%) | – |
| VADER | 1,574 | 444 | 173 | 42.9% (κ 0.17) |
| TextBlob | 1,165 | 758 | 268 | 52.2% (κ 0.24) |

The lexicon methods count polite business wording ("please", "thanks") as positive. Most disagreements are
therefore Neutral vs Positive. Opposite-polarity disagreements are rare: 3.1% with VADER and 2.3% with TextBlob.

### Top three positive and negative employees
Monthly score = Σ(+1 Positive, −1 Negative, 0 Neutral), reset every calendar month.

**Overall (sum of the 24 monthly scores)**

| Rank | Top positive | Total | Top negative | Total |
|---|---|---|---|---|
| 1 | johnny.palmer@enron.com | 51 | rhonda.denton@enron.com | 16 |
| 2 | sally.beck@enron.com | 49 | john.arnold@enron.com | 22 |
| 3 | lydia.delgado@enron.com | 48 | don.baughman@enron.com | 25 |

**Latest month (Dec 2011)**: positive – kayne.coulter (4), eric.bass (3), patti.thompson (3);
negative – bobette.riner (0), don.baughman (1), john.arnold (1).
Lists for every month are in `outputs/top3_positive_by_month.csv` and `outputs/top3_negative_by_month.csv`.
Ties are sorted by score first, then alphabetically.

### Flight risks
Rule: 4 or more Negative messages within any rolling 30-day window, regardless of calendar months.

| Employee | Negatives in peak 30-day window | Window |
|---|---|---|
| **john.arnold@enron.com** | 4 | 2010-06-06 → 2010-06-21 |
| **patti.thompson@enron.com** | 4 | 2010-10-04 → 2010-10-29 |
| **rhonda.denton@enron.com** | 4 | 2010-12-24 → 2011-01-01 |

Watch list (peak of 3): bobette.riner, lydia.delgado.

### Predictive model
A linear regression predicts the monthly score from message metadata (frequency, length, reply/forward ratios,
punctuation, weekend share).
Test R² is 0.18, 5-fold CV R² is 0.20 and the time-split R² (2010 → 2011) is 0.25. Test MAE is 1.12, against 1.44 for the mean baseline.
Message count is the main significant driver (+0.14 score per message). Reply ratio and exclamation use also raise the score.

## Key insights
* Sentiment is stable over time and slightly better in 2011 (negative share fell from 6.9% to 5.7%).
* Replies carry the most emotion (9.6% negative, 33.4% positive). Forwards are almost all neutral.
* Negative messages are short (median 24 words vs 38 for positive).
* The raw monthly score partly reflects activity: employees who send more messages tend to score higher.
* A rolling 30-day window catches bursts that a calendar-month view misses. Rhonda Denton's flag spans New Year.
* The dates look synthetic (evenly spread, ~31% on weekends), so weekday effects are not interpreted.

## Recommendations
1. Contact the three flight-risk employees first, starting with John Arnold (the most negatives overall, 21).
2. Monitor bobette.riner and lydia.delgado, who are one negative message below the threshold.
3. Report net sentiment per message next to the raw score, so that quieter employees are not penalised.
4. Run the rolling-window check weekly rather than monthly.

## Project structure
```
├── Employee_Sentiment_Analysis.ipynb   main notebook (all tasks, commentary, observations)
├── data/test.csv                       raw dataset
├── src/
│   ├── sentiment.py        text cleaning, transformer labeler, VADER and TextBlob cross-checks
│   ├── analysis.py         monthly scores, rankings, rolling flight risk, features, OLS stats
│   ├── observations.py     narrative text used in the notebook and report
│   ├── build_notebook.py   regenerates the notebook
│   └── build_report.py     regenerates report/Final_Report.docx
├── outputs/                CSV tables (labels, scores, rankings, flight risk, model metrics)
├── visualization/          PNG figures (EDA, rankings, flight risk, model performance)
├── report/Final_Report.docx
└── requirements.txt
```

## Reproduce
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace Employee_Sentiment_Analysis.ipynb
cd src && python build_report.py
```
Labels are cached in `outputs/labeled_messages.csv`. Delete that file to re-run the transformer, which takes a few minutes on a CPU.
The model runs in eval mode and all splits use seed 42, so the results are deterministic.
