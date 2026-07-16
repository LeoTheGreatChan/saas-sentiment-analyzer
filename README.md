# Uber Review Sentiment Pipeline
### Automated VOC Intelligence · End-to-End NLP System

**Live Demo → [saas-sentiment-analyzer.streamlit.app](https://saas-sentiment-analyzer.streamlit.app/)**  
**Source Code → [github.com/LeoTheGreatChan/saas-sentiment-analyzer](https://github.com/LeoTheGreatChan/saas-sentiment-analyzer)**

---

## The Problem

Product teams at consumer apps receive thousands of app store reviews every week. The signal is there — pricing complaints, driver behaviour issues, app crashes — but it's buried in noise. Manual triage is slow, inconsistent, and misses critical issues until they've already compounded.

This project automates that triage entirely.

---

## What It Does

A fully automated pipeline that fetches live Uber app reviews from Google Play, scores each one using a fine-tuned BERT model, routes critical reviews to a Gmail alert channel, and surfaces everything in a live executive dashboard — with zero manual intervention required between runs.

**One click in n8n. Everything else is automatic.**

From the latest run:
- **379 reviews** processed across multiple app versions
- **+0.35 average sentiment score** — net positive but with significant negative clusters
- **123 critical alerts** automatically identified and routed
- **68 Gmail alerts** fired for the highest-severity reviews

---

## Pipeline Architecture

![n8n pipeline — all nodes green after a successful run](n8n_pipeline_overview_png.jpg)

The pipeline runs as an n8n workflow with six stages, processing 200 reviews per run with all item counts visible between nodes: 1 → 200 → 200 → 68 (critical) + 132 (normal) → 200 written to Sheets.

**① Trigger — Manual (upgradeable to Schedule)**  
A manual trigger node fires the workflow on demand. Upgrading to a scheduled run — e.g. every Sunday at 9am — requires replacing this single node with a Schedule Trigger. Every downstream node stays identical.

**② Ingest — google-play-scraper**  
A JavaScript Code node fetches the latest Uber reviews from Google Play using `google-play-scraper` (no API key required). Returns 200 structured items per run with consistent fields: review text, app version, timestamp, and community likes count.

**③ Score — DistilBERT via Flask**  
An HTTP Request node POSTs each review to a local Flask endpoint (`POST: http://127.0.0.1:5000/score`) wrapping a DistilBERT model fine-tuned on SST-2. Each review returns a sentiment label (Positive/Negative) and a confidence score from −1.0 to +1.0. The scorer runs as a persistent service — the model loads once and stays warm across all 200 calls.

**④ Triage — IF node with compound alert logic**  
![IF node showing two conditions linked by OR](n8n_if_node_png.jpg)

An IF node applies the alert threshold using two conditions linked by OR:
- `{{ $json.score }} is less than -0.6` — very negative sentiment
- `{{ $json.label=='Negative' && $('google-play-scraper').item.json.thumbsUpCount>0 }}` — negative reviews with community agreement

From the latest run: **68 items to the true branch** (critical), **132 items to the false branch** (normal).

**⑤ Alert — Gmail**  
A Gmail node sends an automated email for each of the 68 critical reviews, including full review text, version number, sentiment score, and timestamp. No manual monitoring required — critical feedback surfaces in the inbox automatically.

**⑥ Write — Google Sheets (deduplicated)**  
![Google Sheets node showing all 7 field mappings](n8n_sheets_node_png.jpg)

A Google Sheets node writes all 200 scored reviews using `Append or Update Row` with `ReviewId` as the deduplication key. Re-running the pipeline on consecutive days never produces duplicate rows — the operation is fully idempotent. Field mappings pull from two upstream nodes: review metadata from `google-play-scraper`, sentiment label and score from `HTTP Request for Sentiment Score`.

---

## Live Dashboard

![Product Insights Dashboard showing 379 reviews, +0.35 avg sentiment, 123 critical alerts](dashboard_overview_png.jpg)

A Streamlit dashboard reads live from Google Sheets via OAuth2, with a 5-minute cache TTL. No redeployment needed when new data arrives — run the n8n workflow and the dashboard updates automatically on the next page load.

**Four headline metric cards** update dynamically when the version filter changes:
- **379** total reviews in current selection
- **+0.35** average sentiment score — colour-coded green for positive
- **123** critical alerts — colour-coded orange
- **Multiple** active versions (changes to a specific version when filtered)

**Performance Trends tab** shows two charts side by side: sentiment by app version (colour-coded bar chart — red for negative, green for positive) and sentiment trend over time (area line chart). The version bar chart clearly shows older releases (4.497–4.518) clustering negative while recent releases (4.630+) trend strongly positive.

**Critical Alerts tab** surfaces the highest-priority reviews ranked by severity:

![Critical Alerts tab showing 123 reviews with left red border styling](dashboard_alerts_png.jpg)

123 reviews with very negative sentiment or high community agreement, each shown with score (colour-coded red), version, likes count, and full review text. The top alert — a French-language review about locked account and unresponsive support — scored −0.971 with 185 community likes, making it the highest-priority issue in the dataset.

**Review Explorer tab** shows the full scored dataset with date, version, review text, sentiment label, score, and likes — filterable by version and exportable as CSV.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | n8n (self-hosted, local) |
| Review ingestion | google-play-scraper (JS, no API key) |
| Sentiment scoring | DistilBERT (distilbert-base-uncased-finetuned-sst-2-english) |
| Scorer API | Python · Flask |
| Alert delivery | Gmail (via n8n OAuth node) |
| Data store | Google Sheets (append + dedup via ReviewId) |
| Dashboard | Python · Streamlit |
| Auth | Google OAuth2 via Streamlit Secrets |
| Deployment | Streamlit Cloud (production) |

---

## Key Design Decisions

**Why n8n over a Python script?**  
n8n makes the pipeline visual, auditable, and modifiable without code changes. The manual→schedule trigger swap is a single node replacement. Non-technical stakeholders can read the canvas and understand what the pipeline does without touching Python.

**Why Google Sheets over a database?**  
For a portfolio demo, Sheets provides a free, inspectable, shareable data store with zero infrastructure overhead. The architecture is identical whether the destination is Sheets, BigQuery, or Postgres — only the write node changes.

**Why a persistent Flask scorer instead of scoring inside n8n?**  
n8n's Code node would reload the DistilBERT model on every execution. A persistent Flask service loads the model once and keeps it warm — scoring 200 reviews takes seconds rather than minutes.

**Why deduplication by ReviewId?**  
Running the pipeline on consecutive days fetches overlapping review windows. Without deduplication, the same review appears multiple times and skews version-level sentiment averages. ReviewId as the match key makes the pipeline idempotent — safe to re-run at any time.

**Why OAuth2 via Streamlit Secrets instead of a service account?**  
The Google Cloud organisation policy in this environment blocks service account key creation. The OAuth2 + refresh token approach stored in Streamlit Secrets is equally secure, works identically in local and production environments, and keeps all credentials out of the codebase.

---

## Extension Opportunities

- **Schedule trigger** — swap Manual Trigger for Schedule Trigger in n8n (one node, no code changes)
- **Multi-app** — add a second Code node branch for competitor apps (e.g. Lyft) and compare sentiment trends side by side
- **Slack alerts** — replace or supplement Gmail node with Slack webhook for team-wide visibility
- **Model upgrade** — swap DistilBERT endpoint for a domain-specific fine-tuned model — Flask wrapper stays identical
- **BigQuery sink** — replace Google Sheets node with BigQuery node for enterprise-scale storage

---

*Built by Leo Chan · July 2026*