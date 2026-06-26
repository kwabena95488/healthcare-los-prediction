# Dataset

This project uses a **public healthcare-analytics dataset** of hospital admission
records. The data is **synthetic / de-identified** — it contains **no real
patient PII**. Every "patient" and "hospital" field is an anonymized code.

## Provenance

- **Source:** Analytics Vidhya — *Janatahack: Healthcare Analytics II* hackathon
  (publicly distributed practice dataset).
- **Task:** predict each admission's **Length of Stay** (`Stay`), an 11-category
  bucket (`0-10`, `11-20`, … `91-100`, `More than 100 Days`).
- **Size:** ~318,438 training rows × 18 columns.
- **Use:** released for educational / competition use. Treat it as a public
  practice dataset; check the original source for the authoritative terms before
  any commercial use.

## What is committed vs. downloaded

| Path | Committed? | Notes |
|------|-----------|-------|
| `data/sample/train_sample.csv` | ✅ yes (~83 KB, 1,000 rows) | Tiny stratified sample so notebooks, `scripts/build_demo.py`, and `app.py` run with **zero download**. All 11 classes present. |
| `data/raw/train.csv` (~26 MB) | ❌ gitignored | Full training set — download separately (below). |
| `data/raw/test.csv` (~10 MB) | ❌ gitignored | Full unlabeled test set. |
| `data/raw/column-descriptions.md` | ✅ yes | Column dictionary. |

The full CSVs are **excluded from version control** (`*.csv` in `.gitignore`)
to keep the repo lightweight and avoid committing bulk data.

## How to get the full data

1. Download the *Healthcare Analytics II* train/test CSVs from the Analytics
   Vidhya datahack page (free account required).
2. Place them at `data/raw/train.csv` and `data/raw/test.csv`.

Everything in this repo runs on the committed **sample** out of the box; the
full download is only needed to reproduce the headline metrics at full scale.

## Columns

See [`raw/column-descriptions.md`](raw/column-descriptions.md). Key fields:
`Hospital_*` (facility codes), `Department`, `Ward_Type`, `Bed Grade`,
`Type of Admission`, `Severity of Illness`, `Age` (decade bucket),
`Admission_Deposit`, `Visitors with Patient`, and the target `Stay`.
`case_id` and `patientid` are anonymized identifiers and are dropped before
modeling.
