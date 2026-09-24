# document-extraction-audit
 Rule-based document extraction with a Gradient Boosting quality classifier.
# Document Extraction & Quality-Audit System

A rule-based document field-extraction pipeline paired with a Gradient Boosting
classifier that predicts whether an extraction is trustworthy — automatically
flagging likely errors for human review.

## The problem

At my current role (Data Operations Officer, InsideMaps Inc.), I manually
review extracted data against source documents, checking for accuracy and
flagging discrepancies. This project automates that process: instead of a
person eyeballing every record, a classifier learns which extractions are
likely to be wrong, so review effort focuses on the records that actually
need it.

## Approach

1. **Field extraction (regex)** — pulls vendor name, date, total amount, and
   currency from receipt/invoice text using pattern matching.
2. **Quality classification (Gradient Boosting)** — trained to predict, from
   signals available *at extraction time* (which fields were found, text
   length), whether a given extraction is likely correct. This is the same
   algorithm family used in my published thesis
   ("Sentiment Analysis of Tweets on Covid Vaccine: A Boosting Based Machine
   Learning Solution", MIET 2022), applied here to a new problem.
3. **Audit layer** — cross-checks extracted fields against a small reference
   vendor list and folds in the classifier's prediction to produce a final
   `clean` / `flagged` verdict with a reason.

**Why classical ML instead of an LLM API:** this was a deliberate scoping
decision to keep the system free to run, dependency-free, and fully
reproducible on any machine — no API keys, no rate limits, no cost. An
LLM-based extraction layer is a natural next iteration (see Limitations).

## Dataset

300 synthetic receipts, generated programmatically with known ground-truth
labels: 150 clean, 150 with realistic OCR-style corruption (digit confusion,
garbled keywords like "TOTAL" → "T0TAL"). Generating labeled data this way
made accuracy genuinely measurable without needing a labeled real-world
dataset.

## Results

Measured on a held-out test set (25% of data, unseen during training):

| Metric | Rule-based extraction | Quality classifier |
|---|---|---|
| Accuracy | 70.7% | 73.3% |
| Precision | — | 73.5% |
| Recall | — | 96.2% |
| F1 | — | 83.3% |

The classifier was deliberately optimized toward **recall over precision** —
in a QA context, missing a genuinely bad extraction is costlier than
over-flagging a good one for a second look.

## How it works

```
receipt text → regex extraction → feature signals → Gradient Boosting
    → predicted reliability → audit layer → clean / flagged + reason
```

Run `cell1_data.py` through `cell4_audit.py` in order (designed for Google
Colab, no installation required — uses only `scikit-learn` and `pandas`,
both preinstalled).

## Limitations

Built and evaluated on synthetic data rather than real scanned documents,
by design, to keep the project free and dependency-free. A production
version would extend this with real OCR input (e.g. Tesseract) and
LLM-based extraction for messier real-world text, which I'm continuing to
build toward as a next step.

## Stack

Python · scikit-learn (Gradient Boosting) · Regex · Pandas · Google Colab
