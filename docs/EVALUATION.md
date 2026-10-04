# Search evaluation

> **Status: methodology and tooling are ready; the numbers must be produced on your dataset.**
> Run the steps below and paste the resulting tables into this file. Do not submit made-up results.

## Method

1. Build the dataset with `scripts/download_dataset.py` and index it.
2. `python eval/run_eval.py --k 10 --out docs/EVAL_RESULTS.md` runs the 14 queries in `eval/queries.json`
   (the five from the brief, plus visual, text-driven and a negative-control query).
3. **Automatic labels:** a result is "relevant" if its Wikimedia Commons metadata (title, categories,
   description from `manifest.json`) matches the query's regex and its type is expected. The search system
   never reads this metadata, so the labels are independent of the system under test, but they are noisy
   (missing descriptions → false negatives, loose categories → false positives).
4. **Manual review (source of truth):** open each query's top-10 in the UI, fill the *Manual verdict* column
   (✔ relevant / ✘ not / ~ partially) and the notes. Report manual P@5 / P@10.
5. **Negative control** (Q14): an unrelated query should return nothing or very little; this checks the score floors.
6. If results look too noisy or too empty, tune `CLIP_MIN_SCORE` / `TEXT_MIN_SCORE` in `.env`, re-run, and record the change.

## Queries (what the user wants, what should appear)

| ID | Query | User intent | Expected assets |
|---|---|---|---|
| Q01 | A woman standing with a cat | photos of a woman with a cat | `image/woman-cat/*` |
| Q02 | Images showing a modern living room | contemporary living-room photos | `image/living-room/*` |
| Q03 | Videos containing construction activity | construction footage | `video/construction/*` |
| Q04 | Brochures related to residential projects | housing/real-estate PDFs | `pdf/real-estate/*`, `pdf/brochure/*` |
| Q05 | Customer testimonial videos | people speaking to camera | `video/interview/*` |
| Q06 | construction workers wearing helmets on site | worker photos | `image/construction/*` |
| Q07 | modern kitchen with island | kitchen interiors | `image/kitchen/*` |
| Q08 | apartment building exterior | residential buildings | `image/building-exterior/*` |
| Q09 | how to operate or install the product instructions | manuals (text retrieval) | `pdf/manual/*` |
| Q10 | annual financial results revenue | annual reports (keyword + semantic) | `pdf/report/*` |
| Q11 | dog running in a park | dog photos | `image/dogs/*` |
| Q12 | wild animals in nature | wildlife video | `video/nature/*` |
| Q13 | timelapse of city traffic at night | city traffic video | `video/city/*` |
| Q14 | spaceship on Mars | **nothing** (negative control) | none |

## Round 1 (baseline) - findings and fixes

Observed in the first manual test round (15 cases, three modalities + negative query + filter test):

* Correct top results for the large majority of queries (e.g. *woman with a cat* → "Woman with a Cat" paintings; *excavator* → the Demolition/excavator videos; *laboratory regulations* → CLIA brochures); video match timestamps and PDF pages worked; latency 33–46 ms.
* **Defects found:** (1) the negative query returned 39 results with two at "100 %" because scores were relative to the top hit; (2) cutoffs did not cut - each video query returned 20–24 results and each PDF query 47–50, so the tail was unrelated (e.g. a music video at 85 % for *excavator*); (3) for *construction* the first seven results were all PDFs (multi-signal score stacking) despite construction images/videos in the dataset; (4) very long PDFs recurred in unrelated queries; (5) several PDF hits showed only a page number, no extracted text.
* **Fixes:** absolute match strength; absolute floor + relative margin per signal; max-based fusion; facet counts by type; extracted text shown for PDF hits; `scripts/calibrate.py` to set floors from data.

## Round 2 (after fixes)

Re-run `python scripts/calibrate.py`, put its values in `.env`, then `python eval/make_report.py` and `python eval/run_eval.py`; paste the outcome below and compare with round 1 (result counts per query, negative-query result count, type mix for *construction*).

## Results

_Paste the summary table and per-query verdicts from `docs/EVAL_RESULTS.md` here._

| Metric | Auto-labelled | Manual |
|---|---|---|
| Mean P@5 | | |
| Mean P@10 | | |
| MRR | | |

## Failure analysis

_Fill in from what you observe. Likely weak spots to check specifically:_

* **Q05 testimonials** - "testimonial" is an abstract concept; CLIP sees "person facing camera". Check whether `ENABLE_ASR=true` improves it.
* **Q01 counting/relations** - CLIP matches "woman" and "cat" separately; verify images with a cat but no woman are not ranked above true matches.
* **Q04/Q09 PDFs** - depends on text-bearing PDFs; scanned brochures only match visually.
* **Q14 negative control** - if it returns many results, raise `CLIP_MIN_SCORE`.
* **Videos** - events between sampled frames are missed.
