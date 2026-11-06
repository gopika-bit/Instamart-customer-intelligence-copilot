# Instamart Customer Intelligence & Support Copilot

## Overview

This repository implements the four-part, offline capstone using fictional synthetic data. It provides a data-cleaning pipeline and SQLite database, customer segmentation and lapse prediction, privacy-safe ticket triage, policy retrieval, and a tool-using refund agent. All currency values are INR. The brand is used only as a fictional scenario; the generated data and policies are not real Instamart information.

The two supplied files, `generate_data.py` and `mock_llm.py`, are included at the repository root. All model-like calls use the local `mock_llm.py`; no API key, paid service, network request, or hosted model is used. Fixed random seeds are used wherever randomness appears.

Repository map:

- `src/part_a.py` — audit, clean, validate, persist, run SQL and pandas analysis.
- `src/part_b.py` — RFM, K-means, PCA, lapse feature engineering, model tuning and retention list.
- `src/part_c.py` — redaction, guardrails, retries, classification validation, token budget and routing.
- `src/part_d.py` — TF-IDF policy retrieval, policy evaluation, refund tools, ReAct loop and checkpoint/resume.
- `run_all.py` — runs Parts A–D in dependency order.
- `sql/queries.sql` — seven executable SQLite queries.
- `outputs/` — generated result files, metrics, model, ticket results, agent runs and refund ledger.
- `state/` — resumable agent checkpoints.

## Reproduce

Run these commands from the repository root. The examples below use a POSIX shell; on Windows activate with `.venv\\Scripts\\activate` instead of step 2.

1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `python -m pip install --upgrade pip`
4. `python -m pip install -r requirements.txt`
5. `python generate_data.py`
6. `python run_all.py`

Step 5 must print exactly `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. Part A prints `ALL CHECKPOINTS PASSED`. The pipeline regenerates raw data, clean CSVs, `data/instamart.db`, models, analysis outputs, ticket results, policy evaluations, agent runs and state checkpoints. All scripts use paths relative to this repository, so running from its root is important. To run a single part after its prerequisites exist, use `python -m src.part_a`, `python -m src.part_b`, `python -m src.part_c` or `python -m src.part_d`.

## Part A

### Raw audit

Raw table shapes and inferred pandas dtypes when loaded with `keep_default_na=False`:

| Table | Shape | Column dtypes |
|---|---:|---|
| `orders.csv` | 13,084 × 9 | `order_id` object; `customer_id` object; `product_id` object; `quantity` int64; `order_ts` object; `status` object; `discount_pct` int64; `order_amount` object; `delivery_minutes` object |
| `customers.csv` | 1,200 × 5 | `customer_id` object; `city` object; `signup_date` object; `age` object; `gender` object |
| `products.csv` | 40 × 4 | `product_id` object; `product_name` object; `category` object; `price_inr` int64 |

Per-column blank counts and percentages (blank strings are counted as missing for this audit; nonnumeric text such as `N/A` is counted separately):

| Table | Column | Blank count | Blank % |
|---|---|---:|---:|
| Orders | order_id | 0 | 0.00% |
| Orders | customer_id | 0 | 0.00% |
| Orders | product_id | 0 | 0.00% |
| Orders | quantity | 0 | 0.00% |
| Orders | order_ts | 0 | 0.00% |
| Orders | status | 0 | 0.00% |
| Orders | discount_pct | 0 | 0.00% |
| Orders | order_amount | 0 | 0.00% |
| Orders | delivery_minutes | 2,541 | 19.42% |
| Customers | customer_id | 0 | 0.00% |
| Customers | city | 0 | 0.00% |
| Customers | signup_date | 0 | 0.00% |
| Customers | age | 61 | 5.08% |
| Customers | gender | 30 | 2.50% |
| Products | product_id | 0 | 0.00% |
| Products | product_name | 0 | 0.00% |
| Products | category | 0 | 0.00% |
| Products | price_inr | 0 | 0.00% |

Specific raw audit counts:

| Audit item | Result |
|---|---:|
| Raw order rows | 13,084 |
| Exact duplicate rows | 119 |
| Rows with status `pending` | 656 |
| Distinct `order_id` values | 12,309 |
| Non-numeric `order_amount` entries | 120 |
| Negative `delivery_minutes` values | 106 |
| Blank `delivery_minutes` values | 2,541 |
| Blank customer ages | 61 |
| Ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank raw gender spellings | 10 |
| Blank genders | 30 |

### Cleaning rules and validation

1. **Customers:** trimmed city strings and applied title case; **476 city values** changed and the 24 raw spellings reduced to six cities. Gender spellings were mapped to `male`, `female`, or `unknown`; **950 gender values** were standardized and 30 blank genders become `unknown`. The six impossible ages (outside 10–100) were set missing. The 61 blank plus six impossible ages (**67 ages**) were filled using the median of that customer's city with `groupby('city').transform('median')`; all 1,200 ages are now present and fall between 18 and 60. `signup_date` is parsed as a datetime.
2. **Orders:** `order_ts` was parsed as datetime. Dropping exact duplicates removed 119 rows (13,084 → 12,965). Sorting each order by timestamp and keeping the latest row per `order_id` removed 656 superseded records (12,965 → 12,309), leaving no `pending` status. Of the raw 120 nonnumeric amount entries, 112 remained after deduplication and were filled using `round(price_inr × quantity × (1 − discount_pct / 100), 2)`. After deduplication, 105 negative delivery values were set to missing. All 774 missing completed-order delivery values after those steps were imputed using the median of the product category among completed orders. Cancelled orders were not imputed: all 1,199 retain missing delivery times because they were never delivered.
3. **Persistence:** the clean tables are written to `data/clean/` and to `data/instamart.db`. The database has `customers`, `products`, and `orders` tables; `order_ts` is stored as `YYYY-MM-DD HH:MM:SS` text. The validation script checks at least ten reference conditions and prints `ALL CHECKPOINTS PASSED` only if they all pass.

Cleaned customer counts:

| City | Customers |
|---|---:|
| Pune | 227 |
| Delhi | 210 |
| Bengaluru | 197 |
| Chennai | 191 |
| Hyderabad | 191 |
| Mumbai | 184 |

| Gender | Customers |
|---|---:|
| female | 610 |
| male | 560 |
| unknown | 30 |

Clean orders contain 12,309 rows: 11,110 completed and 1,199 cancelled. Completed orders have zero missing delivery times; cancelled orders have 1,199 missing delivery times. Completed-order revenue is **INR 2,998,668.80**.

### SQL analysis

All seven queries are saved in `sql/queries.sql` and executed with `pandas.read_sql_query` against the SQLite database.

| Query | Technique / question | Result |
|---|---|---|
| Q1 | `WHERE` completed orders with amount > INR 500, then `HAVING` customer total > INR 8,000 | 30 customers |
| Q2 | Three-table join: revenue by city and product category | 30 cells; total INR 2,998,668.80; largest cell Bengaluru / Electronics = INR 233,909.45 |
| Q3 | Scalar subquery: products above average catalogue price | 11 products |
| Q4 | `LEFT JOIN`: customers without any completed order | 25 customers |
| Q5 | Two CTEs: customer completed spend and average of those customer totals | 550 customers above average; average completed spend per ordering customer INR 2,552.06 |
| Q6 | `RANK`, `DENSE_RANK`, `ROW_NUMBER` partitioned by city | 1,175 customers with completed spend; 8 rows have `RANK()` different from `ROW_NUMBER()` |
| Q7 | Daily completed revenue and cumulative window `SUM` | 183 daily rows; final cumulative total INR 2,998,668.80 |

Top customer per city from Q6:

| City | Customer | Completed spend |
|---|---|---:|
| Bengaluru | C0082 | INR 12,856.20 |
| Chennai | C0930 | INR 10,151.40 |
| Delhi | C0291 | INR 13,742.60 |
| Hyderabad | C0743 | INR 12,730.05 |
| Mumbai | C0693 | INR 14,766.55 |
| Pune | C0426 | INR 10,377.10 |

`RANK()` can leave gaps after ties while `ROW_NUMBER()` always assigns a unique sequential number, so tied spend values can make their results differ.

### pandas cross-check and exploratory analysis

A `pivot_table(..., margins=True)` cross-check returns a grand total of **INR 2,998,668.80**, equal to SQL Q2. Completed-order amount by city:

| City | Sum | Mean | Count |
|---|---:|---:|---:|
| Bengaluru | INR 523,643.40 | INR 276.47 | 1,894 |
| Chennai | INR 463,629.15 | INR 272.72 | 1,700 |
| Delhi | INR 502,829.80 | INR 243.62 | 2,064 |
| Hyderabad | INR 528,589.90 | INR 284.95 | 1,855 |
| Mumbai | INR 479,058.50 | INR 297.00 | 1,613 |
| Pune | INR 500,918.05 | INR 252.48 | 1,984 |

| Completed `order_amount` statistic | Value |
|---|---:|
| Mean | INR 269.91 |
| Median | INR 135.00 |
| Skewness | 4.61 |
| Q1 | INR 64.80 |
| Q3 | INR 297.00 |
| IQR | INR 232.20 |
| Lower 1.5×IQR fence | INR -283.50 |
| Upper 1.5×IQR fence | INR 645.30 |
| Outliers below lower fence | 0 |
| Outliers above upper fence | 897 |

The data describes synthetic completed grocery and small-electronics orders, their amounts, discounts, delivery times and customer locations. The amount distribution is strongly right-skewed: the mean is INR 269.91, above the INR 135.00 median, with skewness 4.61. The IQR rule identifies 897 high-value outliers above INR 645.30 and none below the lower fence; these may reflect larger quantities or expensive electronics rather than errors by themselves. The largest absolute correlation is between order amount and discount percentage (about −0.095); amount and quantity is about +0.087. These are weak associations, and a correlation of this size does not establish causation.

## Part B

### RFM and segmentation

RFM uses completed orders through the 2025-09-30 window with snapshot `2025-10-01 00:00:00`. The table contains 1,175 customers with at least one completed order.

| RFM measure | Result |
|---|---:|
| Recency median / min / max | 19 / 0 / 180 days |
| Frequency mean / max | 9.46 / 33 orders |
| Monetary mean / max | INR 2,552.06 / INR 14,766.55 |

K-means used `StandardScaler` and `KMeans(n_clusters=K, n_init=10, random_state=42)` for each K. We selected **K = 4** because it has the strongest silhouette among the tested values and gives four actionable profiles.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.481 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

Inertia measures within-cluster squared distance, so adding clusters can only keep inertia the same or lower it; K=8 therefore naturally beats K=2 on inertia. That alone cannot select K because many tiny clusters can reduce inertia without improving separation or producing useful business groups; silhouette and interpretability are also needed.

Segment profiles and actions:

| Segment ID | Size | Mean recency (days) | Mean frequency | Mean monetary | Business name | Recommended action |
|---:|---:|---:|---:|---:|---|---|
| 0 | 436 | 28.86 | 3.77 | INR 1,057.62 | Everyday Customers | Use category-based cross-sell and replenishment reminders. |
| 1 | 138 | 24.12 | 6.77 | INR 7,129.70 | Premium High-Value — **Premium** | Offer relevant premium bundles and proactive service. |
| 2 | 419 | 8.77 | 19.40 | INR 3,410.02 | Frequent Loyalists | Reward repeat orders and protect convenience. |
| 3 | 182 | 109.04 | 2.20 | INR 685.99 | At-Risk / Cooling | Test win-back offers and ask about service issues. |

Cluster IDs are arbitrary labels, not a ranking. Segment 1 is marked Premium because it has the highest mean monetary value. PCA on the scaled RFM matrix explains **63.43%** and **21.80%** of variance in its first two components (total **85.23%**). Results are written to `outputs/customer_segments.csv` with the required columns.

### Lapse prediction problem and leakage controls

The cohort contains customers with at least one completed order before 2025-09-01. A customer is labelled lapsed if they have no completed order from 2025-09-01 through 2025-09-30. The cohort has **1,156 customers**, including **443 lapsed (38.32%)**. The stratified split is **924 train / 232 test**.

Features are built only from orders before 2025-09-01 plus the clean customer table: recency, frequency, monetary, average discount, cancellation rate, average delivery time, favourite product category, city, age, gender and tenure. No order on or after 2025-09-01 enters the feature matrix, and the Part B RFM segment labels are not features. Numeric imputers/scalers and categorical encoders live inside a scikit-learn `Pipeline` and are fitted on training rows only; they are only applied to the test rows. The feature matrix is asserted to contain no missing values. No encoder, scaler or resampler saw test rows.

The training-fitted scaler's mean for frequency is **8.115**, while the full-cohort mean is **8.029**; they differ because the training rows are a stratified subset, not the entire cohort.

### Tuning and test metrics

Baseline: always predicting “not lapsed” gives **61.64% accuracy** and **0.00 F1** for the lapsed class.

| Model | Grid | Best parameters | Best CV F1 | Test confusion matrix `[[TN, FP], [FN, TP]]` | Lapsed precision | Lapsed recall | Lapsed F1 |
|---|---|---|---:|---|---:|---:|---:|
| Logistic regression | `C ∈ {0.01, 0.1, 1, 10, 100}`, `max_iter=1000`, 5-fold CV | C = 100 | 0.651 | `[[111, 32], [31, 58]]` | 0.644 | 0.652 | 0.648 |
| Random forest | `max_depth ∈ {3, 5, 8}`, `min_samples_leaf ∈ {5, 10, 20}`, 100 trees, seed 42, 5-fold CV (45 fits) | depth = 8, leaf = 10 | 0.681 | `[[113, 30], [31, 58]]` | 0.659 | 0.652 | 0.655 |

The selected model is the random forest because its test F1 (0.655) is slightly higher than logistic regression and exceeds the 0.55 requirement. Missing a lapsing customer means losing a chance to retain them, while wrongly targeting a loyal customer wastes campaign budget and may annoy them; both errors matter, so the team should tune thresholds with campaign costs and monitor both precision and recall.

Class-balance experiment: the original training split has 570 not-lapsed and 354 lapsed rows. Random oversampling with replacement (seed 42) adds lapsed training rows until the classes are 570 each; the test set remains unchanged.

| Logistic regression | Precision | Recall | F1 |
|---|---:|---:|---:|
| Original training data | 0.644 | 0.652 | 0.648 |
| Oversampled training data | 0.589 | 0.854 | 0.697 |

Recall moved the most, increasing by about 19 percentage points, while precision decreased. This helps the retention team identify more customers who may lapse, but creates more false-positive outreach.

The five leading random-forest features were frequency (0.280), monetary (0.271), recency (0.115), average delivery time (0.061) and cancellation rate (0.052). In the tuned logistic model, recency has a positive coefficient (+0.039), frequency negative (−1.697), and monetary positive (+0.283): greater recency and monetary value increase predicted lapse odds in this fitted model, while higher frequency decreases them. The selected model is saved to `outputs/lapse_model.joblib`, reloaded, and its predictions match exactly. `outputs/retention_list.csv` contains up to 10 Premium customers in the test set ranked by predicted lapse probability; **1** of the listed customers actually lapsed. This retention list is illustrative and not a leakage-free evaluation because the Premium flag is a post-hoc whole-window label that includes September.

## Part C

### Guardrails, privacy and validation

Tickets are processed in file order. Four blank tickets are marked `skipped_empty`; four tickets over 400 estimated tokens (734 tokens each) are marked `flagged_long`. Neither group is sent to the model. The email, phone and order-ID patterns are applied in that order before prompting; a second pattern scan asserts that no matching PII remains. Across all 164 tickets, redactions were **55 emails**, **91 phone numbers**, and **132 order IDs**, **278 total**. The user message content is exactly the redacted ticket text, with no extra instructions appended.

Both system prompts include the `CLASSIFY` marker and only permit `billing`, `delivery`, `technical`, and `fraud`, with JSON-only output of the form `{"category": "...", "reason": "..."}`. The zero-shot prompt contains no `EXAMPLES:` line; the few-shot prompt includes at least two labelled examples, including fraud and technical. Transient failures are retried with exponential backoff and jitter (`BASE_DELAY = 0.05` seconds), with a log line for each retry. Each full run recovers all **19** simulated transient errors. JSON categories are validated against the allowlist; non-JSON text is accepted only when regex word extraction finds exactly one allowed category; all other outputs go to human review rather than being guessed.

| Full-run metric | Zero-shot | Few-shot |
|---|---:|---:|
| Tickets classified | 147 | 147 |
| JSON-valid classifications | 135 | 135 |
| Keyword-valid classifications | 12 | 12 |
| Invalid outputs routed to human review | 9 | 9 |
| Accuracy among classified tickets | 85.7% (126/147) | 100.0% (147/147) |
| Total estimated tokens | 11,428 | 15,484 |
| Tokens per classified ticket | 77.74 | 105.33 |
| Transient errors recovered | 19 | 19 |

Zero-shot confusion table (rows are true category; columns are predicted category; only classified tickets are included):

| True \ Predicted | Billing | Delivery | Technical | Fraud |
|---|---:|---:|---:|---:|
| Billing | 38 | 0 | 0 | 0 |
| Delivery | 0 | 36 | 0 | 0 |
| Technical | 10 | 0 | 29 | 0 |
| Fraud | 11 | 0 | 0 | 23 |

The few-shot run uses 4,056 more tokens overall, or about 27.59 extra tokens per classified ticket, but accuracy improves from 85.7% to 100%. In this synthetic support setting the extra cost is worth paying because the zero-shot prompt confuses 11 fraud and 10 technical tickets with billing, while few-shot examples eliminate those observed mistakes. A production decision should compare the extra token cost with the operational cost of misrouting sensitive fraud reports.

Budget demo: the zero-shot run uses a **5,000-token budget**; once the running usage is already at or above that budget, no further call is made. **87** tickets end with `budget_exhausted` and are routed to human review. The budget may be exceeded by the last successful response because the rule checks usage before the next call.

The final `outputs/triage_results.csv` is a 164-row few-shot run with budget 100,000. `skipped_empty`, `flagged_long`, `flagged_invalid_output` and `budget_exhausted` always route to `human_review` with NORMAL priority. Valid fraud classifications route to `security_team` with HIGH priority; other valid tickets from Premium customers route to `premium_desk` with HIGH priority; remaining valid tickets route to `support_queue` with NORMAL priority. Missing segment records are treated as not Premium. Rule-validated classifications are labelled `classified_json` or `classified_keyword`; the invalid/guardrail cases are explicitly handed to humans.

## Part D

### Policy retrieval and grounded answers

The five policy documents produce **15 chunks** using 40-word chunks with 10-word overlap: refund 4, delivery 3, escalation 3, payment 3 and privacy 2. `TfidfVectorizer()` is fitted on chunk text; retrieval transforms a query and ranks cosine similarity, returning source, score and text. Term frequency (TF) measures how much a term appears in a chunk; inverse document frequency (IDF) downweights terms that occur in many chunks. A limitation is that TF-IDF is lexical, so semantically equivalent phrases with different words may not match well.

| Evaluation | Expected document in top 3 (12 answerable questions) | Expected fact substring in answer | Fallback for unanswerable questions (2) |
|---|---:|---:|---:|
| 10-word overlap | 12/12 | 12/12 | 2/2 |
| Zero overlap | 12/12 | 10/12 | 2/2 |

With zero overlap, the question “What delivery time makes a completed order eligible for a full refund?” and “How soon is money returned when a payment fails but is deducted?” fail the expected-fact check because the answer-bearing wording is split at a chunk boundary. The assistant sends the top three chunks with source labels in the required `CONTEXT:` / `QUESTION:` message and asks the offline model to answer only from that context; the two out-of-scope questions receive the fallback beginning `I could not find`.

### Refund tools and ReAct scenarios

The tool map contains `get_order`, `search_policy`, `calculate_refund`, `issue_refund` and `escalate_to_human`. The first reads `data/instamart.db`; the second uses the same TF-IDF retrieval; the third returns a full order amount only when delivery exceeds 45 minutes; the final two append a timestamped JSON record to `outputs/refund_ledger.jsonl`. Only completed orders can be refunded. Refunds above INR 1,000 are escalated and are not issued automatically. All scenario requests are treated as being inside the 7-day request window.

| Scenario | Status | Refund / escalation amount | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | INR 128.25 | 5 | 992 |
| S2 | REFUND | INR 135.00 | 5 | 992 |
| S3 | ESCALATE | INR 1,999.00 (human approval) | 5 | 979 |
| S4 | DENIED | INR 0.00 | 4 | 681 |
| S5 | DENIED | INR 0.00 | 3 | 394 |
| S6 | NOT_FOUND | INR 0.00 | 2 | 110 |
| S7 | FALLBACK_MAX_ITERATIONS | INR 0.00 | 6 | 3,114 |

S7 completes six model calls under the default 6,000-token limit and stops with `FALLBACK_MAX_ITERATIONS`. The safe-stop demonstrations return **`FALLBACK_TOKEN_LIMIT`** for S1 with `token_limit=200` and **`FALLBACK_TIMEOUT`** for S1 with `time_limit_seconds=0`. Each fallback returns completed steps and a short human-handoff summary.

Crash-and-resume demonstration: the run is checkpointed after each tool observation. A simulated crash occurs immediately after the second tool execution, and `resume_agent` loads the saved messages and continues without repeating the earlier tools. The tool sequence across the original run and resume is exactly: **`get_order` → `search_policy` → `calculate_refund` → `issue_refund`**. The resumed result is `REFUND: INR 128.25 issued.`; there are four tool executions overall and no repeated execution.

### ReAct versus plan-and-execute (under 250 words)

The ReAct implementation makes multiple calls because the model sees each observation before deciding what to do next: S1 takes five calls, S5 takes three, and S6 takes two. This adds prompt/tool-observation tokens compared with a plan-and-execute design that asks for a complete plan once and replans only after an error. However, the observation-driven approach handles unexpected facts naturally: a missing order can stop as `NOT_FOUND`, and a cancelled order can be denied after the database lookup rather than blindly following a prewritten refund plan. A plan-and-execute workflow can be more token-efficient when all data is predictable, but its plan must be revalidated after each tool result to avoid acting on stale assumptions. For this refund workflow I would keep ReAct with deterministic tool-side rules and human escalation: refund issuance is a financial action, and safety and correct handling of cancelled, missing or high-value orders are more important than minimising a few offline model calls.

## Limitations and Next Steps

- The data, policies and tickets are synthetic and do not describe real customers or real company policy. Model evaluation numbers reflect the deterministic offline mock, not a general-purpose language model.
- TF-IDF retrieval is word-based and may miss paraphrases. A next step would be a larger policy evaluation set and retrieval quality checks before any deployment.
- Lapse predictions should be monitored over time and calibrated against real retention campaign costs before being used operationally. The retention list is illustrative and not a leakage-free evaluation because its Premium flag is a post-hoc whole-window label that includes the September period used to define lapse.
- Refund actions are simulated ledger entries only; a real system would require authorization, idempotency, audit controls, human approval workflow, transaction safety and access controls.
- No hosted LLM, API key or network call is required or used by this project.
