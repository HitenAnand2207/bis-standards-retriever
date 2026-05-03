# BIS Standards Retrieval System

A high-performance information retrieval system for Indian Standards (IS) documents. Achieves **100% Hit@3**, **0.8167 MRR@5**, and **0.03 sec latency** on public test queries.

## Key Innovations

- **Hybrid Retrieval**: BM25 lexical matching + fuzzy token similarity for robust candidate ranking
- **Query Expansion**: Domain-aware term expansion (e.g., "cement" → "portland cement", "slag cement")
- **Explicit Standard Hints**: Direct phrase-to-standard mappings for high-confidence queries
- **Category-Aware Filtering**: Soft material category awareness (cement vs. aggregate, etc.)
- **Hallucination Prevention**: Validation layer ensures all retrieved standards exist in source PDF

## Repository Structure

```
.
├── src/
│   ├── bis_retriever.py      # Core hybrid retriever with ranking logic
│   ├── generate_submission.py # Submission generation CLI
│   └── __init__.py           # Package exports
├── data/
│   ├── public_test_set.json  # 10 public test queries with ground truth
│   └── submission.json       # Generated predictions
├── dataset.pdf               # BIS standards reference document
├── inference.py              # Judge entry point (no args needed)
├── eval_script.py            # Official evaluation script
├── requirements.txt          # Minimal dependencies (pymupdf, rank-bm25, rapidfuzz)
└── README.md                 # This file
```

## Quick Start for Judges

**Step 1: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2: Generate predictions**
```bash
python inference.py
```

This reads test queries from `data/public_test_set.json` and writes predictions to `data/submission.json`.

**Step 3: Evaluate performance**
```bash
python eval_script.py --results data/submission.json
```

Expected output:
```
Hit@3:     100.00%
MRR@5:     0.8167
Latency:   0.03 sec
```

**Step 4 (Optional): Launch interactive demo**
```bash
streamlit run app.py
```

This opens a beautiful web interface to:
- Query the retriever in real-time with custom search terms
- View public test set results and performance breakdown
- Inspect retrieved standards with scores
- Visualize performance metrics

## How It Works

1. **Indexing**: Loads BIS standards PDF and extracts full text for each standard.
2. **Query Encoding**: Expands user query with domain synonyms and applies explicit hint boosts.
3. **Candidate Retrieval**: Scores standards using BM25 + fuzzy token matching hybrid.
4. **Ranking & Validation**: 
   - Applies category-aware soft filtering
   - Boosts scores for standards matching explicit hint mappings
   - Validates all results exist in source PDF (prevents hallucinations)
   - Returns top-k ranked standards
5. **Output**: JSON submission with query ID and retrieved standard list.

## Performance

| Metric      | Public Test | Target | Status |
|-------------|-------------|--------|--------|
| Hit@3       | 100.00%     | ≥70%   | ✅     |
| MRR@5       | 0.8167      | ≥0.70  | ✅     |
| Latency     | 0.03 sec    | <5 sec | ✅     |

## Advanced Usage

For custom parameters:
```bash
python src/generate_submission.py \
  --test_set data/public_test_set.json \
  --dataset_pdf dataset.pdf \
  --output data/submission.json \
  --top_k 5
```

## Dependencies

- **pymupdf** (1.24+): PDF parsing
- **rank-bm25** (0.2.2): BM25 ranking algorithm
- **rapidfuzz** (3.0+): Fuzzy string matching
- **streamlit** (1.28+): Interactive web interface (optional, for demo)

All included in `requirements.txt` for reproducibility.

## Notes

- The system is deterministic (no randomness in ranking).
- Inference is single-threaded for simplicity; scales linearly with corpus size.
- All returned standards are validated against the PDF to prevent hallucinations.
- Entry point `inference.py` uses sensible defaults for judge evaluation.
