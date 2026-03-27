# SentimentHub — Multi-Domain Sentiment Analytics

## Quick Start
```bash
pip install -r requirements.txt
streamlit run ui/app.py
```

## Adding a New Customer (3 steps)
1. Copy `configs/template_rules.yaml` → `configs/<new_id>_rules.yaml` and fill in rules
2. Copy `customers/ppt/pipeline.py` → `customers/<new_id>/pipeline.py`, adjust `extract_speaker_text()` if the transcript format differs
3. Add one line to `core/registry.py`:
```python
"new_id": (NewPipeline, _CONFIGS / "new_id_rules.yaml"),
```
and add a display label to `customer_display_names()`.

## Folder Structure
```
sentiment_platform/
├── core/
│   ├── pipeline.py      # Abstract base + data contracts
│   └── registry.py      # Factory: customer_id → pipeline class
├── engines/
│   ├── regex_engine.py  # Config-driven tier evaluator (loads YAML)
│   └── vader.py         # Chunked VADER wrapper
├── customers/
│   ├── ppt/pipeline.py  # PPT: HTML/plain transcript parsing
│   └── default/         # Fallback: raw text → VADER only
├── configs/
│   ├── ppt_rules.yaml   # PPT sentiment tiers + keywords + noise
│   └── template_rules.yaml
├── ui/app.py            # Streamlit application
└── requirements.txt
```
