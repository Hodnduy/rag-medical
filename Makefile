PYTHON     := uv run python
DATA_DIR   := data
RAW_DIR    := $(DATA_DIR)
CHUNK_DIR  := $(DATA_DIR)
MAX_WORDS     := 530
OVERLAP_WORDS := 80
MIN_WORDS     := 40
QUERY   := What are the common symptoms of abdominal aortic aneurysm?
TOP_N   := 3

.DEFAULT_GOAL := help

.PHONY: install
install:
	uv sync

.PHONY: crawl
crawl:
	$(PYTHON) scripts/ingest_nhs.py

.PHONY: chunk
chunk:
	$(PYTHON) scripts/run_chunking.py \
		--in-dir $(RAW_DIR) \
		--out-dir $(CHUNK_DIR) \
		--max-words $(MAX_WORDS) \
		--overlap-words $(OVERLAP_WORDS) \
		--min-words $(MIN_WORDS)

.PHONY: embed
embed:
	$(PYTHON) scripts/run_embedding.py

.PHONY: pipeline
pipeline: crawl chunk embed

.PHONY: rag
rag:
	$(PYTHON) scripts/run_rag.py --query "$(QUERY)" --top-n $(TOP_N)

.PHONY: test-model
test-model:
	$(PYTHON) scripts/test_model.py

.PHONY: clean-cache
clean-cache:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true