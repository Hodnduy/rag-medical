from pathlib import Path
import json
import argparse

from src.infrastructure.splitter.sentence_chunker import Chunker, ChunkerConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=str, required=True, help="Input directory with JSONL files.")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory for chunked JSONL files.")
    ap.add_argument("--max-words", type=int, default=530, help="Maximum words per chunk.")
    ap.add_argument("--overlap-words", type=int, default=80, help="Number of overlapping words between chunks.")
    ap.add_argument("--min-words", type=int, default=40, help="Minimum words per chunk.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunker = Chunker(
        cfg=ChunkerConfig(
            max_words=args.max_words,
            overlap_words=args.overlap_words,
            min_words=args.min_words,
        )
    )

    for fp in in_dir.glob("nhs_cleaned.jsonl"):
        out_fp = out_dir / fp.name
        with fp.open("r", encoding="utf-8") as f_in, out_fp.open("w", encoding="utf-8") as f_out:
            for record in f_in:
                data = json.loads(record)
                chunks = chunker.clean_and_chunk(data)
                for i, chunk in enumerate(chunks):
                    out_record = data.copy()
                    out_record["chunk_index"] = i
                    out_record["chunk_text"] = chunk
                    f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()