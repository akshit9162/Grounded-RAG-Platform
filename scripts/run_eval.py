"""
Runs the evaluation harness against the sample corpus + eval_dataset.json
and prints a scorecard. This is what produces the numbers you put in your
README and resume bullet.

By default runs the lightweight (no-API-key) proxy metrics. Pass
--ragas to run the real RAGAS metrics instead (requires `ragas`,
`datasets`, and ANTHROPIC_API_KEY set for the judge model).

    python scripts/run_eval.py
    python scripts/run_eval.py --ragas
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rag_platform.evaluation.ragas_eval import (  # noqa: E402
    load_eval_dataset,
    run_lightweight_eval,
    run_ragas_eval,
    summarize,
)
from rag_platform.generation.llm import get_llm_client  # noqa: E402
from rag_platform.generation.prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402
from rag_platform.retrieval.hybrid import HybridRetriever  # noqa: E402
from scripts.ingest_corpus import main as ingest_corpus  # noqa: E402

EVAL_DATASET_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "src" / "rag_platform" / "evaluation" / "eval_dataset.json"
)


def generate_fn(question: str, chunks: list[dict]) -> str:
    llm = get_llm_client()
    prompt = build_user_prompt(question, chunks)
    return llm.generate(SYSTEM_PROMPT, prompt, max_tokens=400)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ragas", action="store_true", help="run real RAGAS metrics instead of the lightweight proxy")
    args = parser.parse_args()

    print("Ingesting corpus...")
    pipeline = ingest_corpus()
    retriever = HybridRetriever(bm25_index=pipeline.bm25_index)
    dataset = load_eval_dataset(str(EVAL_DATASET_PATH))

    if args.ragas:
        print("\nRunning real RAGAS evaluation (requires ANTHROPIC_API_KEY as judge)...")
        result = run_ragas_eval(dataset, retriever, generate_fn)
        print(result)
        return

    print(f"\nRunning lightweight eval on {len(dataset)} examples...")
    results = run_lightweight_eval(dataset, retriever, generate_fn)
    scorecard = summarize(results)

    print("\n" + "=" * 50)
    print("SCORECARD (lightweight proxy metrics)")
    print("=" * 50)
    for k, v in scorecard.items():
        print(f"  {k:28s}: {v}")

    out_path = pathlib.Path(__file__).resolve().parents[1] / "data" / "eval_results.json"
    out_path.write_text(json.dumps({"scorecard": scorecard, "per_question": [r.__dict__ for r in results]}, indent=2))
    print(f"\nFull per-question results written to {out_path}")


if __name__ == "__main__":
    main()
