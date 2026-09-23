"""
Evaluation harness. This is the single most important file in the repo for
your resume claim — it's what lets you say "faithfulness improved from X%
to Y%" instead of "it seems to work."

Two modes:
- run_ragas_eval(): uses the real RAGAS library (context_precision,
  context_recall, faithfulness, answer_relevancy) against an LLM judge.
  This is what you should run before/after any pipeline change and report
  in your README.
- run_lightweight_eval(): a dependency-light approximation of the same
  four metrics using lexical overlap instead of an LLM judge, so you can
  get a signal immediately without RAGAS + a judge-model API budget. Use
  this while iterating; use the real RAGAS run for the numbers you quote.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from rag_platform.retrieval.hybrid import HybridRetriever
from rag_platform.text_utils import meaningful_tokens as _tokens


@dataclass
class EvalExample:
    question: str
    ground_truth_answer: str
    ground_truth_doc_ids: list[str]  # which source docs *should* be retrieved


@dataclass
class EvalResult:
    question: str
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float


def load_eval_dataset(path: str) -> list[EvalExample]:
    with open(path) as f:
        raw = json.load(f)
    return [EvalExample(**item) for item in raw]


def run_lightweight_eval(
    dataset: list[EvalExample], retriever: HybridRetriever, generate_fn
) -> list[EvalResult]:
    """
    generate_fn: callable(question, chunks) -> answer_text. Pass the same
    generation call used in serving/app.py so eval reflects the real
    pipeline, not a mocked one.
    """
    results = []
    for example in dataset:
        retrieved = retriever.retrieve(example.question)
        retrieved_doc_ids = {c["payload"].get("doc_id") for c in retrieved}
        relevant_doc_ids = set(example.ground_truth_doc_ids)

        # context_precision: fraction of retrieved chunks that are from a
        # relevant document
        context_precision = (
            len(retrieved_doc_ids & relevant_doc_ids) / len(retrieved_doc_ids)
            if retrieved_doc_ids
            else 0.0
        )
        # context_recall: fraction of relevant documents that were retrieved
        context_recall = (
            len(retrieved_doc_ids & relevant_doc_ids) / len(relevant_doc_ids)
            if relevant_doc_ids
            else 0.0
        )

        answer = generate_fn(example.question, retrieved)

        # faithfulness proxy: answer vocabulary overlap with retrieved context
        context_text = " ".join(c["payload"]["text"] for c in retrieved)
        answer_tok, context_tok = _tokens(answer), _tokens(context_text)
        faithfulness = len(answer_tok & context_tok) / len(answer_tok) if answer_tok else 0.0

        # answer_relevancy proxy: answer vocabulary overlap with the
        # ground-truth answer (crude but directionally useful)
        gt_tok = _tokens(example.ground_truth_answer)
        answer_relevancy = len(answer_tok & gt_tok) / len(gt_tok) if gt_tok else 0.0

        results.append(
            EvalResult(
                question=example.question,
                context_precision=round(context_precision, 3),
                context_recall=round(context_recall, 3),
                faithfulness=round(faithfulness, 3),
                answer_relevancy=round(answer_relevancy, 3),
            )
        )
    return results


def run_ragas_eval(dataset: list[EvalExample], retriever: HybridRetriever, generate_fn):
    """
    Real RAGAS evaluation. Requires `ragas` + `datasets` and an LLM judge
    (defaults to using the same Anthropic model configured for generation).
    Run this before/after any retrieval or prompt change — the four scores
    it produces are the numbers to put in your README and resume bullet.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    questions, answers, contexts, ground_truths = [], [], [], []
    for example in dataset:
        retrieved = retriever.retrieve(example.question)
        answer = generate_fn(example.question, retrieved)
        questions.append(example.question)
        answers.append(answer)
        contexts.append([c["payload"]["text"] for c in retrieved])
        ground_truths.append(example.ground_truth_answer)

    hf_dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )
    return evaluate(
        hf_dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    )


def summarize(results: list[EvalResult]) -> dict:
    if not results:
        return {}
    n = len(results)
    return {
        "n_examples": n,
        "avg_context_precision": round(sum(r.context_precision for r in results) / n, 3),
        "avg_context_recall": round(sum(r.context_recall for r in results) / n, 3),
        "avg_faithfulness": round(sum(r.faithfulness for r in results) / n, 3),
        "avg_answer_relevancy": round(sum(r.answer_relevancy for r in results) / n, 3),
    }
