"""
Dataset Generator
=================
Takes your REAL chunks/documents and generates synthetic QA pairs for evaluation.

The flow:
    1. Takes a chunk of text (from your actual indexed data)
    2. Uses an LLM to generate a question that this chunk answers
    3. Uses the LLM to extract the specific answer from the chunk
    4. Stores: (question, expected_answer, source_chunk, chunk_id)

This gives you a REAL evaluation dataset — when you run retrieval on the question,
the source_chunk should be in the top results. If it's not, your retrieval is broken.

Usage:
    generator = DatasetGenerator(llm_fn=my_llm_call)
    dataset = generator.generate_from_chunks(my_chunks)
    dataset.save("eval_dataset.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Any
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="Logs/ragprobe_dataset_generator.log")

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """A single evaluation sample — one question with its ground truth."""

    question: str                       # Generated question
    expected_answer: str                # The correct answer (extracted from chunk)
    source_chunk: str                   # The chunk that contains the answer
    chunk_id: str = ""                  # ID of the source chunk (for retrieval validation)
    metadata: dict[str, Any] = field(default_factory=dict)  # Extra info (repo, file, etc.)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalDataset:
    """Collection of evaluation samples."""

    samples: list[EvalSample] = field(default_factory=list)
    name: str = "eval_dataset"
    description: str = ""

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def save(self, path: str | Path) -> None:
        """Save dataset to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "description": self.description,
            "sample_count": len(self.samples),
            "samples": [s.to_dict() for s in self.samples],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "EvalDataset":
        """Load dataset from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = [EvalSample(**s) for s in data["samples"]]
        return cls(
            samples=samples,
            name=data.get("name", ""),
            description=data.get("description", ""),
        )


# --- Prompts for QA generation ---

QUESTION_GENERATION_PROMPT = """You are given a chunk of source code or documentation. 
Generate a specific, answerable question that this chunk directly answers.

Rules:
- The question should be something a developer would realistically ask
- The answer MUST be fully contained within the provided chunk
- Be specific — avoid vague questions like "what does this code do?"
- Focus on behavior, logic, parameters, or purpose

Chunk:
---
{chunk}
---

Return ONLY a JSON object:
{{"question": "your generated question", "answer": "the specific answer from the chunk"}}"""


MULTI_QUESTION_PROMPT = """You are given a chunk of source code or documentation.
Generate {count} specific, diverse questions that this chunk directly answers.

Rules:
- Questions should be things a developer would realistically ask
- Each answer MUST be fully contained within the provided chunk
- Make questions diverse: cover behavior, parameters, error handling, usage, etc.
- Be specific — avoid vague questions

Chunk:
---
{chunk}
---

Return ONLY a JSON array:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]"""


class DatasetGenerator:
    """
    Generates evaluation datasets from real chunks.

    Args:
        llm_fn: A callable that takes a prompt string and returns a response string.
                This keeps the generator LLM-agnostic — pass any model's generate function.

    Usage:
        def my_llm(prompt: str) -> str:
            return openai.chat(prompt)

        gen = DatasetGenerator(llm_fn=my_llm)
        dataset = gen.generate_from_chunks(chunks, questions_per_chunk=2)
    """

    def __init__(self, llm_fn: Callable[[str], str]):
        self.llm_fn = llm_fn

    def generate_from_chunks(
        self,
        chunks: list[dict[str, Any]],
        questions_per_chunk: int = 1,
        max_samples: int | None = None,
    ) -> EvalDataset:
        """
        Generate QA pairs from a list of chunks.

        Args:
            chunks: List of dicts with at least 'content' key.
                    Optionally 'chunk_id', 'file_path', 'repo', etc.
            questions_per_chunk: How many questions to generate per chunk.
            max_samples: Maximum total samples to generate (None = no limit).

        Returns:
            EvalDataset with generated QA pairs.
        """
        samples: list[EvalSample] = []

        for chunk_data in chunks:
            if max_samples and len(samples) >= max_samples:
                break

            content = chunk_data.get("content", "")
            chunk_id = chunk_data.get("chunk_id", "")
            metadata = {
                k: v for k, v in chunk_data.items()
                if k not in ("content", "chunk_id")
            }

            # Skip very short chunks (not enough info to ask about)
            if len(content.strip()) < 50:
                logger.info(f"Skipping short chunk (ID: {chunk_id}) — too little content.")
                continue
            
            logger.info(f"Generating QA for chunk (ID: {chunk_id}) with {questions_per_chunk} questions.")
            try:
                if questions_per_chunk == 1:
                    qa_pairs = self._generate_single_qa(content)
                else:
                    qa_pairs = self._generate_multi_qa(content, questions_per_chunk)

                logger.info(f"Successfully generated {len(qa_pairs)} QA pairs for chunk (ID: {chunk_id}).")
                for qa in qa_pairs:
                    samples.append(EvalSample(
                        question=qa["question"],
                        expected_answer=qa["answer"],
                        source_chunk=content,
                        chunk_id=chunk_id,
                        metadata=metadata,
                    ))
            except Exception as e:
                # Skip chunks where generation fails — don't crash the whole run
                logger.error(f"Error generating QA for chunk (ID: {chunk_id}): {e}")
                continue

        return EvalDataset(
            samples=samples,
            name="auto_generated",
            description=f"Generated from {len(chunks)} chunks, {questions_per_chunk} Q per chunk",
        )

    def _generate_single_qa(self, chunk: str) -> list[dict[str, str]]:
        """Generate a single QA pair from a chunk."""
        prompt = QUESTION_GENERATION_PROMPT.format(chunk=chunk[:3000])
        response = self.llm_fn(prompt)
        parsed = self._parse_json(response)
        logger.info(f"Successfully generated single QA pair")

        if isinstance(parsed, dict) and "question" in parsed:
            return [parsed]
        return []

    def _generate_multi_qa(self, chunk: str, count: int) -> list[dict[str, str]]:
        """Generate multiple QA pairs from a chunk."""
        prompt = MULTI_QUESTION_PROMPT.format(chunk=chunk[:3000], count=count)
        response = self.llm_fn(prompt)
        parsed = self._parse_json(response)
        logger.info(f"Successfully generated {count} QA pairs")

        if isinstance(parsed, list):
            return [p for p in parsed if isinstance(p, dict) and "question" in p]
        elif isinstance(parsed, dict) and "question" in parsed:
            return [parsed]
        return []

    def _parse_json(self, text: str) -> Any:
        """Parse JSON from LLM response, handling common formatting issues."""
        text = text.strip()
        # Try to find JSON in the response
        start_bracket = text.find("[")
        start_brace = text.find("{")

        if start_bracket != -1 and (start_brace == -1 or start_bracket < start_brace):
            end = text.rfind("]") + 1
            text = text[start_bracket:end]
        elif start_brace != -1:
            end = text.rfind("}") + 1
            text = text[start_brace:end]

        return json.loads(text)
