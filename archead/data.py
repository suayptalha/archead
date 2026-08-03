from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from datasets import load_dataset


def stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(str(v) for _, v in sorted(value.items()) if v is not None)
    if isinstance(value, (list, tuple)):
        return "\n".join(str(v) for v in value if v is not None)
    return str(value)


def row_to_text(row: Dict[str, object], columns: Sequence[str]) -> str:
    parts: List[str] = []
    for col in columns:
        if col in row:
            s = stringify(row[col]).strip()
            if s:
                parts.append(s)
    if not parts:
        for col in ("text", "content", "code", "question", "answer", "translation", "output"):
            if col in row:
                s = stringify(row[col]).strip()
                if s:
                    parts.append(s)
    return "\n".join(parts).strip()


def load_texts(
    *,
    dataset: Optional[str],
    dataset_config: Optional[str],
    split: str,
    text_columns: Sequence[str],
    max_rows: int,
    min_chars: int,
    seed: int,
    streaming: bool,
    row_offset: int,
    text_file: Optional[str] = None,
) -> List[str]:
    texts: List[str] = []
    if text_file:
        p = Path(text_file)
        if p.suffix.lower() == ".jsonl":
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    s = row_to_text(json.loads(line), text_columns)
                    if len(s) >= min_chars:
                        texts.append(s)
                    if len(texts) >= max_rows:
                        break
        else:
            buf: List[str] = []
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if line:
                        buf.append(line)
                    elif buf:
                        s = "\n".join(buf).strip()
                        if len(s) >= min_chars:
                            texts.append(s)
                        buf = []
                    if len(texts) >= max_rows:
                        break
            if buf and len(texts) < max_rows:
                s = "\n".join(buf).strip()
                if len(s) >= min_chars:
                    texts.append(s)
    else:
        if not dataset:
            raise ValueError("Either dataset or text_file is required")
        ds = (
            load_dataset(dataset, dataset_config, split=split, streaming=streaming)
            if dataset_config and dataset_config != "default"
            else load_dataset(dataset, split=split, streaming=streaming)
        )
        skipped = 0
        for row in ds:
            if skipped < row_offset:
                skipped += 1
                continue
            s = row_to_text(row, text_columns)
            if len(s) >= min_chars:
                texts.append(s)
            if len(texts) >= max_rows:
                break
    if not texts:
        raise RuntimeError("No usable texts loaded")
    rng = random.Random(seed)
    rng.shuffle(texts)
    return texts

