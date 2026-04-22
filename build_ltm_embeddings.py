#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线构建 LTM embedding 索引。"""

import argparse
import logging
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

from config import LTM_EMBEDDINGS_PATH, LTM_PATH
from memory import ltm as ltm_module
from memory.ltm import load_ltm, save_ltm_embeddings
from utils.llm import reset_embedding_cache
from utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class _ProgressBar:
    """简单终端进度条，避免为脚本额外引入依赖；支持多线程调用 update。"""

    def __init__(self, total: int, desc: str = "Building embeddings", width: int = 32) -> None:
        self.total = max(0, total)
        self.desc = desc
        self.width = max(10, width)
        self.current = 0
        self._lock = threading.Lock()
        self._render()

    def update(self, step: int = 1) -> None:
        with self._lock:
            self.current = min(self.total, self.current + step)
        self._render()

    def close(self) -> None:
        if self.total > 0:
            print(file=sys.stderr)

    def _render(self) -> None:
        if self.total <= 0:
            return
        with self._lock:
            current, total, width = self.current, self.total, self.width
        ratio = current / total
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        message = (
            f"\r{self.desc}: [{bar}] {current}/{total} "
            f"({ratio * 100:5.1f}%)"
        )
        print(message, end="", file=sys.stderr, flush=True)


@contextmanager
def _embedding_progress(total: int):
    """包装 embedding 调用，为离线构建显示实时进度。"""
    if total <= 0:
        yield
        return

    progress = _ProgressBar(total)
    original_get_embedding = ltm_module.get_embedding
    noisy_loggers = ["httpx", "openai._base_client"]
    original_levels = {
        name: logging.getLogger(name).level
        for name in noisy_loggers
    }

    def _get_embedding_with_progress(text: str):
        result = original_get_embedding(text)
        progress.update()
        return result

    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)
    ltm_module.get_embedding = _get_embedding_with_progress
    try:
        yield
    finally:
        ltm_module.get_embedding = original_get_embedding
        for name, level in original_levels.items():
            logging.getLogger(name).setLevel(level)
        progress.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="为 ltm.json 预计算 embedding 索引")
    parser.add_argument("--ltm", type=str, default=str(LTM_PATH), help="LTM JSON 文件路径")
    parser.add_argument(
        "--out",
        type=str,
        default=str(LTM_EMBEDDINGS_PATH),
        help="输出的 embedding 索引文件路径",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        metavar="N",
        help="并行计算的线程数（默认 8，设为 1 则串行）",
    )
    args = parser.parse_args()

    ltm_path = Path(args.ltm)
    out_path = Path(args.out)

    logger.info("开始离线构建 LTM embedding 索引 ltm=%s out=%s", ltm_path, out_path)
    reset_embedding_cache()
    ltm = load_ltm(ltm_path)
    total_entries = len(ltm_module.get_all_qa_entries(ltm))
    max_workers = args.workers if args.workers > 1 else None
    with _embedding_progress(total_entries):
        index_data = ltm_module.build_ltm_embeddings(
            ltm, source_path=ltm_path, max_workers=max_workers
        )
    save_ltm_embeddings(index_data, out_path)
    logger.info(
        "LTM embedding 索引构建完成 out=%s entry_count=%s",
        out_path,
        index_data.get("entry_count", 0),
    )
    print(f"embedding index saved to: {out_path}")
    print(f"entry_count: {index_data.get('entry_count', 0)}")


if __name__ == "__main__":
    main()
