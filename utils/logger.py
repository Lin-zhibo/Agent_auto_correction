# -*- coding: utf-8 -*-
"""
集中配置日志：输出到 log 目录下的文件，并可选输出到控制台。
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

from config import LOG_DIR

# 日志格式：时间 | 级别 | 模块 | 消息
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


class _SizeOrLineRotatingFileHandler(RotatingFileHandler):
    """同时按文件大小或行数触发滚动的 FileHandler。"""

    def __init__(
        self,
        filename: str | Path,
        *,
        max_bytes: int,
        max_lines: int,
        backup_count: int = 5,
        encoding: str = "utf-8",
    ) -> None:
        super().__init__(
            filename=filename,
            mode="a",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding,
        )
        self._max_lines = max_lines
        self._line_count = self._count_existing_lines()

    def _count_existing_lines(self) -> int:
        if not self.baseFilename:
            return 0
        path = Path(self.baseFilename)
        if not path.exists():
            return 0
        try:
            with path.open("r", encoding=self.encoding or "utf-8", errors="ignore") as f:
                return sum(1 for _ in f)
        except OSError:
            return 0

    def shouldRollover(self, record: logging.LogRecord) -> int:  # noqa: N802
        if super().shouldRollover(record):
            return 1
        if self._max_lines <= 0:
            return 0
        msg = self.format(record)
        next_lines = msg.count("\n") + 1
        return 1 if (self._line_count + next_lines) > self._max_lines else 0

    def doRollover(self) -> None:  # noqa: N802
        super().doRollover()
        self._line_count = 0

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        added_lines = msg.count("\n") + 1
        if self.shouldRollover(record):
            self.doRollover()
        if self.stream is None:
            self.stream = self._open()
        self.stream.write(msg + self.terminator)
        self.flush()
        self._line_count += added_lines


def _ensure_log_dir() -> Path:
    """确保 log 目录存在。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR


def setup_logging(
    *,
    level: int = logging.INFO,
    log_file: str | Path | None = "app.log",
    console: bool = True,
) -> None:
    """
    配置根日志：写入 log 目录下的文件，并可选输出到控制台。
    重复调用不会重复添加 handler。
    """
    global _initialized
    if _initialized:
        return
    _ensure_log_dir()
    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if log_file:
        path = LOG_DIR / log_file if isinstance(log_file, str) else Path(log_file)
        fh = _SizeOrLineRotatingFileHandler(
            filename=path,
            max_bytes=1024 * 1024,
            max_lines=50_000,
            backup_count=5,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        root.addHandler(ch)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    获取模块用 logger。若尚未 setup，会先执行一次 setup_logging()。
    """
    if not _initialized:
        setup_logging()
    return logging.getLogger(name)


def truncate_for_log(text: str | None, max_len: int = 250, suffix: str = "...") -> str:
    """
    截断长文本用于日志输出，便于观察中间结果又不刷屏。
    """
    if text is None:
        return ""
    s = str(text).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - len(suffix)] + suffix
