# -*- coding: utf-8 -*-
"""
基于 Chroma 向量数据库的 RAG 检索模块。
知识源：/data/ltmk.json（动态知识库）
"""

import json
import hashlib
import shutil
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_EMBEDDING_MODEL, LTMK_PATH, CHROMA_PERSIST_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


def _get_embedding_function():
    """获取 OpenAI 嵌入函数。"""
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    
    model = OPENAI_EMBEDDING_MODEL or "text-embedding-3-small"
    return OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        api_base=OPENAI_BASE_URL or None,
        model_name=model,
    )


# 使用 langchain RecursiveCharacterTextSplitter 对长文本进行分块
# embedding 模型最大 token 数为 8192，使用 tiktoken 按 token 数切分，chunk_size=7000 tokens 留有安全余量
import tiktoken

try:
    _encoding = tiktoken.encoding_for_model("text-embedding-3-small")
except Exception:
    _encoding = tiktoken.get_encoding("cl100k_base")


def _tiktoken_len(text: str) -> int:
    """使用 tiktoken 计算文本的实际 token 数。"""
    return len(_encoding.encode(text))


_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=7000,
    chunk_overlap=200,
    length_function=_tiktoken_len,
    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", "，", ",", " ", ""],
)


def _split_doc(content: str) -> list[str]:
    """使用 langchain 文本分割器将长文档拆分为多个 chunk，防止超出 embedding 模型 token 上限。"""
    chunks = _text_splitter.split_text(content)
    if len(chunks) > 1:
        logger.info("文档过长（%d 字符），拆分为 %d 个 chunk", len(content), len(chunks))
    return chunks


def _generate_doc_id(content: str, index: int) -> str:
    """为文档生成唯一 ID。"""
    hash_val = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]
    return f"doc_{index}_{hash_val}"


def load_ltmk() -> list[str]:
    """从 ltmk.json 加载知识库。"""
    if not LTMK_PATH.exists():
        logger.warning("知识库文件不存在: %s", LTMK_PATH)
        return []
    try:
        with open(LTMK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        knowledge = data.get("knowledge", [])
        logger.info("加载知识库成功，共 %d 条知识", len(knowledge))
        return knowledge
    except Exception as e:
        logger.error("加载知识库失败: %s", e)
        return []


def save_ltmk(knowledge: list[str]) -> None:
    """保存知识库到 ltmk.json。"""
    LTMK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LTMK_PATH, "w", encoding="utf-8") as f:
        json.dump({"knowledge": knowledge}, f, ensure_ascii=False, indent=2)
    logger.info("保存知识库成功，共 %d 条知识", len(knowledge))


class ChromaRAGSearch:
    """
    基于 Chroma 向量数据库的 RAG 检索。
    
    特性：
    - 使用 OpenAI Embedding 进行向量化
    - 支持动态增删改知识
    - 持久化存储，无需每次重建索引
    - 自动检测知识库变化并增量更新
    """

    COLLECTION_NAME = "knowledge_base"

    def __init__(self, persist: bool = True):
        """
        初始化 Chroma RAG。
        
        :param persist: 是否持久化存储（默认 True）
        """
        self.persist = persist
        self._client = None
        self._collection = None
        self._embedding_fn = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """延迟初始化，确保 Chroma 客户端已就绪。"""
        if self._initialized:
            return
        
        try:
            self._do_initialize()
        except Exception as e:
            if self._is_index_corrupted(e):
                logger.warning("检测到索引损坏，自动修复中...")
                self.fix_index()
            else:
                raise

    def _do_initialize(self) -> None:
        """执行实际的初始化逻辑。"""
        logger.info("初始化 Chroma RAG...")
        
        # 创建嵌入函数
        self._embedding_fn = _get_embedding_function()
        
        # 创建 Chroma 客户端
        if self.persist:
            CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(CHROMA_PERSIST_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False),
            )
        
        # 获取或创建 collection
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"description": "知识库向量索引"},
        )
        
        # 同步知识库
        self._sync_knowledge()
        
        self._initialized = True
        logger.info("Chroma RAG 初始化完成，当前索引包含 %d 条文档", self._collection.count())

    @staticmethod
    def _is_index_corrupted(exc: Exception) -> bool:
        """判断异常是否为索引损坏错误。"""
        error_keywords = [
            "Error loading hnsw index",
            "Error creating hnsw segment reader",
            "Error constructing hnsw segment reader",
            "Error executing plan",
            "hnsw",
        ]
        msg = str(exc).lower()
        return any(kw.lower() in msg for kw in error_keywords)

    def fix_index(self) -> None:
        """
        修复损坏的 Chroma 索引：
        1. 关闭当前客户端连接
        2. 删除持久化目录
        3. 重新初始化并从 ltmk.json 重建索引
        """
        logger.warning("开始修复 Chroma 索引...")

        # 重置内部状态
        self._client = None
        self._collection = None
        self._embedding_fn = None
        self._initialized = False

        # 删除损坏的持久化目录
        if self.persist and CHROMA_PERSIST_DIR.exists():
            try:
                shutil.rmtree(CHROMA_PERSIST_DIR)
                logger.info("已删除损坏的索引目录: %s", CHROMA_PERSIST_DIR)
            except Exception as e:
                logger.error("删除索引目录失败: %s", e)
                raise

        # 重新初始化（会自动从 ltmk.json 重建索引）
        self._do_initialize()
        logger.info("Chroma 索引修复完成，重建了 %d 条文档", self._collection.count())

    def _sync_knowledge(self) -> None:
        """
        同步 ltmk.json 与 Chroma 索引。
        检测新增、删除的知识并更新索引。
        """
        knowledge = load_ltmk()
        if not knowledge:
            logger.warning("知识库为空，跳过同步")
            return
        
        # 生成当前知识的 ID 和内容映射（长文档会被拆分为多个 chunk）
        current_docs = {}
        for i, content in enumerate(knowledge):
            if content and content.strip():
                text = content.strip()
                chunks = _split_doc(text)
                for chunk_idx, chunk in enumerate(chunks):
                    doc_id = _generate_doc_id(f"{text[:50]}_{chunk_idx}", i) if len(chunks) > 1 else _generate_doc_id(text, i)
                    current_docs[doc_id] = chunk
        
        # 获取已索引的文档 ID
        existing_ids = set()
        try:
            if self._collection.count() > 0:
                result = self._collection.get()
                existing_ids = set(result.get("ids", []))
        except Exception as e:
            if self._is_index_corrupted(e):
                logger.warning("_sync_knowledge 中检测到索引损坏，将清空并重建")
                self._client.delete_collection(self.COLLECTION_NAME)
                self._collection = self._client.get_or_create_collection(
                    name=self.COLLECTION_NAME,
                    embedding_function=self._embedding_fn,
                    metadata={"description": "知识库向量索引"},
                )
                existing_ids = set()
            else:
                raise
        
        current_ids = set(current_docs.keys())
        
        # 计算需要添加和删除的文档
        to_add_ids = current_ids - existing_ids
        to_delete_ids = existing_ids - current_ids
        
        # 删除过时文档
        if to_delete_ids:
            self._collection.delete(ids=list(to_delete_ids))
            logger.info("删除 %d 条过时文档", len(to_delete_ids))
        
        # 添加新文档（分批，防止请求体过大导致 413 错误）
        if to_add_ids:
            add_ids = list(to_add_ids)
            add_docs = [current_docs[doc_id] for doc_id in add_ids]
            batch_size = 20  # 每批最多 20 条，避免 embedding 请求过大
            for start in range(0, len(add_ids), batch_size):
                batch_ids = add_ids[start:start + batch_size]
                batch_docs = add_docs[start:start + batch_size]
                self._collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                )
                logger.info("已添加第 %d~%d 条文档（共 %d 条）",
                            start + 1, min(start + batch_size, len(add_ids)), len(add_ids))
        
        if not to_add_ids and not to_delete_ids:
            logger.debug("知识库无变化，跳过同步")

    def search(self, query: str, ltm: dict[str, Any] | None = None, top_k: int = 5) -> str:
        """
        语义检索：根据问题查询最相关的知识。
        
        :param query: 用户问题
        :param ltm: 兼容参数（保持与旧接口一致，实际不使用）
        :param top_k: 返回的最大结果数
        :return: 检索到的知识文本（多条用换行分隔）
        """
        self._ensure_initialized()
        
        if not query or not query.strip():
            logger.warning("查询为空")
            return "（查询为空）"
        
        if self._collection.count() == 0:
            logger.warning("知识库索引为空")
            return "（知识库暂无内容）"
        
        logger.info("RAG 检索开始 query=%s top_k=%d", query[:50] + "..." if len(query) > 50 else query, top_k)
        
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, self._collection.count()),
            )
            
            documents = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            
            if not documents:
                logger.info("RAG 检索无结果")
                return "<>"
            
            # 过滤相似度过低的结果（距离阈值，越小越相似）
            filtered = []
            for doc, dist in zip(documents, distances):
                # Chroma 使用 L2 距离，阈值可调整
                if dist < 1.5:  # 可根据实际情况调整
                    filtered.append(doc)
                else:
                    logger.debug("过滤低相似度结果: distance=%.3f", dist)
            
            if not filtered:
                logger.info("RAG 检索结果全部被过滤（相似度过低）")
                return "<nothing>"
            
            result_text = "\n\n".join(filtered)
            logger.info("RAG 检索完成，返回 %d 条结果，总长度 %d 字符", len(filtered), len(result_text))
            return result_text
            
        except Exception as e:
            logger.error("RAG 检索异常: %s", e)
            return "<error>"

    def add_knowledge(self, content: str) -> bool:
        """
        添加新知识到知识库。
        
        :param content: 知识内容
        :return: 是否成功
        """
        self._ensure_initialized()
        
        if not content or not content.strip():
            return False
        
        content = content.strip()
        
        # 更新 ltmk.json
        knowledge = load_ltmk()
        if content in knowledge:
            logger.info("知识已存在，跳过添加")
            return True
        
        knowledge.append(content)
        save_ltmk(knowledge)
        
        # 更新 Chroma 索引（长文档拆分为多个 chunk）
        chunks = _split_doc(content)
        for chunk_idx, chunk in enumerate(chunks):
            doc_id = _generate_doc_id(f"{content[:50]}_{chunk_idx}", len(knowledge) - 1) if len(chunks) > 1 else _generate_doc_id(content, len(knowledge) - 1)
            self._collection.add(
                ids=[doc_id],
                documents=[chunk],
            )
        
        logger.info("添加知识成功: %s...", content[:50] if len(content) > 50 else content)
        return True

    def delete_knowledge(self, content: str) -> bool:
        """
        从知识库删除知识。
        
        :param content: 知识内容
        :return: 是否成功
        """
        self._ensure_initialized()
        
        if not content or not content.strip():
            return False
        
        content = content.strip()
        
        # 更新 ltmk.json
        knowledge = load_ltmk()
        if content not in knowledge:
            logger.warning("知识不存在，无法删除")
            return False
        
        idx = knowledge.index(content)
        knowledge.remove(content)
        save_ltmk(knowledge)
        
        # 更新 Chroma 索引
        doc_id = _generate_doc_id(content, idx)
        try:
            self._collection.delete(ids=[doc_id])
        except Exception as e:
            logger.warning("从索引删除失败（可能 ID 不匹配）: %s", e)
            # 重新同步
            self._sync_knowledge()
        
        logger.info("删除知识成功")
        return True

    def refresh(self) -> None:
        """强制刷新索引，重新同步 ltmk.json。"""
        self._ensure_initialized()
        self._sync_knowledge()
        logger.info("索引刷新完成，当前共 %d 条文档", self._collection.count())

    def count(self) -> int:
        """返回索引中的文档数量。"""
        self._ensure_initialized()
        return self._collection.count()


# 全局单例（延迟初始化）
_chroma_rag_instance: ChromaRAGSearch | None = None


def get_chroma_rag() -> ChromaRAGSearch:
    """获取 Chroma RAG 单例。"""
    global _chroma_rag_instance
    if _chroma_rag_instance is None:
        _chroma_rag_instance = ChromaRAGSearch(persist=True)
    return _chroma_rag_instance
