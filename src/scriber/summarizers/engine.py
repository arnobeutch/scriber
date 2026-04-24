"""RAG engine to summarize meeting transcripts using LangChain, ChromaDB, and Ollama."""

from pathlib import Path

from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings

from scriber import constants as my_constants
from scriber.logger import my_logger

CHROMA_PERSIST_DIR = Path("chroma_db")
_CHUNK_SIZE = 500
_CHUNK_OVERLAP = 50


def pack_utterances(
    utterances: list[tuple[str, str]],
    chunk_size: int = _CHUNK_SIZE,
    overlap: int = _CHUNK_OVERLAP,
) -> list[str]:
    """Greedily pack speaker-tagged utterances into ~chunk_size-char chunks.

    Each utterance stays intact (no mid-sentence splits). A tail of the
    previous chunk (~``overlap`` chars, rounded up to whole utterances) is
    prepended to the next chunk for retrieval continuity. If a single
    utterance exceeds ``chunk_size``, it becomes its own chunk.
    """
    lines = [f"{speaker} : {text}" for speaker, text in utterances]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1  # +1 for the joining newline
        if current and current_len + line_len > chunk_size:
            chunks.append("\n".join(current))
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(current):
                if tail_len >= overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            current = tail
            current_len = tail_len
        current.append(line)
        current_len += line_len

    if current:
        chunks.append("\n".join(current))

    return chunks


def build_vectorstore_from_utterances(
    utterances: list[tuple[str, str]],
    model: str,
) -> Chroma:
    """Return persistent Chroma vector store updated with new transcript.

    Args:
        utterances (list[tuple[str, str]]): Speaker-tagged utterances.
        model (str): Ollama model name.

    Returns:
        Chroma: Persistent vector store.

    """
    documents = [Document(page_content=chunk) for chunk in pack_utterances(utterances)]

    embeddings = OllamaEmbeddings(model=model)

    # Load existing or create new Chroma DB
    if CHROMA_PERSIST_DIR.exists():
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=str(CHROMA_PERSIST_DIR),
        )
    else:
        vectorstore = Chroma.from_documents(  # pyright: ignore[reportUnknownMemberType]  # langchain client_settings / collection_metadata typed as Unknown
            documents=documents,
            embedding=embeddings,
            persist_directory=str(CHROMA_PERSIST_DIR),
        )
        return vectorstore  # noqa: RET504

    # Append new documents
    vectorstore.add_documents(documents)

    return vectorstore


def generate_summary(
    utterances: list[tuple[str, str]],
    language: str,
    model: str,
    prompt: str | None = None,
) -> str:
    """Return markdown-formatted summary from structured utterances.

    Args:
        utterances: Speaker-tagged utterances.
        language: 'fr' or 'en' — used only when ``prompt`` is not given.
        model: Model name for Ollama (e.g., 'mistral').
        prompt: Override the built-in language-defaulted prompt (e.g. for
            summary-mode templates).

    Returns:
        str: Markdown summary of the meeting in the requested language.

    """
    vectorstore = build_vectorstore_from_utterances(utterances, model)

    my_logger.info("Generating summary...")
    llm = ChatOllama(model=model)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        return_source_documents=False,
    )

    if prompt is None:
        prompt = (
            my_constants.RAG_FRENCH_PROMPT if language == "fr" else my_constants.RAG_ENGLISH_PROMPT
        )
    result = qa_chain.invoke({"query": prompt})
    return result["result"]
