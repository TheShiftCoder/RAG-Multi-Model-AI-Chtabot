"""Document Chunking Module.

Splits normalized text into meaningful chunks with configurable size and overlap,
respecting paragraph and sentence boundaries.
"""

from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[str]:
    """Splits text into chunks of roughly `chunk_size` characters with `chunk_overlap`.

    Uses a recursive splitting strategy prioritizing paragraphs and sentences.
    """
    if not text or not text.strip():
        return []

    if len(text) <= chunk_size:
        return [text.strip()]

    # Split into paragraphs first
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > chunk_size:
            # Paragraph itself is too big: split by sentences
            sentences = [s.strip() for s in para.replace(".\n", ". ").split(". ") if s.strip()]
            for s in sentences:
                s_full = s + "."
                if current_length + len(s_full) > chunk_size and current_chunk:
                    chunk_str = " ".join(current_chunk)
                    chunks.append(chunk_str)
                    # Handle overlap by keeping the trailing items
                    overlap_len = 0
                    overlap_items: List[str] = []
                    for item in reversed(current_chunk):
                        if overlap_len + len(item) <= chunk_overlap:
                            overlap_items.insert(0, item)
                            overlap_len += len(item)
                        else:
                            break
                    current_chunk = overlap_items
                    current_length = sum(len(x) for x in current_chunk)

                current_chunk.append(s_full)
                current_length += len(s_full)
        else:
            if current_length + len(para) > chunk_size and current_chunk:
                chunk_str = "\n\n".join(current_chunk)
                chunks.append(chunk_str)
                # Apply overlap
                overlap_len = 0
                overlap_items = []
                for item in reversed(current_chunk):
                    if overlap_len + len(item) <= chunk_overlap:
                        overlap_items.insert(0, item)
                        overlap_len += len(item)
                    else:
                        break
                current_chunk = overlap_items
                current_length = sum(len(x) for x in current_chunk)

            current_chunk.append(para)
            current_length += len(para)

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return [c for c in chunks if c.strip()]

