from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Dict, List, Any

_bullet_pat = re.compile(r"^\s*([-•*])\s*")
_multi_space = re.compile(r"[ \t]+")
_sentence_splitter = re.compile(r"(?<=[\.\!\?])\s+(?=[A-Z0-9])")
_heading_pat = re.compile(r"^(#+\s+|[A-Z][A-Za-z0-9 ,-/()']{0,80}:)$")

@dataclass(frozen=True)
class ChunkerConfig:
    max_words: int = 530
    overlap_words: int = 80
    min_words: int = 40

class Chunker:
    def __init__(self, cfg: ChunkerConfig = ChunkerConfig()):
        self.cfg = cfg

    @staticmethod
    def _normalize_lines(text: str) -> List[str]:
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n \t\uFEFF")
        text = _multi_space.sub(" ", text)
        lines = [ln.strip() for ln in text.split("\n")]
        lines = [ln for ln in lines if ln]
        out = []
        for ln in lines:
            m = _bullet_pat.match(ln)
            if m:
                ln = "- " + ln[m.end():].strip()
            out.append(ln)
        return out

    @staticmethod
    def _dedup(lines: List[str]) -> List[str]:
        out = []
        seen_in_block = set()
        in_bullet_block = False

        def norm(s: str) -> str:
            return re.sub(r"\s+", " ", s.strip().lower())
        
        for ln in lines:
            is_bullet = ln.startswith("- ")
            if is_bullet:
                if not in_bullet_block:
                    in_bullet_block = True
                    seen_in_block.clear()
                key = norm(ln[2:])
                if key in seen_in_block:
                    continue
                seen_in_block.add(key)
                out.append(ln)
            else:
                in_bullet_block = False
                out.append(ln)
        return out
    
    def _is_loose_heading(ln: str) -> bool:
        w = ln.strip()
        if not w or len(w.split()) > 5: 
            return False
        if re.search(r"[.:;!?]$", w): 
            return False
        return bool(re.match(r"^[A-Z][A-Za-z0-9 ,-/()']+$", w))
    
    @staticmethod
    def _join_to_paragraphs(lines: List[str]) -> str:
        paragraphs, buff = [], []
        def flush():
            if buff:
                paragraphs.append(" ".join(buff).strip())
                buff.clear()
        for ln in lines:
            if ln.startswith("- ") or _heading_pat.match(ln) or Chunker._is_loose_heading(ln):
                flush()
                paragraphs.append(ln)
            else:
                buff.append(ln)
        flush()
        return "\n".join(paragraphs).strip()

    @staticmethod
    def _sentences_for_chunking(text: str) -> List[str]:
        units = []
        for block in text.split("\n"):
            b = block.strip()
            if not b: 
                continue
            if b.startswith("- ") or _heading_pat.match(b):
                units.append(b)
            else:
                for s in _sentence_splitter.split(b):
                    s = s.strip()
                    if s:
                        units.append(s)
        return units

    @staticmethod
    def _wc(s: str) -> int:
        return len(s.split())

    def _min_words_filter(self, texts: List[str]) -> List[str]:
        if not texts: return texts
        out: List[str] = []
        for t in texts:
            first = t.split("\n", 1)[0]
            if self._wc(t) >= self.cfg.min_words or first.startswith("- ") or first.startswith("#"):
                out.append(t)
            elif out:
                out[-1] = (out[-1] + "\n" + t).strip()
            else:
                out.append(t)
        return out

    def _chunk_units(self, units: List[str]) -> List[str]:
        if not units: return []
        max_words, overlap = self.cfg.max_words, self.cfg.overlap_words

        if sum(self._wc(u) for u in units) <= max_words:
            return ["\n".join(units)]

        chunks, cur, cur_words = [], [], 0
        for u in units:
            w = self._wc(u)
            if cur_words + w <= max_words:
                cur.append(u); cur_words += w
            else:
                chunks.append("\n".join(cur).strip())
                back, bw = [], 0
                if overlap > 0 and cur:
                    for x in reversed(cur):
                        xw = self._wc(x)
                        if bw + xw > overlap: break
                        back.append(x); bw += xw
                    back.reverse()
                cur = back + [u]
                cur_words = sum(self._wc(x) for x in cur)

        if cur:
            chunks.append("\n".join(cur).strip())
        return self._min_words_filter(chunks)

    def clean_and_chunk(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        condition = (record.get("condition") or "").strip()
        section   = (record.get("section") or "").strip()
        url       = (record.get("url") or "").strip()
        raw       = record.get("content") or ""

        lines   = self._normalize_lines(raw)
        lines   = self._dedup(lines)
        cleaned = self._join_to_paragraphs(lines)

        units = self._sentences_for_chunking(cleaned)
        texts = self._chunk_units(units)

        context_prefix = f"Condition: {condition}. Section: {section}. "

        out: List[Dict[str, Any]] = []
        for i, txt in enumerate(texts):
            enriched_text = context_prefix + txt
            out.append({
                "id": f"{condition}:{section}:{i}",
                "text": enriched_text,
                "meta": {
                    "condition": condition,
                    "section": section,
                    "url": url,
                    "word_count": self._wc(txt),
                    "raw_content": txt,
                    "chunk_index": i,
                    "chunk_count": len(texts),
                    "cleaned": True,
                    "max_words": self.cfg.max_words,
                    "overlap_words": self.cfg.overlap_words,
                }
            })
        return out