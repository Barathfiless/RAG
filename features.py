"""
Advanced PDF Features Module — No external APIs required.
Provides: image relevance detection, quiz generation,
smart outline building, and glossary extraction.
"""
import re
import random
from nltk.tokenize import sent_tokenize

# ─── Shared Stop Words ────────────────────────────────────────────────────────
_STOP = {
    'the','a','an','is','are','was','were','be','been','being',
    'have','has','had','do','does','did','will','would','could',
    'should','may','might','shall','can','this','that','these','those',
    'it','its','they','them','we','our','you','your','he','she',
    'his','her','and','or','but','if','as','so','of','to','in','on','at'
}

# ─── Image Relevance Detection ────────────────────────────────────────────────

_IMAGE_KEYWORDS = {
    'image', 'images', 'picture', 'pictures', 'photo', 'photos',
    'figure', 'figures', 'fig', 'diagram', 'diagrams', 'chart', 'charts',
    'graph', 'graphs', 'illustration', 'illustrations', 'visual', 'visuals',
    'show me', 'display', 'depict', 'draw', 'drawing', 'screenshot',
    'plot', 'plots', 'schematic', 'architecture', 'symbol', 'table',
    'map', 'flowchart', 'tree', 'structure', 'layout'
}

def is_image_query(query: str) -> bool:
    """Return True only if the user's query is asking about something visual."""
    q_lower = query.lower()
    tokens = set(re.findall(r'\b\w+\b', q_lower))
    # Check single word keywords
    if tokens & _IMAGE_KEYWORDS:
        return True
    # Check multi-word phrases
    for phrase in ['show me', 'show image', 'any image', 'any picture', 'display image']:
        if phrase in q_lower:
            return True
    return False


# ─── Quiz Generator ───────────────────────────────────────────────────────────

def generate_quiz(chunks: list, n: int = 5) -> list:
    """
    Auto-generate Multiple Choice Questions from PDF content.
    Uses heuristics — no external API required.
    Returns list of {question, options: [A,B,C,D], answer_index, source_page}
    """
    from utils import _extract_key_phrases
    questions = []
    seen_q = set()

    # Build distractor pool from document concepts
    full_text = " ".join(c["text"] for c in chunks[:200])
    all_phrases = _extract_key_phrases(full_text, max_phrases=80)
    distractor_pool = [p.title() for p in all_phrases]

    for chunk in chunks:
        if len(questions) >= n:
            break
        sents = sent_tokenize(chunk["text"])
        for sent in sents:
            if len(questions) >= n:
                break
            sent = sent.strip()
            if len(sent) < 20 or len(sent) > 400:
                continue

            # Find definition-style sentences: "X is/refers to/means Y"
            match = None
            for kw, q_verb in [
                (' is ', 'is'),
                (' are ', 'are'),
                (' refers to ', 'does ... refer to'),
                (' means ', 'does ... mean'),
                (' consists of ', 'does ... consist of'),
                (' is defined as ', 'is ... defined as'),
            ]:
                idx = sent.lower().find(kw)
                if idx > 5:
                    subject = sent[:idx].strip()
                    predicate = sent[idx + len(kw):].strip()
                    # Validate subject
                    subject_clean = re.sub(r'^(the|a|an)\s+', '', subject, flags=re.I).strip()
                    words = subject_clean.split()
                    if (2 <= len(words) <= 5
                            and subject_clean.lower() not in {'it', 'this', 'that', 'they', 'these'}
                            and len(predicate) > 10):
                        match = (subject_clean, predicate, kw.strip())
                        break

            if not match:
                continue

            subject, predicate, rel = match
            q_text = f"What does **{subject.title()}** {rel.replace(' ', ' ')}?"
            q_key = q_text.lower()[:60]
            if q_key in seen_q:
                continue
            seen_q.add(q_key)

            # Correct answer: the predicate (trimmed)
            correct = predicate[:120].rstrip('.,;')
            if len(correct) < 8:
                continue

            # Distractors: 3 random different concept phrases
            candidates = [d for d in distractor_pool if d.lower() != subject.lower() and d.lower() not in correct.lower()]
            random.shuffle(candidates)
            used = candidates[:3]
            if len(used) < 3:
                continue

            options = [correct] + used
            random.shuffle(options)
            answer_index = options.index(correct)

            questions.append({
                "question":     q_text,
                "options":      options,
                "answer_index": answer_index,
                "source_page":  chunk.get("page", "?")
            })

    return questions[:n]


# ─── Smart Outline Builder ────────────────────────────────────────────────────

_HEADER_PAT_NUMBERED = re.compile(r'^(\d+(?:\.\d+)*)\s+([A-Z].{4,80})$')
_HEADER_PAT_MODULE   = re.compile(r'^(module|chapter|section|unit|part|appendix)\s+(\w+.{0,60})$', re.I)
_HEADER_PAT_ALLCAPS  = re.compile(r'^([A-Z][A-Z\s\-/:,]{4,60}[A-Z])$')

def generate_outline(chunks: list) -> list:
    """
    Build a hierarchical outline of the document based on structural cues.
    Returns list of {level: int, title: str, page: int}
    """
    outline = []
    seen = set()

    for chunk in chunks:
        for line in chunk["text"].split('\n'):
            line = line.strip()
            if not line or len(line) < 4 or len(line) > 120:
                continue
            key = line.lower()[:70]
            if key in seen:
                continue

            level = None
            if _HEADER_PAT_NUMBERED.match(line):
                num_part = line.split()[0]
                level = min(num_part.count('.') + 1, 3)
            elif _HEADER_PAT_MODULE.match(line):
                level = 1
            elif _HEADER_PAT_ALLCAPS.match(line):
                level = 1

            if level is not None:
                seen.add(key)
                outline.append({
                    "level": level,
                    "title": line[:100],
                    "page":  chunk.get("page", "?")
                })

    return outline[:100]


# ─── Glossary Builder ─────────────────────────────────────────────────────────

_DEF_PATTERNS = [
    re.compile(r'^(.{3,50}?)\s+(?:is defined as|is known as|refers to|denotes|represents)\s+(.{10,250})', re.I),
    re.compile(r'^(.{3,50}?)\s+is\s+(?:a|an|the)\s+(.{10,200}[^?])', re.I),
    re.compile(r'^([A-Z][a-zA-Z\s]{3,50}?):\s+(.{15,250})$'),
]

def generate_glossary(chunks: list) -> list:
    """
    Auto-extract glossary of key terms and definitions.
    Returns list of {term: str, definition: str, page: int}
    """
    glossary = []
    seen_terms = set()

    for chunk in chunks:
        sents = sent_tokenize(chunk["text"])
        for sent in sents:
            sent = sent.strip()
            if len(sent) < 15 or len(sent) > 400:
                continue
            for pat in _DEF_PATTERNS:
                m = pat.match(sent)
                if m:
                    term = m.group(1).strip().rstrip('.,;')
                    defn = m.group(2).strip().rstrip('.,;')
                    term = re.sub(r'^(the|a|an)\s+', '', term, flags=re.I).strip()
                    t_lower = term.lower()
                    if (len(term) < 3 or len(term) > 60
                            or len(defn) < 10
                            or t_lower in _STOP
                            or t_lower in seen_terms
                            or not re.search(r'[A-Za-z]{3,}', term)):
                        continue
                    seen_terms.add(t_lower)
                    glossary.append({
                        "term":       term.title(),
                        "definition": defn[:300],
                        "page":       chunk.get("page", "?")
                    })
                    break  # one pattern per sentence

    glossary.sort(key=lambda x: x["term"].lower())
    return glossary[:120]
