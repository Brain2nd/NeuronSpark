"""Synthetic fact corpus for the level-2 consolidation experiment (``docs/experiments/consolidation_level2.md``).

Facts are (entity, attribute, value) triples with invented entity names (two invented syllabic words, never in the
training data) and values from closed sets of common words whose " value" is one token of the architecture's tokenizer.
A session document mentions each fact ``m`` times in varied templates, interleaved with distractor text from the dev
stream, and ends with <|endoftext|>.  Probes are cloze prompts whose answer is the value's token: the same template as
a mention (memorisation), an unseen template (rephrase), and a two-hop template that goes through another fact of the
same entity (composition).  All probes of a session use a disjoint entity set from every other session (``seed``).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EOT = 151643

ATTRIBUTES: Dict[str, Dict[str, Sequence[str]]] = {
    "city": {
        "values": ("Paris", "London", "Rome", "Berlin", "Madrid", "Vienna", "Prague", "Dublin", "Oslo", "Lisbon",
                   "Athens", "Boston", "Chicago", "Toronto", "Sydney", "Tokyo", "Cairo", "Lima", "Moscow", "Delhi"),
        "mention": ("{E} lives in {V}.", "The home of {E} is in {V}.", "{E} has lived in {V} for many years.",
                    "If you want to visit {E}, go to {V}."),
        "probe": ("{E} lives in",),
        "rephrase": ("The city where {E} lives is", "{E} makes a home in the city of"),
    },
    "colour": {
        "values": ("red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white", "grey",
                   "gold", "silver"),
        "mention": ("The favourite colour of {E} is {V}.", "{E} likes the colour {V} best.",
                    "Everything {E} owns is {V}.", "{E} always wears {V}."),
        "probe": ("The favourite colour of {E} is",),
        "rephrase": ("Ask {E} for a colour and the answer is", "The colour {E} loves most is"),
    },
    "job": {
        "values": ("doctor", "teacher", "farmer", "baker", "painter", "nurse", "driver", "singer", "lawyer", "pilot",
                   "cook", "writer", "dancer", "builder", "soldier", "sailor", "judge", "tailor"),
        "mention": ("{E} works as a {V}.", "By trade, {E} is a {V}.", "{E} has been a {V} since school.",
                    "People call {E} the {V}."),
        "probe": ("{E} works as a",),
        "rephrase": ("The job of {E} is to be a", "For a living, {E} is a"),
    },
    "pet": {
        "values": ("dog", "cat", "rabbit", "horse", "parrot", "goat", "duck", "pig", "hamster", "turtle", "pony",
                   "lizard", "frog", "mouse", "snake"),
        "mention": ("{E} has a pet {V}.", "{E} keeps a {V} at home.", "The {V} of {E} sleeps by the door.",
                    "{E} walks with a {V} every morning."),
        "probe": ("{E} has a pet",),
        "rephrase": ("The animal that {E} keeps is a", "At home, {E} looks after a"),
    },
}
# two-hop: the value of attribute B of the entity whose attribute A is V_A
TWO_HOP = {
    ("city", "pet"): "The pet of the person who lives in {VA} is a",
    ("city", "job"): "The person who lives in {VA} works as a",
    ("job", "colour"): "The favourite colour of the {VA} is",
    ("pet", "city"): "The owner of the {VA} lives in",
}
SYLLABLES = ("ba", "be", "bo", "da", "de", "do", "fa", "fe", "ga", "go", "ka", "ke", "ko", "la", "le", "lo", "ma", "me",
             "mo", "na", "ne", "no", "pa", "pe", "po", "ra", "re", "ro", "sa", "se", "so", "ta", "te", "to", "va", "ve",
             "vo", "za", "ze", "zo")


@dataclass
class Fact:
    entity: str
    attribute: str
    value: str
    answer_id: int                                    # the first token of " value"


@dataclass
class Session:
    index: int
    facts: List[Fact]
    doc_ids: np.ndarray                               # the session document (1-D int64), ends with EOT
    probes: List[Dict] = field(default_factory=list)  # {"kind", "fact", "prompt_ids", "answer_id", ...}


def _name(rng: random.Random, used: set) -> str:
    while True:
        first = "".join(rng.choice(SYLLABLES) for _ in range(rng.choice((2, 3)))).capitalize()
        last = "".join(rng.choice(SYLLABLES) for _ in range(rng.choice((2, 3)))).capitalize()
        n = f"{first} {last}"
        if n not in used:
            used.add(n)
            return n


def single_token_values(tok) -> Dict[str, List[Tuple[str, int]]]:
    """Per attribute, the values whose ' value' is one token: [(value, id)]."""
    out = {}
    for a, spec in ATTRIBUTES.items():
        vs = []
        for v in spec["values"]:
            ids = tok.encode(" " + v, add_special_tokens=False).ids
            if len(ids) == 1:
                vs.append((v, ids[0]))
        if len(vs) < 6:
            raise RuntimeError(f"attribute {a}: too few single-token values ({len(vs)})")
        out[a] = vs
    return out


def build_session(index: int, tok, dev_stream: np.ndarray, *, n_entities: int = 16, mentions: int = 2,
                  distractor_tokens: int = 64, seed: int = 0, values=None) -> Session:
    """One session: ``n_entities`` entities, each with every attribute (facts = 4 x entities), each fact mentioned
    ``mentions`` times; between sentences a distractor window of ``distractor_tokens`` dev tokens; EOT at the end.
    Probes: same-template, rephrase, two-hop (for each entity and each two-hop pair)."""
    rng = random.Random(seed * 1000 + index)
    values = values or single_token_values(tok)
    used = set()
    entities = [_name(rng, used) for _ in range(n_entities)]
    facts: List[Fact] = []
    for e in entities:
        for a in ATTRIBUTES:
            v, vid = rng.choice(values[a])
            facts.append(Fact(e, a, v, vid))
    # distinct values per attribute inside a session make the two-hop probes unambiguous
    for a in ATTRIBUTES:
        pool = values[a][:]
        rng.shuffle(pool)
        if len(pool) < n_entities:
            pool = (pool * (n_entities // len(pool) + 1))
        for i, f in enumerate([f for f in facts if f.attribute == a]):
            f.value, f.answer_id = pool[i % len(pool)]
    sentences = []
    for f in facts:
        temps = list(ATTRIBUTES[f.attribute]["mention"])
        rng.shuffle(temps)
        for i in range(mentions):
            sentences.append(temps[i % len(temps)].format(E=f.entity, V=f.value))
    rng.shuffle(sentences)
    ids: List[int] = []
    nd = len(dev_stream)
    for s in sentences:
        if distractor_tokens:
            p = rng.randrange(0, nd - distractor_tokens)
            ids += [int(x) for x in dev_stream[p:p + distractor_tokens]]
            ids.append(tok.encode("\n", add_special_tokens=False).ids[0])
        ids += tok.encode(s + " ", add_special_tokens=False).ids
    ids.append(EOT)
    sess = Session(index, facts, np.asarray(ids, dtype=np.int64))
    by_entity = {e: {f.attribute: f for f in facts if f.entity == e} for e in entities}
    for f in facts:
        spec = ATTRIBUTES[f.attribute]
        sess.probes.append({"kind": "same", "fact": f, "prompt": spec["probe"][0].format(E=f.entity)})
        sess.probes.append({"kind": "rephrase", "fact": f, "prompt": rng.choice(spec["rephrase"]).format(E=f.entity)})
    for e in entities:
        for (a, b), temp in TWO_HOP.items():
            fa, fb = by_entity[e][a], by_entity[e][b]
            sess.probes.append({"kind": "twohop", "fact": fb, "prompt": temp.format(VA=fa.value)})
    for p in sess.probes:
        p["prompt_ids"] = [EOT] + tok.encode(p["prompt"], add_special_tokens=False).ids
        p["answer_id"] = p["fact"].answer_id
    return sess


def invariant_items(dev_stream: np.ndarray, n: int, seq: int, seed: int) -> List[Tuple[int, int]]:
    """``n`` (start, length) dev windows of ``seq`` + 1 tokens whose last token is a prediction target (the base model's
    correct ones become the invariant set)."""
    rng = np.random.default_rng(seed)
    starts = rng.choice(len(dev_stream) - seq - 1, size=n, replace=False)
    return [(int(s), seq) for s in starts]
