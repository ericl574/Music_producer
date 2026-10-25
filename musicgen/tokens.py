"""01 — A fixed, reversible vocabulary of musical events.

Each note is represented as five tokens:
    STEP_4 ROLE_lead PITCH_72 DUR_2 VEL_6
STEP counts 16th notes within a 4/4 bar. BAR begins a new bar.

Why fixed vocabulary? Pretraining, fine-tuning, and inference MUST use
identical integer -> event mappings, otherwise pretrained weights make no sense.
"""

ROLES = ("lead", "chords", "pad", "bass", "arp", "drums", "other")
SPECIAL = ("PAD", "BOS", "EOS", "BAR")
VOCAB = (
    list(SPECIAL)
    + [f"STEP_{i}" for i in range(16)]
    + [f"ROLE_{r}" for r in ROLES]
    + [f"PITCH_{i}" for i in range(128)]
    + [f"DUR_{i}" for i in range(1, 33)]
    + [f"VEL_{i}" for i in range(1, 9)]
)
TOKEN_TO_ID = {token: idx for idx, token in enumerate(VOCAB)}
ID_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_ID.items()}
PAD = TOKEN_TO_ID["PAD"]
BOS = TOKEN_TO_ID["BOS"]
EOS = TOKEN_TO_ID["EOS"]
BAR = TOKEN_TO_ID["BAR"]


def token_id(name: str) -> int:
    return TOKEN_TO_ID[name]


def decode_ids(ids: list[int]) -> list[str]:
    return [ID_TO_TOKEN[int(i)] for i in ids]


def token_ids(prefix: str, values) -> list[int]:
    return [TOKEN_TO_ID[f"{prefix}_{v}"] for v in values]
