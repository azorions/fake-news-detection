"""Label convention used everywhere: 1 is the positive class, "unreliable"."""

RELIABLE = 0
UNRELIABLE = 1

LABEL_NAMES: dict[int, str] = {RELIABLE: "reliable", UNRELIABLE: "unreliable"}
