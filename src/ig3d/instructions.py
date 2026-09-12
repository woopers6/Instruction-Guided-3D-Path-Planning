"""Instruction corpus: 128 sentences over 8 classes, 16 each."""
from __future__ import annotations

import numpy as np

TIER = {
    "open": 1, "tight": 1, "low": 1, "high": 1, "shortest": 1,
    "level": 2,
    "climb_open": 3, "descend_open": 3,
}
CLASSES = tuple(TIER)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

OPEN = [
    "Aim for wider paths.", "Prefer the roomier route.", "Stay in open space.",
    "Give obstacles a wide berth.", "Keep plenty of clearance on all sides.",
    "Choose the widest opening you can find.", "Avoid tight squeezes.",
    "Fly through the most open part of the volume.", "Steer clear of narrow gaps.",
    "Keep a safe margin from every surface.", "Take the least confined route.",
    "Maximise clearance along the way.", "Do not cut it close to anything.",
    "Use the large openings, not the small ones.", "Leave a lot of space around you.",
    "Prefer open volume over cramped shortcuts.",
]

TIGHT = [
    "Mind small gaps.", "Prefer the narrow passage.", "Squeeze through the tight openings.",
    "Take the constricted route.", "Use the small hole rather than the big one.",
    "Thread through the tightest gap.", "Keep clearance to a minimum.",
    "Go where the space is tightest.", "Slip through the narrow slot.",
    "Choose the smaller of the two openings.", "Fly close to the surfaces.",
    "Take the cramped way through.", "Minimise clearance along the way.",
    "Avoid the wide open parts.", "Pick the pinch point.", "Keep the route tight.",
]

LOW = [
    "Stay low.", "Fly close to the floor.", "Keep a low altitude throughout.",
    "Remain near ground level.", "Do not fly high.", "Hug the deck.",
    "Keep well below the ceiling.", "Operate at low level.",
    "Choose the lowest workable height.", "Stay in the bottom of the volume.",
    "Keep your cruising height small.", "Fly a low profile.",
    "Stick to the lower levels.", "Stay nearer the floor than the ceiling.",
    "Do not rise into the upper space.", "The lower the route, the better.",
]

HIGH = [
    "Stay high.", "Fly well above the clutter.", "Keep a high altitude throughout.",
    "Remain near the top of the volume.", "Do not fly low.", "Hug the ceiling.",
    "Keep well above the floor.", "Operate at high level.",
    "Choose the highest workable height.", "Stay in the upper part of the space.",
    "Keep your cruising height large.", "Fly a high profile.",
    "Stick to the upper levels.", "Stay nearer the ceiling than the floor.",
    "Do not drop into the lower space.", "The higher the route, the better.",
]

SHORTEST = [
    "Take the shortest path.", "Fly straight to the goal.", "Minimise travel distance.",
    "Do not detour.", "Take the most direct route.", "Distance is all that matters.",
    "Get there as quickly as you can.", "Optimise for length only.",
    "Reach the goal in the fewest metres.", "Do not go out of your way.",
    "Cut straight across.", "Be direct.", "Waste no distance.",
    "Ignore clearance and height, just be short.", "Keep the trip as short as possible.",
    "Route directly to the destination.",
]

LEVEL = [
    "Keep it level.", "Minimise vertical manoeuvring.",
    "Avoid changing altitude if you can.", "Hold a steady height.",
    "Going up and down costs more than going across.",
    "Prefer horizontal travel to vertical travel.",
    "Keep the vertical profile as flat as you can.",
    "Do not bob up and down.", "Fly a flat trajectory.",
    "Height changes are expensive, so avoid them.",
    "Stay on one level as much as possible.",
    "Trade sideways distance for a flatter route.",
    "Avoid altitude changes even if the route gets longer.",
    "Keep the climb and descent totals small.",
    "A level route is worth a detour.", "Do not waste energy going vertical.",
]

CLIMB_OPEN = [
    "Do your climbing in the open, not in the clutter.",
    "Only gain height where there is room around you.",
    "Do not ascend near obstacles.",
    "Climb in clear air, never inside a gap.",
    "Gain your altitude before you enter the tight parts.",
    "Rising close to a surface is dangerous, so rise in the open.",
    "Never climb while squeezing through something.",
    "If you must go up, do it where there is space.",
    "Avoid upward motion in confined areas.",
    "Get your height in the open volume, then transit.",
    "Climbing near walls is not acceptable.",
    "Ascend only where clearance is generous.",
    "Do not gain altitude inside a narrow passage.",
    "Pick somewhere roomy to make your climb.",
    "Upward legs belong in open space.",
    "Keep climbs away from the clutter.",
]

DESCEND_OPEN = [
    "Do your descending in the open, not through the gaps.",
    "Only lose height where there is room around you.",
    "Do not descend near obstacles.",
    "Drop down in clear air, never inside a gap.",
    "Lose your altitude after you leave the tight parts.",
    "Sinking close to a surface is dangerous, so sink in the open.",
    "Never descend while squeezing through something.",
    "If you must go down, do it where there is space.",
    "Avoid downward motion in confined areas.",
    "Transit first, then give up your height in the open volume.",
    "Descending near walls is not acceptable.",
    "Descend only where clearance is generous.",
    "Do not lose altitude inside a narrow passage.",
    "Pick somewhere roomy to make your descent.",
    "Downward legs belong in open space.",
    "Keep descents away from the clutter.",
]

BY_CLASS = {
    "open": OPEN, "tight": TIGHT, "low": LOW, "high": HIGH, "shortest": SHORTEST,
    "level": LEVEL, "climb_open": CLIMB_OPEN, "descend_open": DESCEND_OPEN,
}

ALL_SENTENCES: list[str] = []
ALL_LABELS_LIST: list[int] = []
for _c in CLASSES:
    for _s in BY_CLASS[_c]:
        ALL_SENTENCES.append(_s)
        ALL_LABELS_LIST.append(CLASS_TO_IDX[_c])
ALL_LABELS = np.asarray(ALL_LABELS_LIST)

assert all(len(v) == 16 for v in BY_CLASS.values()), \
    {k: len(v) for k, v in BY_CLASS.items()}
assert len(set(ALL_SENTENCES)) == len(ALL_SENTENCES), "duplicate sentence"
assert len(ALL_SENTENCES) == 128


def split_indices(n_heldout_per_class: int = 4, seed: int = 0):
    """Per-class train/held-out split over indices into ALL_SENTENCES."""
    rng = np.random.default_rng(seed)
    train, held = [], []
    for ci in range(len(CLASSES)):
        idx = rng.permutation(np.flatnonzero(ALL_LABELS == ci))
        held.extend(idx[:n_heldout_per_class].tolist())
        train.extend(idx[n_heldout_per_class:].tolist())
    return np.sort(np.asarray(train)), np.sort(np.asarray(held))


def classes_in_tier(tier: int) -> tuple[str, ...]:
    return tuple(c for c in CLASSES if TIER[c] == tier)
