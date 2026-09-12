"""Instruction -> conditioning vector."""
from __future__ import annotations

import os

# never download models on this machine (government rule: no Chinese/CCP-affiliated models)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from .instructions import ALL_LABELS, ALL_SENTENCES, CLASSES  # noqa: E402

from .paths import CACHE as CACHE_DIR  # noqa: E402

NLI_AXES = {
    "clearance": ("The drone should fly through wide open spaces with plenty of clearance.",
                  "The drone should squeeze through narrow, tight gaps."),
    "altitude": ("The drone should fly high, near the ceiling.",
                 "The drone should fly low, near the floor."),
    "level": ("The drone should avoid changing its altitude.",
              "The drone should freely climb and descend."),
    "climb_site": ("The drone should only climb where there is plenty of open space.",
                   "The drone may climb anywhere, even close to obstacles."),
    "descend_site": ("The drone should only descend where there is plenty of open space.",
                     "The drone may descend anywhere, even close to obstacles."),
    "directness": ("The drone should take the shortest possible route.",
                   "The drone should accept a longer route."),
}
POLES = [h for pair in NLI_AXES.values() for h in pair]

PREMISE = "The drone is instructed: {}"
NLI_CACHE_VERSION = "poles-v2"


def _is_text(x) -> bool:
    """True for a sentence or a sequence of sentences; False for indices."""
    if isinstance(x, str):
        return True
    try:
        return len(x) > 0 and isinstance(x[0], str)
    except TypeError:
        return False


def logodds(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    return (np.log(p + eps) - np.log(1.0 - p + eps)).astype(np.float32)


class OneHotEncoder:
    def __init__(self, n_classes: int = len(CLASSES)):
        self.dim = n_classes
        self.table = np.eye(n_classes, dtype=np.float32)[ALL_LABELS]

    def __call__(self, idx):
        return self.table[idx]


class EmbeddingEncoder:
    """Frozen sentence embedding + PCA fitted on the TRAIN sentences only."""

    def __init__(self, model_name: str, train_idx, out_dim: int = 16):
        self.model_name = model_name
        raw = self._raw(model_name)
        X = raw[train_idx]
        self.mean = X.mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(X - self.mean, full_matrices=False)
        self.comps = vt[:out_dim]
        z = (X - self.mean) @ self.comps.T
        self.scale = z.std(0, keepdims=True) + 1e-6
        self.dim = out_dim
        self.table = self._project(raw)
        self._enc = None

    @staticmethod
    def _raw(model_name):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / ("emb_" + model_name.replace("/", "__") + ".npz")
        if path.exists():
            z = np.load(path, allow_pickle=True)
            if list(z["sentences"]) == ALL_SENTENCES:
                return z["emb"]
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer(model_name).encode(
            ALL_SENTENCES, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False).astype(np.float32)
        np.savez(path, emb=emb, sentences=np.asarray(ALL_SENTENCES, dtype=object))
        return emb

    def _project(self, raw):
        return (((raw - self.mean) @ self.comps.T) / self.scale).astype(np.float32)

    def encode_text(self, sentences):
        if self._enc is None:
            from sentence_transformers import SentenceTransformer
            self._enc = SentenceTransformer(self.model_name)
        raw = self._enc.encode(list(sentences), normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
        return self._project(raw.astype(np.float32))

    def __call__(self, x):
        """Corpus indices -> table rows, or raw sentences -> fresh encodings."""
        if _is_text(x):
            return self.encode_text([x] if isinstance(x, str) else list(x))
        return self.table[x]


class NLIEncoder:
    """Zero-shot entailment against both poles of every preference axis."""

    def __init__(self, model_name: str, device=None, batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.dim = len(POLES)
        self._mdl = None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / ("nli_poles_" + model_name.replace("/", "__") + ".npz")
        probs = None
        if path.exists():
            z = np.load(path, allow_pickle=True)
            if (str(z["version"]) == NLI_CACHE_VERSION
                    and list(z["sentences"]) == ALL_SENTENCES and list(z["poles"]) == POLES):
                probs = z["probs"]
        if probs is None:
            probs = self.pole_probs(ALL_SENTENCES)
            np.savez(path, probs=probs, version=NLI_CACHE_VERSION,
                     sentences=np.asarray(ALL_SENTENCES, dtype=object),
                     poles=np.asarray(POLES, dtype=object))
        self.probs = probs
        self.table = logodds(probs)

    def _load(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._mdl = AutoModelForSequenceClassification.from_pretrained(
            self.model_name).to(dev).eval()
        self._dev = dev
        lab = {k.lower(): v for k, v in self._mdl.config.label2id.items()}
        self._ent = next(v for k, v in lab.items() if k.startswith("entail"))

    def _entail(self, premises, hypothesis):
        import torch
        out = np.zeros(len(premises), np.float32)
        with torch.no_grad():
            for i in range(0, len(premises), self.batch_size):
                chunk = premises[i:i + self.batch_size]
                enc = self._tok(chunk, [hypothesis] * len(chunk), return_tensors="pt",
                                padding=True, truncation=True).to(self._dev)
                p = self._mdl(**enc).logits.float().softmax(-1)[:, self._ent]
                out[i:i + len(chunk)] = p.cpu().numpy()
        return out

    def pole_probs(self, sentences) -> np.ndarray:
        """(N, 12) entailment probability of each sentence against each pole."""
        if self._mdl is None:
            self._load()
        prem = [PREMISE.format(s) for s in sentences]
        return np.stack([self._entail(prem, h) for h in POLES], axis=1)

    def encode_text(self, sentences) -> np.ndarray:
        return logodds(self.pole_probs(sentences))

    @staticmethod
    def signed_coefficients(probs: np.ndarray) -> np.ndarray:
        """(N, 6) interpretable coefficient per axis, in [-1, 1]."""
        p, n = probs[:, 0::2], probs[:, 1::2]
        return ((p - n) / (p + n + 1e-6) * np.maximum(p, n)).astype(np.float32)

    def __call__(self, x):
        """Corpus indices -> table rows, or raw sentences -> fresh encodings."""
        if _is_text(x):
            return self.encode_text([x] if isinstance(x, str) else list(x))
        return self.table[x]


def build_encoder(model_cfg, train_idx):
    if model_cfg.conditioning == "onehot":
        return OneHotEncoder()
    if model_cfg.conditioning == "embedding":
        return EmbeddingEncoder(model_cfg.encoder_model, train_idx, model_cfg.emb_dim)
    if model_cfg.conditioning == "nli":
        return NLIEncoder(model_cfg.nli_model)
    raise ValueError(model_cfg.conditioning)
