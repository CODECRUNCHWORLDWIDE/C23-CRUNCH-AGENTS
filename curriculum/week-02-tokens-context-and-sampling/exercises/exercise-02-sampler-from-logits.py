#!/usr/bin/env python3
# Exercise 2 — A sampler from logits (temperature / top-k / top-p / min-p in NumPy)
#
# Goal: Implement, from RAW logits, the sampling pipeline Lecture 2 describes:
#         penalties -> temperature -> truncation (top-k / top-p / min-p) -> softmax -> draw
#       so the knobs stop being magic. You write the four truncation/scaling
#       transforms; a verification harness then PROVES each one does what the
#       lecture said (e.g. top-p keeps the smallest set whose cumulative prob >= p).
#
# Estimated time: 60 minutes. Runnable.
#
# WHY THIS MATTERS
#
#   Lecture 2 claimed temperature is not "creativity" and top-p is not
#   "diversity" — each is a transform on a probability distribution over the
#   next token. The only way to truly believe that is to implement the
#   transforms and watch the distribution change. After this file, "the model
#   is too random" is a knob you understand, not a mystery you fear.
#
# HOW TO USE THIS FILE
#
#   Standalone. Only needs NumPy:
#
#       pip install numpy
#       python3 exercise-02-sampler-from-logits.py
#
#   It builds a fixed toy logit vector (so results are reproducible), runs your
#   transforms, and prints a verification report. Every check should print PASS.
#
# THE TODOs
#
#   Four gaps are marked "# TODO N:". Fill them to complete the sampler.
#   Everything else — the harness, softmax, temperature, the draw — is done.
#
# ACCEPTANCE CRITERIA
#
#   [ ] All four transforms implemented; the harness prints PASS for every check.
#   [ ] top_k keeps exactly k tokens (the k highest-logit ones).
#   [ ] top_p keeps the SMALLEST set whose cumulative probability >= p, and that
#       set's probability mass is >= p.
#   [ ] min_p keeps exactly the tokens with prob >= min_p * p_max.
#   [ ] The temperature sweep shows the chosen-token distribution FLATTEN as T
#       rises (entropy increases monotonically from T=0.1 to T=1.5).
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import numpy as np

# A fixed, hand-built logit vector over a tiny 8-token "vocabulary". Using a
# fixed vector makes every check below reproducible. Token 3 is the clear
# favorite; tokens 0 and 7 are near-zero probability.
TOY_LOGITS = np.array([0.1, 1.0, 2.0, 4.0, 1.5, 0.5, 2.5, -1.0], dtype=np.float64)


# -----------------------------------------------------------------------------
# Provided: softmax (with the subtract-the-max numerical-stability trick) and
# temperature scaling. Study these — your transforms follow the same style.
# -----------------------------------------------------------------------------
def softmax(logits: np.ndarray) -> np.ndarray:
    """logits -> probability distribution. Shift by max for numerical stability."""
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by 1/T before softmax. T<1 sharpens, T>1 flattens, T->0 greedy."""
    if temperature <= 0:
        out = np.full_like(logits, -np.inf)
        out[int(np.argmax(logits))] = 0.0  # one survivor -> argmax
        return out
    return logits / temperature


# -----------------------------------------------------------------------------
# YOUR WORK: the three truncation transforms. Each takes a logit vector and
# returns a logit vector with the DROPPED tokens set to -inf (so softmax gives
# them probability 0). Do NOT renormalize here — softmax does that at the draw.
# -----------------------------------------------------------------------------
def apply_top_k(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep the k highest-logit tokens; set the rest to -inf."""
    if k <= 0 or k >= logits.size:
        return logits
    # TODO 1: find the value of the k-th largest logit, then set every logit
    #         STRICTLY BELOW that value to -inf. Hint: np.partition(logits, -k)[-k]
    #         gives the k-th largest value. Copy the array first so you don't
    #         mutate the caller's logits.
    #         Replace the line below.
    out = logits.copy()
    return out


def apply_top_p(logits: np.ndarray, p: float) -> np.ndarray:
    """Keep the smallest set of tokens whose cumulative probability >= p (nucleus)."""
    if not (0.0 < p < 1.0):
        return logits
    probs = softmax(logits)
    order = np.argsort(probs)[::-1]            # indices, most-probable first
    cumulative = np.cumsum(probs[order])
    # TODO 2: compute how many tokens to KEEP. You want the smallest prefix of
    #         `order` whose cumulative probability reaches p — i.e. keep up to
    #         and INCLUDING the token that crosses p (so the kept mass is >= p
    #         and the set is never empty). Hint: np.searchsorted(cumulative, p)
    #         returns the index where p would be inserted; +1 includes the
    #         crossing token. Set `cutoff` accordingly.
    cutoff = len(order)  # replace this: currently keeps everything
    keep = order[:cutoff]
    out = np.full_like(logits, -np.inf)
    out[keep] = logits[keep]
    return out


def apply_min_p(logits: np.ndarray, min_p: float) -> np.ndarray:
    """Keep tokens whose probability >= min_p * (probability of the top token)."""
    if not (0.0 < min_p < 1.0):
        return logits
    probs = softmax(logits)
    # TODO 3: compute the threshold = min_p * (max probability), then set every
    #         logit whose probability is BELOW the threshold to -inf. Copy first.
    #         Replace the two lines below.
    threshold = 0.0
    out = logits.copy()
    return out


def sample_next(logits: np.ndarray, rng: np.random.Generator, *,
                temperature: float = 1.0, top_k: int = 0,
                top_p: float = 1.0, min_p: float = 0.0,
                generated_ids: list[int] | None = None,
                repetition_penalty: float = 1.0) -> int:
    """The full pipeline: penalties -> temperature -> truncation -> softmax -> draw."""
    z = logits.astype(np.float64).copy()
    if generated_ids and repetition_penalty != 1.0:
        for tid in set(generated_ids):
            z[tid] = z[tid] / repetition_penalty if z[tid] > 0 else z[tid] * repetition_penalty
    z = apply_temperature(z, temperature)
    if top_k:
        z = apply_top_k(z, top_k)
    if top_p < 1.0:
        z = apply_top_p(z, top_p)
    if min_p > 0.0:
        z = apply_min_p(z, min_p)
    probs = softmax(z)
    # TODO 4: draw one token id from `probs`. Hint: rng.choice(len(probs), p=probs).
    #         Return it as an int. Replace the line below.
    return int(np.argmax(probs))


# -----------------------------------------------------------------------------
# Verification harness. You do not edit below this line. If your transforms are
# correct, every check prints PASS.
# -----------------------------------------------------------------------------
def entropy(probs: np.ndarray) -> float:
    """Shannon entropy in nats; higher = flatter distribution."""
    p = probs[probs > 0]
    return float(-(p * np.log(p)).sum())


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    return condition


def verify() -> bool:
    ok = True
    print("=" * 70)
    print("VERIFYING THE SAMPLER")
    print("=" * 70)

    # --- top-k keeps exactly k tokens -------------------------------------
    print("\ntop-k:")
    for k in (1, 3, 5):
        out = apply_top_k(TOY_LOGITS, k)
        survivors = int(np.isfinite(out).sum())
        ok &= check(f"k={k} keeps exactly {k} tokens", survivors == k,
                    f"survivors={survivors}")
    # the survivors must be the k HIGHEST-logit tokens
    out3 = apply_top_k(TOY_LOGITS, 3)
    kept = set(np.where(np.isfinite(out3))[0].tolist())
    expected = set(np.argsort(TOY_LOGITS)[-3:].tolist())
    ok &= check("k=3 keeps the 3 highest-logit tokens", kept == expected,
                f"kept={sorted(kept)} expected={sorted(expected)}")

    # --- top-p keeps the smallest set with cumulative prob >= p -----------
    print("\ntop-p (nucleus):")
    for p in (0.5, 0.8, 0.95):
        out = apply_top_p(TOY_LOGITS, p)
        kept_mass = softmax(TOY_LOGITS)[np.isfinite(out)].sum()
        n_kept = int(np.isfinite(out).sum())
        ok &= check(f"p={p}: kept mass >= p", kept_mass >= p - 1e-9,
                    f"kept_mass={kept_mass:.3f} over {n_kept} tokens")
    # minimality: dropping the last kept token would fall BELOW p
    out_p = apply_top_p(TOY_LOGITS, 0.8)
    probs = softmax(TOY_LOGITS)
    kept_idx = np.where(np.isfinite(out_p))[0]
    kept_sorted = kept_idx[np.argsort(probs[kept_idx])]  # least->most probable kept
    mass_without_smallest = probs[kept_sorted[1:]].sum() if len(kept_sorted) > 1 else 0.0
    ok &= check("p=0.8: set is minimal (drop one -> below p)",
                mass_without_smallest < 0.8,
                f"mass without smallest kept = {mass_without_smallest:.3f}")

    # --- min-p keeps tokens with prob >= min_p * p_max --------------------
    print("\nmin-p:")
    for mp in (0.1, 0.3, 0.5):
        out = apply_min_p(TOY_LOGITS, mp)
        thresh = mp * probs.max()
        expected_keep = set(np.where(probs >= thresh)[0].tolist())
        got_keep = set(np.where(np.isfinite(out))[0].tolist())
        ok &= check(f"min_p={mp}: keeps prob >= {thresh:.3f}",
                    expected_keep == got_keep,
                    f"kept={len(got_keep)} tokens")

    # --- temperature flattens the distribution ----------------------------
    print("\ntemperature (entropy should rise with T):")
    entropies = []
    for T in (0.1, 0.5, 1.0, 1.5):
        e = entropy(softmax(apply_temperature(TOY_LOGITS, T)))
        entropies.append(e)
        print(f"    T={T:<4} entropy={e:.3f} nats")
    rising = all(entropies[i] < entropies[i + 1] for i in range(len(entropies) - 1))
    ok &= check("entropy increases monotonically as T rises", rising,
                f"entropies={[round(e, 3) for e in entropies]}")

    # --- the draw: greedy (T->0) is deterministic, high-T spreads ----------
    print("\nthe draw (empirical, 4000 samples each):")
    rng = np.random.default_rng(0)
    greedy = [sample_next(TOY_LOGITS, rng, temperature=0.01) for _ in range(200)]
    ok &= check("T~0 always picks the argmax token (3)", set(greedy) == {3},
                f"unique picks={sorted(set(greedy))}")
    rng = np.random.default_rng(1)
    hot = [sample_next(TOY_LOGITS, rng, temperature=1.5) for _ in range(4000)]
    distinct_hot = len(set(hot))
    ok &= check("T=1.5 spreads picks across many tokens", distinct_hot >= 6,
                f"distinct tokens chosen={distinct_hot}")

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED — fix the TODOs above")
    print("=" * 70)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)


# -----------------------------------------------------------------------------
# Expected output (once all four TODOs are filled correctly)
# -----------------------------------------------------------------------------
#
# ======================================================================
# VERIFYING THE SAMPLER
# ======================================================================
#
# top-k:
#   [PASS] k=1 keeps exactly 1 tokens  (survivors=1)
#   [PASS] k=3 keeps exactly 3 tokens  (survivors=3)
#   [PASS] k=5 keeps exactly 5 tokens  (survivors=5)
#   [PASS] k=3 keeps the 3 highest-logit tokens  (kept=[2, 3, 6] expected=[2, 3, 6])
#
# top-p (nucleus):
#   [PASS] p=0.5: kept mass >= p  (kept_mass=0.646 over 1 tokens)
#   [PASS] p=0.8: kept mass >= p  (kept_mass=0.878 over 3 tokens)
#   [PASS] p=0.95: kept mass >= p  (kept_mass=0.963 over 5 tokens)
#   [PASS] p=0.8: set is minimal (drop one -> below p)  (mass without smallest kept = 0.790)
#
# min-p:
#   [PASS] min_p=0.1: keeps prob >= 0.065  (kept=3 tokens)
#   [PASS] min_p=0.3: keeps prob >= 0.194  (kept=1 tokens)
#   [PASS] min_p=0.5: keeps prob >= 0.323  (kept=1 tokens)
#
# temperature (entropy should rise with T):
#     T=0.1  entropy=0.000 nats
#     T=0.5  entropy=0.336 nats
#     T=1.0  entropy=1.198 nats
#     T=1.5  entropy=1.624 nats
#   [PASS] entropy increases monotonically as T rises  (entropies=[0.0, 0.336, 1.198, 1.624])
#
# the draw (empirical, 4000 samples each):
#   [PASS] T~0 always picks the argmax token (3)  (unique picks=[3])
#   [PASS] T=1.5 spreads picks across many tokens  (distinct tokens chosen=8)
#
# ======================================================================
# ALL CHECKS PASSED
# ======================================================================
#
# THE LESSON: every knob is a transform on the distribution. top-k is a fixed
# count, top-p is an adaptive cumulative-mass cut, min-p scales with the model's
# confidence, and temperature flattens or sharpens before any of them. The draw
# is the only randomness — seed it and it's reproducible. "Too random" is a knob.
# -----------------------------------------------------------------------------
