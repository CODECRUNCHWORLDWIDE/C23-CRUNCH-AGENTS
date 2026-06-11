#!/usr/bin/env python3
# Exercise 3 — The LiteLLM router and failover (one client, many backends)
#
# Goal: Stand up the routing layer the capstone uses. You will configure a
#       LiteLLM-style router with (a) a self-hosted backend alias and (b) a
#       vendor fallback, then PROVE failover by pointing the primary at a dead
#       backend and watching the request spill to the fallback — with the
#       CALLER's code unchanged. This is a controlled rehearsal of the week-24
#       chaos drill ("kill a replica; verify the router fails over").
#
# Estimated time: 50 minutes. Runnable.
#
# HOW TO USE THIS FILE
#
#   This file implements a SMALL, self-contained router (the same routing LOGIC
#   LiteLLM's proxy implements: try primary backend(s), on failure fall back to
#   the next) so the failover lesson runs WITHOUT a GPU or any real network.
#   The real tool is `litellm --config config.yaml`; a sample config.yaml is
#   printed by --show-config so you can run the production version too.
#
#       python3 exercise-03-litellm-router.py            # runs the failover demo
#       python3 exercise-03-litellm-router.py --show-config   # prints litellm config.yaml
#
#   To run it for REAL against a live vLLM + a vendor:
#       export ANTHROPIC_API_KEY=sk-ant-...
#       litellm --config config.yaml --port 4000
#       # then point the openai client at http://localhost:4000/v1 with model="local-14b"
#
# ACCEPTANCE CRITERIA
#
#   [ ] A router resolves a model ALIAS to one of N backends (decoupling caller
#       from backend URL).
#   [ ] When all primary backends fail (health check / call error), the request
#       FALLS OVER to the vendor fallback, and the caller code does not change.
#   [ ] The demo prints which backend served each request, including the
#       failover case.
#   [ ] You can explain how this maps to the capstone's "local first, vendor on
#       failure" resilience story and the week-24 chaos drill.
#
# Expected output is at the bottom of the file.

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field


# --- A backend: a thing that can serve a request, or fail ---------------------
@dataclass
class Backend:
    name: str
    kind: str           # "vllm" (self-hosted) or "vendor"
    healthy: bool = True

    def serve(self, prompt: str) -> str:
        """Simulate serving. A real backend would POST to api_base/chat/completions
        (vLLM) or call the vendor SDK. Here we just succeed if healthy, raise if not."""
        if not self.healthy:
            raise ConnectionError(f"{self.name} is down")
        tag = "[LOCAL/vLLM]" if self.kind == "vllm" else "[VENDOR/Claude]"
        return f"{tag} answer to: {prompt[:40]}..."


# --- The router: alias -> [primary backends], with a fallback chain -----------
@dataclass
class Router:
    # alias -> list of primary backends (load-balanced round-robin)
    routes: dict[str, list[Backend]]
    # alias -> ordered fallback aliases to try if ALL primaries fail
    fallbacks: dict[str, list[str]] = field(default_factory=dict)
    _rr: dict[str, itertools.cycle] = field(default_factory=dict)

    def _next_backend(self, alias: str) -> list[Backend]:
        """Return healthy primaries for an alias, in round-robin-rotated order."""
        backends = [b for b in self.routes.get(alias, []) if b.healthy]
        return backends

    def complete(self, model: str, prompt: str) -> tuple[str, str]:
        """Resolve `model` (an alias), try its primaries, then its fallbacks.
        Returns (served_text, backend_name). Raises if everything is down."""
        tried: list[str] = []
        # 1. Try the requested alias's healthy primaries (round-robin).
        for backend in self._round_robin(model):
            try:
                return backend.serve(prompt), backend.name
            except ConnectionError:
                tried.append(backend.name)
        # 2. All primaries failed -> walk the fallback chain.
        for fb_alias in self.fallbacks.get(model, []):
            for backend in self._round_robin(fb_alias):
                try:
                    text, name = backend.serve(prompt), backend.name
                    return f"(failed over) {text}", name
                except ConnectionError:
                    tried.append(backend.name)
        raise RuntimeError(f"all backends failed for '{model}': tried {tried}")

    def _round_robin(self, alias: str) -> list[Backend]:
        # Rotate so load spreads across replicas. Only the HEALTHY ones serve.
        healthy = self._next_backend(alias)
        if not healthy:
            return []
        cyc = self._rr.setdefault(alias, itertools.cycle(range(len(healthy))))
        start = next(cyc) % len(healthy)
        return healthy[start:] + healthy[:start]


# --- The caller: code that NEVER changes regardless of routing ----------------
def caller(router: Router, prompt: str) -> None:
    # The application always asks for the alias "local-14b". It does not know
    # (or care) which replica serves it, or whether it fell over to the vendor.
    text, backend = router.complete(model="local-14b", prompt=prompt)
    print(f"  served by {backend:<14} -> {text}")


SAMPLE_CONFIG = """\
# config.yaml — the real LiteLLM proxy config (run: litellm --config config.yaml --port 4000)
model_list:
  - model_name: local-14b              # two vLLM replicas, same alias -> load-balanced
    litellm_params:
      model: openai/Qwen/Qwen2.5-14B-Instruct
      api_base: http://vllm-a:8000/v1
      api_key: "none"
  - model_name: local-14b
    litellm_params:
      model: openai/Qwen/Qwen2.5-14B-Instruct
      api_base: http://vllm-b:8000/v1
      api_key: "none"
  - model_name: frontier               # vendor model, also the fallback target
    litellm_params:
      model: anthropic/claude-sonnet-4-6
      api_key: os.environ/ANTHROPIC_API_KEY
litellm_settings:
  fallbacks: [{"local-14b": ["frontier"]}]   # all local replicas down -> vendor
  num_retries: 2
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-config", action="store_true",
                    help="print a real litellm config.yaml and exit")
    args = ap.parse_args()
    if args.show_config:
        print(SAMPLE_CONFIG)
        return 0

    # Build the router: two self-hosted replicas under "local-14b", a vendor
    # under "frontier", and a fallback from local-14b -> frontier.
    replica_a = Backend("vllm-replica-a", "vllm")
    replica_b = Backend("vllm-replica-b", "vllm")
    vendor = Backend("claude-sonnet", "vendor")
    router = Router(
        routes={"local-14b": [replica_a, replica_b], "frontier": [vendor]},
        fallbacks={"local-14b": ["frontier"]},
    )

    print("=== Phase 1: both replicas healthy -> served locally, load-balanced ===")
    for i in range(4):
        caller(router, f"request {i}: explain continuous batching")

    print("\n=== Phase 2: kill ONE replica (the week-24 drill) -> survives on the other ===")
    replica_a.healthy = False
    for i in range(4):
        caller(router, f"request {i}: explain PagedAttention")

    print("\n=== Phase 3: kill BOTH replicas -> falls over to the VENDOR ===")
    replica_b.healthy = False
    for i in range(3):
        caller(router, f"request {i}: explain prefix caching")

    print("\nLESSON: the caller asked for 'local-14b' every time and its code never")
    print("changed. The router served locally, survived one replica dying, and")
    print("spilled to the vendor when both were down. That is the capstone's")
    print("local-first / vendor-on-failure resilience, and the week-24 chaos drill.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# Expected output (shape)
# -----------------------------------------------------------------------------
#
# === Phase 1: both replicas healthy -> served locally, load-balanced ===
#   served by vllm-replica-a -> [LOCAL/vLLM] answer to: request 0: explain continuous batching...
#   served by vllm-replica-b -> [LOCAL/vLLM] answer to: request 1: explain continuous batching...
#   served by vllm-replica-a -> ...
#   served by vllm-replica-b -> ...
#
# === Phase 2: kill ONE replica (the week-24 drill) -> survives on the other ===
#   served by vllm-replica-b -> [LOCAL/vLLM] answer to: request 0: explain PagedAttention...
#   served by vllm-replica-b -> ...   (replica-a is out of rotation; no errors)
#
# === Phase 3: kill BOTH replicas -> falls over to the VENDOR ===
#   served by claude-sonnet  -> (failed over) [VENDOR/Claude] answer to: request 0: ...
#   served by claude-sonnet  -> (failed over) ...
#
# LESSON: the caller asked for 'local-14b' every time and its code never changed...
# -----------------------------------------------------------------------------
