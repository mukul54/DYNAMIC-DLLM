"""Decoding-speed accounting for Dynamic-dLLM evaluation runs.

Collects the three numbers reported alongside accuracy in the paper tables:

* **throughput** — decoding tokens / second, aggregated over the whole benchmark
  (total generated tokens divided by total generation wall-time, following the
  Fast-dLLM protocol).
* **tokens per step** — how many tokens the model commits per denoising step,
  i.e. ``gen_length / NFE`` averaged over the benchmark.  Fixed-schedule greedy
  decoding gives exactly ``1.0``; parallel decoding gives more.
* **NFE** — number of model forward passes (denoising steps) per sequence.

Only generation is timed: tokenization, filtering and metric computation are
excluded, so the number reflects decoding speed rather than harness overhead.
"""

import json
import os
from typing import Optional

import torch


def count_generated_tokens(out: torch.Tensor, eos_token_id: Optional[int]) -> int:
    """Number of tokens produced before the first EOS, summed over the batch.

    ``out`` is the generation region only, shape ``(B, gen_length)``.  A dLLM
    always fills every masked position, so the raw count is always
    ``B * gen_length``; the Fast-dLLM protocol instead counts the tokens up to
    the first ``<eos>`` of each sequence, which is what this returns.
    """
    if eos_token_id is None:
        return int(out.numel())

    is_eos = out == eos_token_id
    has_eos = is_eos.any(dim=1)
    # argmax on a bool tensor returns the first True index.
    first_eos = is_eos.to(torch.int64).argmax(dim=1)
    lengths = torch.where(
        has_eos, first_eos, torch.full_like(first_eos, out.shape[1])
    )
    return int(lengths.sum().item())


class SpeedMeter:
    """Accumulates timing / NFE statistics across all batches of a run."""

    def __init__(self, name: str = "run", gen_length: int = 0, eos_token_id: Optional[int] = None):
        self.name = name
        self.gen_length = gen_length
        self.eos_token_id = eos_token_id

        self.total_time = 0.0          # seconds spent inside generate()
        self.total_sequences = 0
        self.total_batches = 0
        self.total_nfe_sequences = 0   # sum over sequences of the NFE of their batch
        self.total_tokens_eos = 0      # tokens before <eos> (Fast-dLLM protocol)
        self.total_tokens_full = 0     # every masked position that was filled

    def record(self, out: torch.Tensor, nfe: int, elapsed: float) -> None:
        """Log one generation call.

        Args:
            out: generated region, shape ``(B, gen_length)``.
            nfe: denoising steps (model forward passes) used for this batch.
            elapsed: wall-clock seconds for the call, measured after
                ``torch.cuda.synchronize()``.
        """
        batch_size, gen_length = out.shape
        self.gen_length = self.gen_length or gen_length

        self.total_time += elapsed
        self.total_sequences += batch_size
        self.total_batches += 1
        self.total_nfe_sequences += nfe * batch_size
        self.total_tokens_eos += count_generated_tokens(out, self.eos_token_id)
        self.total_tokens_full += batch_size * gen_length

    def summary(self) -> dict:
        t = self.total_time or float("nan")
        nfe = self.total_nfe_sequences or float("nan")
        seqs = self.total_sequences or float("nan")
        return {
            "name": self.name,
            "gen_length": self.gen_length,
            "sequences": self.total_sequences,
            "batches": self.total_batches,
            "generation_time_s": round(self.total_time, 3),
            # Throughput, both conventions. `throughput_tok_s` is the headline
            # number (tokens up to <eos>, Fast-dLLM protocol); the `_full`
            # variant counts every filled position instead.
            "throughput_tok_s": round(self.total_tokens_eos / t, 2),
            "throughput_tok_s_full": round(self.total_tokens_full / t, 2),
            # Tokens per denoising step. `_full` is the one to quote: it equals
            # gen_length / NFE and is exactly 1.0 for fixed-schedule greedy.
            "tokens_per_step": round(self.total_tokens_full / nfe, 3),
            "tokens_per_step_eos": round(self.total_tokens_eos / nfe, 3),
            "avg_nfe_per_sequence": round(self.total_nfe_sequences / seqs, 2),
            "avg_latency_per_sequence_s": round(self.total_time / seqs, 4),
            "tokens_before_eos": self.total_tokens_eos,
            "tokens_total": self.total_tokens_full,
        }

    def render(self) -> str:
        s = self.summary()
        return (
            f"\n===== Decoding speed [{s['name']}] =====\n"
            f"  sequences            : {s['sequences']} (batches: {s['batches']})\n"
            f"  gen_length           : {s['gen_length']}\n"
            f"  generation time      : {s['generation_time_s']} s\n"
            f"  throughput           : {s['throughput_tok_s']} tok/s   (tokens before <eos>)\n"
            f"  throughput (all pos) : {s['throughput_tok_s_full']} tok/s\n"
            f"  tokens / step        : {s['tokens_per_step']}\n"
            f"  avg NFE / sequence   : {s['avg_nfe_per_sequence']}\n"
            f"  latency / sequence   : {s['avg_latency_per_sequence_s']} s\n"
            f"=========================================\n"
        )

    def dump(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
