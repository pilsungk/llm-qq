# llm-qq

Code, item sets, and immutable audit records for the paper:

> **Auditing Question-Order Effects in Large Language Models with the QQ
> Equality: Mechanism Characterization and a Saturation Caveat**
> Pilsung Kang (Dankook University).
> arXiv: [link to be added upon submission]

The paper develops the QQ equality (Wang et al., PNAS 2014) into an
audit criterion for sequential judgments of autoregressive LLMs, and
reports a first-signal pilot in which a pre-specified saturation
diagnostic shows that forced-binary next-token log-probabilities were
inadequate for distribution-level QQ audits under the tested conditions.

## Repository layout

```
core/    validated mathematical core (QQ/OSS/Gamma metrics, ground-truth
         simulators, synthetic validation runner)
src/     LLM-side pipeline: logprob backend, item handling, pilot and
         redesign (Track P) runners, metrics, reporting
items/   frozen item set used by the reported runs (pilot_v1_2.json)
tests/   assertion suite (pytest)
runs/    immutable audit records (JSON) and run reports (Markdown) for
         the two executions reported in the paper
specs/   frozen execution specs; each audit record references its spec
         by spec_id and spec_sha256, verifiable against these files
notebooks/  figure generation (reads runs/*.json, writes figures/)
```

## Environment

- Python 3.11+, CUDA GPU (the reported runs used an RTX 5070 Ti, bf16)
- `pip install -r requirements.txt`
  (PyTorch CUDA wheels: install per https://pytorch.org for your CUDA
  version; the reported runs used torch 2.11.0+cu128)
- Model weights are downloaded from Hugging Face at run time
  (`Qwen/Qwen3-4B-Instruct-2507`); no weights are included here.

## Reproducing the reported runs

- Track M pilot (18 pairs): `src/pilot_runner.py` (seed 20260718)
- Track P redesign run (8 pairs): `src/r1_runner.py` (seed 20260719)
- Unit tests: `pytest tests/` (51 tests)
- Figures: run `notebooks/make_figures.ipynb`

Each run writes a JSON audit record (model and tokenizer revisions,
chat-template hash, canonical label token ids, mappings, seeds, decoding
parameters, per-position probabilities, gate results) before and during
execution. The records in `runs/` are the immutable reference outputs
for the paper; re-runs on the same hardware and versions should
reproduce them up to floating-point determinism of the GPU stack.

## Audit-record integrity

Every audit record carries `spec_id` and `spec_sha256` fields naming the
frozen execution spec it was run under. The corresponding spec files are
in `specs/`, so the linkage between the pre-specified design and each
execution is independently checkable.

## License

Code is released under the MIT License (see `LICENSE`). The audit
records and run reports in `runs/` and the specs in `specs/` may be
reused with attribution (CC BY 4.0).

## Citation

Citation metadata will be added when the arXiv identifier is assigned.
