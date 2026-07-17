# logprob_backend.py
# ex18 R11 smoke test: HF inference backend for the binary_logprob and
# binary_rejection protocols.
#
# Responsibilities:
#   - Load an instruction-tuned causal LM and its tokenizer.
#   - Render multi-turn conversations ONLY via tokenizer.apply_chat_template
#     with add_generation_prompt=True (never hand-assemble template strings).
#   - Canonical single-token label validation in the rendered template context
#     (SRS 4.4 / 4.4.1).
#   - binary_logprob: next-token label probabilities at a generation position,
#     binary-conditioned P(y | y or n), raw mass, and top-5 diagnostics.
#   - binary_rejection: stage-wise rejection sampling with the fixed decoding
#     params (T=1, top_p=1, top_k disabled, max_new_tokens=1).
#
# ASCII only. English comments. No gradient anywhere; one forward at a time.

import hashlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fixed decoding for every sampling-based protocol (SRS 4.2). Immutable.
DECODING = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,          # 0 disables top-k filtering in transformers
    "do_sample": True,
    "max_new_tokens": 1,
}

# Human-visible canonical answer labels. The token id actually used for each
# branch is resolved by validate_label_pair() in the rendered context.
CANONICAL_DISPLAY = {"y": "Yes", "n": "No"}

# Candidate label spellings tried during single-token validation: bare and
# leading-space, per branch. Both members of a variant must be single-token.
LABEL_VARIANTS = [
    {"name": "bare", "y": "Yes", "n": "No"},
    {"name": "leading_space", "y": " Yes", "n": " No"},
]


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without a real model)
# ---------------------------------------------------------------------------

def single_token_added(base_ids, full_ids):
    """Return (is_single, added_id).

    is_single is True iff full_ids equals base_ids with exactly one extra
    token appended at the end (i.e. the label added exactly one token in the
    rendered context). added_id is that token id, or None.
    """
    if len(full_ids) != len(base_ids) + 1:
        return False, None
    if list(full_ids[:len(base_ids)]) != list(base_ids):
        return False, None
    return True, full_ids[-1]


def validate_label_pair(encode, rendered, variants=LABEL_VARIANTS):
    """Choose the first label variant whose Yes and No each add exactly one
    token when appended to the rendered prompt.

    encode: callable(text) -> list[int] token ids (no special tokens added).
    rendered: rendered prompt string ending at the generation position.

    Returns a dict describing the outcome:
      {passed, variant, token_ids{y,n}, per_candidate{...}}
    passed is False if no variant yields a single-token pair.
    """
    base_ids = encode(rendered)
    per_candidate = {}
    chosen = None
    for var in variants:
        ok_pair = True
        ids = {}
        for key in ("y", "n"):
            label = var[key]
            full_ids = encode(rendered + label)
            is_single, added = single_token_added(base_ids, full_ids)
            per_candidate["%s:%s" % (var["name"], key)] = {
                "label": label,
                "is_single": is_single,
                "added_id": added,
                "n_tokens_added": len(full_ids) - len(base_ids),
            }
            if is_single:
                ids[key] = added
            else:
                ok_pair = False
        if ok_pair and chosen is None:
            chosen = {"variant": var["name"], "token_ids": ids}
    result = {
        "passed": chosen is not None,
        "variant": chosen["variant"] if chosen else None,
        "token_ids": chosen["token_ids"] if chosen else None,
        "per_candidate": per_candidate,
    }
    return result


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class LogprobBackend:
    def __init__(self, model_id, dtype=torch.bfloat16, device_map="cuda",
                 quantization_config=None):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        load_kwargs = {"device_map": device_map}
        if quantization_config is not None:
            # 8-bit path (M2). dtype is governed by the quantization config.
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["torch_dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        # Resolved by validate() before any measurement.
        self.yes_id = None
        self.no_id = None

    # -- chat template ------------------------------------------------------

    def render(self, messages):
        """Render a message list to a prompt string ending at the generation
        position. The ONLY path from messages to text."""
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def chat_template_sha256(self):
        tmpl = self.tokenizer.chat_template
        if tmpl is None:
            return None
        return hashlib.sha256(tmpl.encode("utf-8")).hexdigest()

    def _encode(self, text):
        # Rendered text already carries template special tokens; do not add more.
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def _to_tensors(self, rendered):
        enc = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        return enc.input_ids.to(self.device), enc.attention_mask.to(self.device)

    # -- label validation ---------------------------------------------------

    def validate_labels(self, probe_messages):
        """Validate canonical labels in the rendered context of probe_messages.
        Sets self.yes_id / self.no_id on success. Returns the result dict."""
        rendered = self.render(probe_messages)
        result = validate_label_pair(self._encode, rendered)
        if result["passed"]:
            self.yes_id = result["token_ids"]["y"]
            self.no_id = result["token_ids"]["n"]
        return result

    # -- binary_logprob -----------------------------------------------------

    def position_readout(self, messages, topk=5, id_y=None, id_n=None):
        """Forward pass on the rendered prompt; return label probabilities at
        the generation position.

        id_y / id_n select the two canonical token ids to read. They default
        to self.yes_id / self.no_id (used by the Yes/No smoke test). The pilot
        passes A/B or Yes/No token ids explicitly per label scheme.

        Returns dict:
          p_yes, p_no, raw_mass (= p_yes + p_no), other_mass (= 1 - raw_mass),
          p_yes_cond (= p_yes / raw_mass), top5 [{id,string,prob}].
        Here 'yes'/'no' name the FIRST/SECOND canonical token generically.
        """
        if id_y is None:
            id_y = self.yes_id
        if id_n is None:
            id_n = self.no_id
        rendered = self.render(messages)
        input_ids, attn = self._to_tensors(rendered)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        p_yes = float(probs[id_y].item())
        p_no = float(probs[id_n].item())
        raw_mass = p_yes + p_no
        top_p, top_i = torch.topk(probs, topk)
        top5 = []
        for prob, tid in zip(top_p.tolist(), top_i.tolist()):
            top5.append({
                "id": int(tid),
                "string": self.tokenizer.decode([tid]),
                "prob": float(prob),
            })
        return {
            "p_yes": p_yes,
            "p_no": p_no,
            "raw_mass": raw_mass,
            "other_mass": 1.0 - raw_mass,
            "p_yes_cond": (p_yes / raw_mass) if raw_mass > 0 else float("nan"),
            "top5": top5,
        }

    # -- binary_rejection ---------------------------------------------------

    def _sample_batch(self, messages, k, id_y=None, id_n=None):
        """Sample k first-position tokens with the fixed decoding params.
        Returns a list of labels in {'y','n','other'} of length k, where
        'y'/'n' name the id_y/id_n canonical tokens generically."""
        if id_y is None:
            id_y = self.yes_id
        if id_n is None:
            id_n = self.no_id
        rendered = self.render(messages)
        input_ids, attn = self._to_tensors(rendered)
        input_ids = input_ids.repeat(k, 1)
        attn = attn.repeat(k, 1)
        gen_kwargs = dict(DECODING)
        gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids, attention_mask=attn, **gen_kwargs
            )
        new_tokens = out[:, input_ids.shape[1]].tolist()
        labels = []
        for tid in new_tokens:
            if tid == id_y:
                labels.append("y")
            elif tid == id_n:
                labels.append("n")
            else:
                labels.append("other")
        return labels

    def collect_accepted(self, messages, need, batch=64, max_attempts=None,
                         id_y=None, id_n=None):
        """Rejection-sample the answer at this position until `need` canonical
        answers are accepted. Returns (accepted_labels, stats).

        stats: {attempts, accepted, other, shortfall}. attempts counts every
        draw examined (attempts vs accepted for the audit record).
        """
        if max_attempts is None:
            max_attempts = max(need * 200, 4000)
        accepted = []
        attempts = 0
        other = 0
        done = False
        while not done and attempts < max_attempts:
            for lab in self._sample_batch(messages, batch, id_y=id_y, id_n=id_n):
                attempts += 1
                if lab == "other":
                    other += 1
                else:
                    accepted.append(lab)
                    if len(accepted) >= need:
                        done = True
                        break
        stats = {
            "attempts": attempts,
            "accepted": len(accepted),
            "other": other,
            "shortfall": len(accepted) < need,
        }
        return accepted, stats
