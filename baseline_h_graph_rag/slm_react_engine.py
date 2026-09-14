"""
Baseline H — Phases 2 & 3: Adaptive SLM ReAct Retrieval + Grounded Synthesis
================================================================================

**Architecture.** A SINGLE ``meta-llama/Llama-3.2-3B-Instruct`` model
autonomously performs ALL roles itself, ReAct-style, with no separate
Executive model at all — a deliberate architectural departure from
Baselines E/F/G, all of which rely on Claude as the Executive:

  1. **Action generation** — choose ONE search strategy (lexical / semantic
     / temporal) + a query string / event-type + optional date_hint, given
     the question and evidence gathered so far.
  2. **Sufficiency assessment** — decide SUFFICIENT / INSUFFICIENT given the
     evidence pool.
  3. **Grounded synthesis** — write the final answer strictly from the
     retrieved chunks, with inline ``[Date: ... | Section: ...]`` citations.

**Adaptive Retrieval Loop control (the key novelty vs. Baseline G):**
  - **Duplicate-strategy force-halt.** A normalized signature string
    ``"strategy|query|date_hint"`` is tracked across hops. If the model's
    newly proposed action exactly matches a prior one, the loop halts
    BEFORE executing that redundant retrieval (``halt_reason =
    "duplicate_strategy"``). This is the cycle-breaking guard against
    infinite loops that small models are prone to (re-proposing the same
    search when they don't know what else to try) — a formal, mathematical
    termination condition rather than a fixed-hop guess.
  - **Hard cap**: ``MAX_HOPS = 5`` (``halt_reason = "max_hops"``).
  - **Early stop**: ``halt_reason = "sufficient"`` once the sufficiency
    step returns SUFFICIENT.

This is a Modal-hosted ``@app.cls`` service (keeps the 3B model + MedCPT
query encoder loaded in GPU memory across an entire patient's queries,
avoiding repeated cold starts). See ``orchestrator.py`` for the local driver
that calls this service per patient.
"""
import json
import os
import re
import time

import modal

app = modal.App("baseline-h-slm-react")

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
QUERY_ENCODER = "ncbi/MedCPT-Query-Encoder"
MAX_HOPS = 5
TOP_K = 5

EVENT_TYPE_VOCAB = [
    "diagnosis", "disposition", "encounter", "imaging", "lab_result",
    "medication", "other", "procedure", "triage", "vital_sign",
]

ACTION_SYSTEM_PROMPT = (
    "You are an autonomous clinical retrieval agent. You search ONE specific patient's "
    "electronic health record to gather evidence needed to answer a clinical question. "
    "Each turn you must choose exactly ONE search strategy: "
    "\"lexical\" (keyword/BM25 search over chart text), "
    "\"semantic\" (meaning-based similarity search over chart text), or "
    "\"temporal\" (search the patient's clinical event graph by event type and approximate "
    "date). Respond with STRICT JSON only -- a single JSON object, no markdown fences, no "
    "prose outside the JSON."
)

ACTION_USER_TEMPLATE = """Patient ID: {person_id}
Clinical Question: {question}

Evidence gathered so far ({n_chunks} chunk(s)):
{evidence_summary}

Search strategies already tried (exact repeats are forbidden -- you MUST propose something
different, e.g. a different strategy, a different query, or a different date_hint):
{strategy_history_str}

Choose your NEXT search action. Output STRICT JSON with exactly these fields:
- "thought": one short sentence on what evidence is still missing.
- "strategy": one of "lexical", "semantic", "temporal".
- "query": for "lexical"/"semantic": a short natural-language search phrase; for "temporal":
  ONE of these event types EXACTLY: {event_types}.
- "date_hint": an approximate date relevant to this search as "YYYY-MM-DD", "YYYY-MM", or
  "YYYY", or the string "none" if there is no temporal signal."""

SUFFICIENCY_SYSTEM_PROMPT = (
    "You are a strict clinical evidence auditor. Decide whether the evidence chunks below "
    "are sufficient to fully and accurately answer the clinical question, using ONLY this "
    "evidence (no outside medical knowledge). Respond with STRICT JSON only."
)
SUFFICIENCY_USER_TEMPLATE = """Clinical Question: {question}

Evidence gathered so far:
{evidence_full_text}

Output STRICT JSON with exactly these fields:
- "verdict": "SUFFICIENT" or "INSUFFICIENT".
- "reasoning": one short sentence explaining why."""

SYNTHESIS_SYSTEM_PROMPT = (
    "You are a clinical documentation assistant. Answer the clinical question STRICTLY and "
    "ONLY using the evidence chunks provided below. Do not use any outside knowledge, do not "
    "guess, and do not fabricate details not present in the evidence. If the evidence is "
    "incomplete, say so explicitly while answering with what IS supported. For every factual "
    "claim, include an inline citation in the EXACT format [Date: YYYY-MM-DD | Section: "
    "<section>] matching one of the evidence chunks' date and section given below."
)
SYNTHESIS_USER_TEMPLATE = """Clinical Question: {question}

Evidence chunks (the ONLY source of truth -- cite [Date: ... | Section: ...] for every claim):
{evidence_full_text}

Write the final grounded answer now."""

slm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands("uv pip install --system torch transformers numpy accelerate")
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .add_local_python_source("common")
    .add_local_dir("shared/workflow_H_indices", remote_path="/root/data/workflow_H_indices")
    .add_local_python_source("baseline_h_graph_rag")
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


def _extract_json_object(text):
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip()).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def _normalize_signature(strategy, query, date_hint):
    s = (strategy or "").strip().lower()
    q = re.sub(r"\s+", " ", (query or "").strip().lower())
    d = (date_hint or "none").strip().lower()
    return f"{s}|{q}|{d}"


def _format_evidence_summary(evidence_pool, max_chunks=15, snippet_len=150):
    if not evidence_pool:
        return "(none yet)"
    items = sorted(evidence_pool.values(), key=lambda c: c.get("date") or "")
    return "\n".join(f"- [{c.get('date')} | {c.get('section')}] {c.get('raw_text', '')[:snippet_len]}"
                      for c in items[-max_chunks:])


def _format_evidence_full(evidence_pool, max_chars=6000):
    if not evidence_pool:
        return "(no evidence retrieved)"
    items = sorted(evidence_pool.values(), key=lambda c: c.get("date") or "")
    text = "\n\n".join(f"[Date: {c.get('date')} | Section: {c.get('section')}] {c.get('raw_text', '')}" for c in items)
    return text[:max_chars] + "\n...(truncated)" if len(text) > max_chars else text


def _parse_cited_chunk_ids(final_answer, evidence_pool):
    pattern = re.compile(r"\[Date:\s*([^\|\]]+?)\s*\|\s*Section:\s*([^\]]+?)\s*\]")
    by_date_section = {}
    for cid, c in evidence_pool.items():
        key = ((c.get("date") or "").strip(), (c.get("section") or "").strip())
        by_date_section.setdefault(key, []).append(cid)
    cited = []
    for date_str, section_str in pattern.findall(final_answer or ""):
        for cid in by_date_section.get((date_str.strip(), section_str.strip()), []):
            if cid not in cited:
                cited.append(cid)
    return cited


@app.cls(
    image=slm_image, gpu="a100", timeout=6 * 3600,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,
)
class SLMReActEngine:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
        from baseline_h_graph_rag.tri_modal_retrieval import load_temporal_graph

        print(f"[SLMReActEngine] Loading {MODEL_NAME} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to("cuda").eval()

        print(f"[SLMReActEngine] Loading {QUERY_ENCODER} ...")
        self.qe_tokenizer = AutoTokenizer.from_pretrained(QUERY_ENCODER)
        self.qe_model = AutoModel.from_pretrained(QUERY_ENCODER).to("cuda").eval()

        print("[SLMReActEngine] Loading temporal graph ...")
        self.graph = load_temporal_graph(base_dir="/root/data/workflow_H_indices")
        print("[SLMReActEngine] Ready.")

    def _embed_query(self, text):
        import torch
        with torch.no_grad():
            encoded = self.qe_tokenizer([text], truncation=True, padding=True, return_tensors="pt", max_length=64).to("cuda")
            embeds = self.qe_model(**encoded).last_hidden_state[:, 0, :]
        return embeds[0].float().cpu().numpy()

    def _generate_json(self, system_prompt, user_prompt, max_new_tokens=300):
        import torch
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                       pad_token_id=self.tokenizer.pad_token_id)
        raw = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return raw, _extract_json_object(raw)

    def _generate_text(self, system_prompt, user_prompt, max_new_tokens=500):
        import torch
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                       pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def _process_one_query(self, index, person_id, question):
        t0 = time.time()
        evidence_pool, strategy_history, hops, halt_reason = {}, [], [], None

        for hop_idx in range(1, MAX_HOPS + 1):
            action_prompt = ACTION_USER_TEMPLATE.format(
                person_id=person_id, question=question, n_chunks=len(evidence_pool),
                evidence_summary=_format_evidence_summary(evidence_pool),
                strategy_history_str="\n".join(f"s{i+1}: {sig}" for i, sig in enumerate(strategy_history)) or "(none yet)",
                event_types=", ".join(EVENT_TYPE_VOCAB),
            )
            raw_action, action = self._generate_json(ACTION_SYSTEM_PROMPT, action_prompt, max_new_tokens=250)
            if action is None or not isinstance(action, dict):
                action = {"thought": "(parse error, defaulting to lexical)", "strategy": "lexical",
                           "query": question, "date_hint": "none"}

            strategy, query, date_hint = action.get("strategy", "lexical"), action.get("query", question), action.get("date_hint", "none")
            sig = _normalize_signature(strategy, query, date_hint)

            if sig in strategy_history:
                hops.append({"hop": hop_idx, "action": action, "signature": sig, "duplicate_halt": True})
                halt_reason = "duplicate_strategy"
                break
            strategy_history.append(sig)

            query_embedding = self._embed_query(query) if strategy == "semantic" else None
            hits = index.search(strategy, query=query, query_embedding=query_embedding, date_hint=date_hint, top_k=TOP_K)
            new_chunk_ids = [h["chunk_id"] for h in hits if h["chunk_id"] not in evidence_pool]
            for h in hits:
                evidence_pool.setdefault(h["chunk_id"], h)

            raw_suff, suff = self._generate_json(
                SUFFICIENCY_SYSTEM_PROMPT,
                SUFFICIENCY_USER_TEMPLATE.format(question=question, evidence_full_text=_format_evidence_full(evidence_pool)),
                max_new_tokens=150,
            )
            verdict = str((suff or {}).get("verdict", "INSUFFICIENT")).strip().upper()

            hops.append({"hop": hop_idx, "action": action, "signature": sig, "duplicate_halt": False,
                         "new_chunk_ids_retrieved": new_chunk_ids, "total_evidence_pool_size": len(evidence_pool),
                         "sufficiency_verdict": verdict})

            if verdict == "SUFFICIENT":
                halt_reason = "sufficient"
                break
            if hop_idx == MAX_HOPS:
                halt_reason = "max_hops"
                break

        final_answer = (
            self._generate_text(SYNTHESIS_SYSTEM_PROMPT,
                                 SYNTHESIS_USER_TEMPLATE.format(question=question, evidence_full_text=_format_evidence_full(evidence_pool)))
            if evidence_pool else "No evidence could be retrieved from the patient's chart for this question."
        )

        return {
            "final_answer": final_answer,
            "cited_chunk_ids": _parse_cited_chunk_ids(final_answer, evidence_pool),
            "n_iterations": len(hops), "halt_reason": halt_reason, "strategy_history": strategy_history,
            "hops": hops, "final_evidence_pool_chunk_ids": list(evidence_pool.keys()),
            "final_evidence_pool_size": len(evidence_pool), "elapsed_sec": round(time.time() - t0, 2),
        }

    @modal.method()
    def run_patient_queries(self, person_id: int, queries: list):
        from baseline_h_graph_rag.tri_modal_retrieval import PatientTriModalIndex

        index = PatientTriModalIndex(person_id, base_dir="/root/data/workflow_H_indices", graph=self.graph)
        results = []
        for q in queries:
            question = q["question"]
            try:
                r = self._process_one_query(index, person_id, question)
                record = {"person_id": person_id, "question": question, "ground_truth_answer": q.get("answer"),
                          "reasoning_type": q.get("reasoning_type"), "difficulty": q.get("difficulty"),
                          "is_temporal": bool(q.get("is_temporal")), **r}
            except Exception as e:
                import traceback
                record = {"person_id": person_id, "question": question, "ground_truth_answer": q.get("answer"),
                          "error": str(e), "traceback": traceback.format_exc(), "final_answer": None}
            results.append(record)
            print(f"  [pid {person_id}] {question[:60]!r} -> {record.get('n_iterations')} iter(s), "
                  f"halt={record.get('halt_reason')}, {record.get('elapsed_sec')}s")
        return results

    @modal.method()
    def health_check(self):
        return {"status": "ok", "model": MODEL_NAME, "query_encoder": QUERY_ENCODER}


@app.local_entrypoint()
def main(person_id: int = 8855233, limit: int = 2):
    import pandas as pd
    svc = SLMReActEngine()
    print("Health check:", svc.health_check.remote())
    df = pd.read_excel("shared/spreadsheet.xlsx")
    sub = df[df["person_id"] == person_id].to_dict("records")[:limit]
    for r in svc.run_patient_queries.remote(person_id, sub):
        print(json.dumps(r, indent=2, default=str)[:2000])
