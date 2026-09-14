"""
Baseline F — Semantic Memory Preparation: Trend-Only Reflection Generation + LoRA Training
=============================================================================================

**Role in the architecture.** Baseline F ("Standard Hybrid RAG") pairs the
episodic RAG index (Stream A, ``build_episodic_index.py``) with a
lightweight *semantic* memory stream (Stream B): a small per-patient LoRA
adapter that internalizes high-level trends and disease trajectories, while
deliberately SKIPPING exact point-in-time facts (dates, dosages, lab values)
-- those are the RAG index's job. This division of labor is what the
project calls the "epistemic hierarchy": trust RAG for facts/dates, trust
the adapter for narrative trends/sequencing.

**Stream B pipeline.**
  1. **Trend reflection generation** (frozen Qwen2.5-32B-Instruct): for
     each 30-visit overlapping chunk, generate a short natural-language
     reflection describing the patient's overall trajectory across that
     chunk (e.g. "worsening glycemic control with three hypoglycemic
     episodes"), explicitly instructed to omit exact dates/values.
  2. **LoRA adapter training** (``meta-llama/Llama-3.1-8B-Instruct``,
     r=16, alpha=32, all-linear-layer targets): one adapter per patient,
     trained with masked causal-LM loss on (chunk_summary_prompt ->
     trend_reflection) pairs, mirroring the training recipe used for
     Baseline G's retrieval-policy adapters (``baseline_g_adaptive_retrieval/train_adapters.py``)
     but on trend narratives instead of retrieval policies, and on the 8B
     model instead of 1B (Baseline F is the "8B Hybrid Dual-Memory"
     configuration in this project's nomenclature).

Output: ``/data/baseline_f_adapters/patient_{id}/`` (peft adapter files).
"""
import os

import modal

app = modal.App("baseline-f-trend-adapter")

GENERATOR_MODEL = "Qwen/Qwen2.5-32B-Instruct"
MEMORY_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAX_SEQ_LEN = 4096
TARGET_STEPS, MIN_EPOCHS, MAX_EPOCHS, LR = 40, 3, 24, 2e-4

TREND_SYSTEM_PROMPT = (
    "You are a clinical trend analyst. You will be shown a chunk of a patient's "
    "consecutive visit notes. Summarize the OVERALL TREND or trajectory evident across "
    "this chunk in 2-4 sentences (e.g. worsening/improving condition, emerging pattern of "
    "symptoms, change in care approach). Deliberately OMIT exact dates, doses, and lab "
    "values -- a separate factual retrieval system already covers those. Focus purely on "
    "the narrative arc / semantic trend."
)

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands("uv pip install --system vllm==0.6.6.post1")
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .add_local_python_source("common")
)

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands("uv pip install --system torch transformers accelerate peft numpy tqdm")
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .add_local_python_source("common")
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("temporal-memo-data", create_if_missing=True)


@app.function(image=vllm_image, gpu="H100", timeout=7200,
               volumes={"/data": data_volume, "/root/.cache/huggingface": hf_cache},
               secrets=[modal.Secret.from_name("huggingface-secret")])
def generate_trend_reflections(person_ids: list = None):
    """Stream B, step 1: generate trend-only reflections for every 30-visit
    chunk of every requested patient, via the frozen 32B generator."""
    import sys
    sys.path.insert(0, "/root")
    import json

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from common.data_utils import CANONICAL_PATIENT_IDS, load_canonical_cohort, build_overlapping_chunks

    person_ids = person_ids or CANONICAL_PATIENT_IDS
    cohort = load_canonical_cohort("/data/patient_sequences.jsonl", person_ids)

    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
    llm = LLM(model=GENERATOR_MODEL, dtype="bfloat16", max_model_len=16384, gpu_memory_utilization=0.95)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=300)

    out_dir = "/data/baseline_f_reflections"
    os.makedirs(out_dir, exist_ok=True)

    for patient in cohort:
        pid = patient["person_id"]
        chunks = build_overlapping_chunks(patient.get("visits", []))
        prompts = []
        for chunk in chunks:
            visits_text = "\n\n".join(
                f"[{v.get('visit_datetime')}] {(v.get('text') or '')[:1200]}" for v in chunk["visits"]
            )
            user_msg = f"Visits {chunk['start']}-{chunk['end']} ({chunk['date_start']} to {chunk['date_end']}):\n\n{visits_text}"
            messages = [{"role": "system", "content": TREND_SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        outputs = llm.generate(prompts, sampling_params)
        with open(f"{out_dir}/trend_{pid}.jsonl", "w") as f:
            for chunk, out in zip(chunks, outputs):
                f.write(json.dumps({
                    "person_id": pid, "chunk_index": chunk["chunk_index"],
                    "prompt": prompts[chunk["chunk_index"]],
                    "completion": out.outputs[0].text.strip(),
                }) + "\n")
        print(f"[OK] patient {pid}: {len(chunks)} trend reflections generated")

    data_volume.commit()
    return {"n_patients": len(cohort)}


def _build_example(tokenizer, prompt, completion, max_seq_len=MAX_SEQ_LEN):
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    completion_ids = list(tokenizer(completion, add_special_tokens=False)["input_ids"])
    eos = tokenizer.eos_token_id
    input_ids = prompt_ids + completion_ids + [eos]
    labels = [-100] * len(prompt_ids) + completion_ids + [eos]
    if len(input_ids) > max_seq_len:
        overflow = len(input_ids) - max_seq_len
        input_ids, labels = input_ids[overflow:], labels[overflow:]
    return input_ids, labels


@app.function(image=train_image, gpu="A100", timeout=7200,
               volumes={"/data": data_volume, "/root/.cache/huggingface": hf_cache},
               secrets=[modal.Secret.from_name("huggingface-secret")])
def train_trend_adapters(person_ids: list = None):
    """Stream B, step 2: train one r=16 LoRA adapter per patient on
    Llama-3.1-8B-Instruct to internalize that patient's trend reflections."""
    import sys
    sys.path.insert(0, "/root")
    import json
    import gc

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from common.data_utils import CANONICAL_PATIENT_IDS

    person_ids = person_ids or CANONICAL_PATIENT_IDS

    tokenizer = AutoTokenizer.from_pretrained(MEMORY_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(MEMORY_MODEL, torch_dtype=torch.bfloat16).to("cuda")
    base_model.config.use_cache = False

    out_root = "/data/baseline_f_adapters"
    os.makedirs(out_root, exist_ok=True)
    summary = {}

    for pid in person_ids:
        path = f"/data/baseline_f_reflections/trend_{pid}.jsonl"
        if not os.path.exists(path):
            print(f"[SKIP] {pid}: no reflections found, run generate_trend_reflections first.")
            continue
        examples = [json.loads(l) for l in open(path) if l.strip()]
        tokenized = [_build_example(tokenizer, e["prompt"], e["completion"]) for e in examples]

        epochs = max(MIN_EPOCHS, min(MAX_EPOCHS, round(TARGET_STEPS / max(len(examples), 1))))

        lora_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                               target_modules=TARGET_MODULES, task_type="CAUSAL_LM")
        model = get_peft_model(base_model, lora_cfg)
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

        losses = []
        for epoch in range(epochs):
            for input_ids, labels in tokenized:
                ii = torch.tensor([input_ids], device="cuda")
                ll = torch.tensor([labels], device="cuda")
                out = model(input_ids=ii, attention_mask=torch.ones_like(ii), labels=ll)
                optimizer.zero_grad()
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                losses.append(out.loss.item())

        pdir = f"{out_root}/patient_{pid}"
        os.makedirs(pdir, exist_ok=True)
        model.save_pretrained(pdir)
        tokenizer.save_pretrained(pdir)
        summary[str(pid)] = {"n_examples": len(examples), "epochs": epochs,
                              "final_loss": sum(losses[-3:]) / min(3, len(losses))}
        print(f"[OK] patient {pid}: {len(examples)} examples, {epochs} epochs, "
              f"final_loss={summary[str(pid)]['final_loss']:.4f}")

        model.gradient_checkpointing_disable()
        model = model.unload()
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()

    with open(f"{out_root}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    data_volume.commit()
    return summary


@app.local_entrypoint()
def main(step: str = "reflect"):
    if step == "reflect":
        print(generate_trend_reflections.remote())
    elif step == "train":
        print(train_trend_adapters.remote())
    else:
        raise ValueError("step must be 'reflect' or 'train'")
