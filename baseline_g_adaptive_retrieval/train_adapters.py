"""
Baseline G — Phase 2b: Train 20 patient-specific LoRA adapters
==================================================================

Trains one fresh LoRA adapter (``r=16, alpha=32``, all-linear-layer targets)
per patient on top of the frozen ``meta-llama/Llama-3.2-1B-Instruct`` base
model, using each patient's own Phase-2a retrieval-policy SFT data. This is
the "Retrieval-Policy Distillation" step: the small model learns to
reproduce the 32B generator's *search strategy* for this specific patient's
chart, not clinical facts.

Key design choices (documented inline where they matter):
  - **Masked causal-LM loss**: loss is computed ONLY on completion tokens
    (prompt tokens masked to ``label = -100``), completion terminated with
    EOS — standard SFT masking so the model isn't penalized for not
    "predicting" the prompt.
  - **Adaptive epoch count**: targets ~``TARGET_STEPS`` optimizer steps per
    patient regardless of how many examples that patient has (patients can
    have anywhere from ~2 to ~70 chunk/policy examples), so no adapter is
    drastically under- or over-trained relative to the others.
  - **Per-patient adapter isolation**: after each patient, the LoRA layers
    are unloaded and CUDA memory is aggressively freed before moving to the
    next patient — repeated attach/detach on long clinical sequences
    otherwise fragments the allocator and can OOM over 20 patients.

Output: /data/baseline_g_adapters/1B/patient_{id}/ (adapter_config.json +
adapter weights, peft ``save_pretrained`` format).
"""
import json
import os

import modal

app = modal.App("baseline-g-phase2-lora-adapters")

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

CANONICAL_PATIENT_IDS = [
    8855233, 8858035, 8860166, 8860822, 8911710,
    8922374, 8945136, 8955898, 8979075, 9070451,
    9077267, 9088069, 9102132, 9126037, 9354504,
    9989372, 11235719, 11617093, 12015052, 12355020,
]

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands("uv pip install --system torch transformers accelerate peft numpy tqdm requests")
    .env({"HF_HOME": "/root/.cache/huggingface", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("temporal-memo-data", create_if_missing=True)

MAX_SEQ_LEN = 8192
TARGET_STEPS = 40          # target optimizer steps per patient (adaptive epochs)
MIN_EPOCHS, MAX_EPOCHS = 3, 24
LR = 2e-4
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_example(tokenizer, prompt, completion, max_seq_len=MAX_SEQ_LEN):
    """Tokenize a (prompt, completion) pair into input_ids/labels with the
    prompt masked out of the loss."""
    messages = [
        {"role": "system", "content": "You are an expert clinical retrieval engineer."},
        {"role": "user", "content": prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = list(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    completion_ids = list(tokenizer(completion, add_special_tokens=False)["input_ids"])
    eos_id = tokenizer.eos_token_id
    input_ids = prompt_ids + completion_ids + [eos_id]
    labels = [-100] * len(prompt_ids) + completion_ids + [eos_id]
    if len(input_ids) > max_seq_len:
        overflow = len(input_ids) - max_seq_len
        input_ids, labels = input_ids[overflow:], labels[overflow:]
    return input_ids, labels


@app.function(
    image=train_image, gpu="A100", timeout=7200,
    volumes={"/root/.cache/huggingface": hf_cache, "/data": data_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_all_patients(debug: bool = False, debug_patients: int = 1, debug_max_steps: int = 5):
    import gc
    import time

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    patient_ids = CANONICAL_PATIENT_IDS[:debug_patients] if debug else CANONICAL_PATIENT_IDS

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to("cuda")
    base_model.config.use_cache = False
    print(f"[{time.strftime('%H:%M:%S')}] Base model loaded in {time.time()-t0:.1f}s")

    out_root = "/data/baseline_g_adapters/1B"
    os.makedirs(out_root, exist_ok=True)
    summary, global_step = {}, 0

    for pid in patient_ids:
        sft_path = f"/data/baseline_g_sft_data/sft_{pid}.jsonl"
        if not os.path.exists(sft_path):
            print(f"[SKIP] {pid}: no SFT data found, run build_sft_data.py first.")
            continue
        examples = [json.loads(l) for l in open(sft_path) if l.strip()]
        n_examples = len(examples)
        if debug:
            examples = examples[:2]

        tokenized = [build_example(tokenizer, ex["prompt"], ex["completion"]) for ex in examples]
        epochs = 1 if debug else max(MIN_EPOCHS, min(MAX_EPOCHS, round(TARGET_STEPS / max(n_examples, 1))))

        lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                  target_modules=TARGET_MODULES, task_type="CAUSAL_LM")
        model = get_peft_model(base_model, lora_config)
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
        patient_t0 = time.time()
        losses = []
        max_steps_this_patient = debug_max_steps if debug else None
        step_in_patient = 0

        for epoch in range(epochs):
            order = list(range(len(tokenized)))
            if not debug:
                import random
                random.Random(epoch).shuffle(order)
            for idx in order:
                input_ids, labels = tokenized[idx]
                ii = torch.tensor([input_ids], dtype=torch.long, device="cuda")
                ll = torch.tensor([labels], dtype=torch.long, device="cuda")
                attn = torch.ones_like(ii)
                out = model(input_ids=ii, attention_mask=attn, labels=ll)
                optimizer.zero_grad()
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                losses.append(out.loss.item())
                global_step += 1
                step_in_patient += 1
                del out, ii, ll, attn
                if max_steps_this_patient and step_in_patient >= max_steps_this_patient:
                    break
            if max_steps_this_patient and step_in_patient >= max_steps_this_patient:
                break

        patient_time = time.time() - patient_t0
        final_loss = sum(losses[-3:]) / min(3, len(losses))
        print(f"  -> Patient {pid} done in {patient_time:.1f}s. first_loss={losses[0]:.4f} "
              f"final_loss={final_loss:.4f} n_steps={len(losses)}")

        patient_out_dir = f"{out_root}/patient_{pid}"
        os.makedirs(patient_out_dir, exist_ok=True)
        model.save_pretrained(patient_out_dir)
        tokenizer.save_pretrained(patient_out_dir)

        summary[str(pid)] = {"n_examples": n_examples, "epochs": epochs, "n_steps": len(losses),
                              "first_loss": losses[0], "final_loss_avg_last3": final_loss,
                              "train_time_sec": round(patient_time, 1)}
        with open(f"{patient_out_dir}/patient_summary.json", "w") as f:
            json.dump(summary[str(pid)], f, indent=2)
        data_volume.commit()  # persist immediately -> safe to resume mid-run

        # Detach this patient's LoRA layers and aggressively free GPU memory
        # before starting the next patient's fresh adapter.
        model.gradient_checkpointing_disable()
        model = model.unload()
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    with open(f"{out_root}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    data_volume.commit()
    print(f"ALL DONE in {time.time()-t0:.1f}s.\n{json.dumps(summary, indent=2)}")
    return summary


@app.local_entrypoint()
def main(debug: bool = False):
    result = train_all_patients.remote(debug=debug)
    print("DONE:", json.dumps(result, indent=2) if not isinstance(result, str) else result)
