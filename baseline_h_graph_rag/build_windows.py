"""
Baseline H — Phase 1, Step B (prep): Build windowed graph-extraction jobs.

Window definition: 30-visit windows, stride 25 (5-visit overlap) — the same
convention used everywhere else in this project (see
``common.data_utils.DEFAULT_CHUNK_SIZE/STRIDE``), applied here over each
patient's *visit_idx* range (not chunk index), gathering every fine-grained
structural chunk from Step A whose visit_idx falls in the window.

The 5-visit overlap is intentional: it lets ``build_temporal_graph.py``
merge/dedupe events that get independently extracted in two consecutive
windows, preserving continuity across window boundaries instead of losing
events that happen to fall near a boundary.

To keep each LLM call's prompt bounded, if a window's concatenated chunk
text exceeds ``MAX_CHARS_PER_WINDOW`` we keep the longest (most
information-dense) chunks up to the cap, dropping short procedural entries.

Output: shared/workflow_H_indices/graph_extraction_jobs.jsonl
  one line per window: {person_id, window_idx, visit_start, visit_end,
                        n_chunks_total, n_chunks_kept,
                        chunks: [{uuid, date, section, text}]}
"""
import json
import os

OUT_DIR = "shared/workflow_H_indices"
CHUNK_SIZE = 30   # visits per window
STRIDE = 25       # -> 5 visit overlap
MAX_CHARS_PER_WINDOW = 9000


def build_patient_windows(pid: str):
    d = os.path.join(OUT_DIR, f"patient_{pid}")
    with open(os.path.join(d, "chunks.jsonl")) as f:
        chunks = [json.loads(l) for l in f if l.strip()]

    n_visits = max(c["visit_idx"] for c in chunks) + 1
    by_visit = {}
    for c in chunks:
        by_visit.setdefault(c["visit_idx"], []).append(c)

    if n_visits <= CHUNK_SIZE:
        ranges = [(0, n_visits - 1)]
    else:
        ranges, start = [], 0
        while start < n_visits:
            end = min(start + CHUNK_SIZE - 1, n_visits - 1)
            ranges.append((start, end))
            if end == n_visits - 1:
                break
            start += STRIDE

    windows = []
    for widx, (vs, ve) in enumerate(ranges):
        win_chunks = []
        for v in range(vs, ve + 1):
            win_chunks.extend(by_visit.get(v, []))
        n_total = len(win_chunks)

        kept = win_chunks
        total_chars = sum(len(c["text"]) for c in win_chunks)
        if total_chars > MAX_CHARS_PER_WINDOW:
            sorted_by_density = sorted(win_chunks, key=lambda c: -len(c.get("raw_text") or ""))
            acc, sel = 0, []
            for c in sorted_by_density:
                if acc >= MAX_CHARS_PER_WINDOW:
                    break
                sel.append(c)
                acc += len(c["text"])
            sel_uuids = {c["uuid"] for c in sel}
            kept = [c for c in win_chunks if c["uuid"] in sel_uuids]  # restore chronological order

        windows.append({
            "person_id": pid, "window_idx": widx, "visit_start": vs, "visit_end": ve,
            "n_chunks_total": n_total, "n_chunks_kept": len(kept),
            "chunks": [{"uuid": c["uuid"], "date": c["date"], "section": c["section"], "text": c["text"]}
                       for c in kept],
        })
    return windows


def main():
    with open(os.path.join(OUT_DIR, "manifest.json")) as f:
        manifest = json.load(f)
    patient_ids = list(manifest["patients"].keys())

    all_windows = []
    for pid in patient_ids:
        wins = build_patient_windows(pid)
        all_windows.extend(wins)
        kept_total = sum(w["n_chunks_kept"] for w in wins)
        raw_total = sum(w["n_chunks_total"] for w in wins)
        print(f"[OK] patient {pid}: {len(wins)} windows, chunks kept {kept_total}/{raw_total}")

    out_path = os.path.join(OUT_DIR, "graph_extraction_jobs.jsonl")
    with open(out_path, "w") as f:
        for w in all_windows:
            f.write(json.dumps(w) + "\n")

    sizes = [sum(len(c["text"]) for c in w["chunks"]) for w in all_windows]
    print(f"[DONE] total_windows={len(all_windows)} across {len(patient_ids)} patients")
    print(f"[STATS] window char size: min={min(sizes)} max={max(sizes)} avg={sum(sizes)/len(sizes):.0f}")
    print(f"[OUT] {out_path}")


if __name__ == "__main__":
    main()
