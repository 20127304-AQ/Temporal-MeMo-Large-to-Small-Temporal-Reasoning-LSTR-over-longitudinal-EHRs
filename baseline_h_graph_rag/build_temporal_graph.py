"""
Baseline H — Phase 1, Step B (post-processing): Build the Temporal Knowledge Graph
======================================================================================

Merges the raw Qwen2.5-32B window extractions (``step_b_raw_results.json``,
from ``extract_temporal_events.py``) into the final temporal knowledge
graph:
  - **nodes**: events, each containing ONLY a UUID pointer (``node_id ==
    chunk_uuid``) to a Step A structural chunk, plus a controlled
    ``event_type`` category and a time interval (``time_start``/
    ``time_end``). **NO clinical text is stored in the graph itself.**
  - **edges**: Allen's 13 interval relations, computed **deterministically**
    from the ``(time_start, time_end)`` intervals extracted by the LLM (see
    ``extract_temporal_events.py`` for why this split of labor exists).

**Validation performed:**
  - every ``chunk_uuid`` the LLM output must exist in Step A's global UUID
    map AND belong to the SAME ``person_id`` as the window it came from
    (dropped + counted otherwise — never silently trusted).
  - events referencing the same ``chunk_uuid`` across overlapping windows
    (the intentional 5-visit overlap from ``build_windows.py``) are
    de-duplicated, keeping the first extraction seen.
  - malformed/missing time strings fall back to the chunk's own recorded
    ``visit_datetime`` rather than being silently dropped.

Output: shared/workflow_H_indices/temporal_graph.json
"""
import json
import os
from datetime import datetime

IDX_DIR = "shared/workflow_H_indices"
RAW_RESULTS_PATH = os.path.join(IDX_DIR, "step_b_raw_results.json")
OUT_PATH = os.path.join(IDX_DIR, "temporal_graph.json")

ALLEN_RELATIONS = [
    "before", "after", "meets", "met_by", "overlaps", "overlapped_by",
    "starts", "started_by", "finishes", "finished_by", "during", "contains", "equals",
]

EVENT_TYPE_VOCAB = {
    "encounter", "triage", "diagnosis", "medication", "procedure",
    "lab_result", "imaging", "disposition", "vital_sign", "other",
}

# Some fraction of raw LLM outputs drift outside the requested 10-value
# controlled vocabulary (e.g. "examination", "social_history"). Rather than
# silently dropping those events, we canonicalize them into the closest
# requested category and report the out-of-vocabulary count transparently in
# the graph's metadata block.
EVENT_TYPE_CANONICALIZATION = {
    "examination": "procedure", "investigation": "lab_result", "assessment": "diagnosis",
    "impression": "diagnosis", "conclusion": "diagnosis", "plan": "disposition",
    "admission": "encounter", "hospital_course": "encounter", "progress": "encounter",
    "history": "other", "social": "other", "social_history": "other",
    "social_work": "other", "risk_assessment": "other", "nutrition_assessment": "other",
    "allergy": "other", "event": "other",
}


def canonicalize_event_type(t):
    if t in EVENT_TYPE_VOCAB:
        return t, False
    return EVENT_TYPE_CANONICALIZATION.get(t, "other"), True


def parse_time(s, fallback_iso):
    if not s:
        return parse_time(fallback_iso, fallback_iso)
    s = s.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:19], fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return parse_time(fallback_iso, fallback_iso)


def allen_relation(a, b):
    """a, b are (start, end) datetime tuples with start <= end. This is the
    complete, deterministic implementation of Allen's 13 interval relations
    -- the ONLY place in the pipeline where temporal LOGIC (as opposed to
    temporal EXTRACTION) happens, and it is pure arithmetic, never an LLM
    call, so it cannot hallucinate."""
    a_s, a_e = a
    b_s, b_e = b
    if a_e < b_s:
        return "before"
    if a_s > b_e:
        return "after"
    if a_e == b_s:
        return "meets"
    if a_s == b_e:
        return "met_by"
    if a_s == b_s and a_e == b_e:
        return "equals"
    if a_s == b_s and a_e < b_e:
        return "starts"
    if a_s == b_s and a_e > b_e:
        return "started_by"
    if a_e == b_e and a_s > b_s:
        return "finishes"
    if a_e == b_e and a_s < b_s:
        return "finished_by"
    if a_s > b_s and a_e < b_e:
        return "during"
    if a_s < b_s and a_e > b_e:
        return "contains"
    if a_s < b_s < a_e < b_e:
        return "overlaps"
    if b_s < a_s < b_e < a_e:
        return "overlapped_by"
    return "overlaps"  # last-resort fallback for any residual boundary case


def load_chunk_lookup(global_map):
    lookup = {}
    for pid in {v["person_id"] for v in global_map.values()}:
        with open(os.path.join(IDX_DIR, f"patient_{pid}", "chunks.jsonl")) as f:
            lookup[pid] = {json.loads(l)["uuid"]: json.loads(l) for l in f if l.strip()}
    return lookup


def main():
    with open(os.path.join(IDX_DIR, "global_uuid_map.json")) as f:
        global_map = json.load(f)
    with open(RAW_RESULTS_PATH) as f:
        raw = json.load(f)

    chunk_lookup = load_chunk_lookup(global_map)

    n_events_raw = n_dropped_invalid_uuid = n_dropped_person_mismatch = 0
    n_time_fallback_used = n_dedup_merged = n_event_type_canonicalized = 0
    events_by_patient: dict = {}

    for r in raw["results"]:
        pid = str(r["person_id"])
        parsed = r["parsed"]
        if not parsed or "events" not in parsed:
            continue
        for ev in parsed["events"]:
            n_events_raw += 1
            cu = ev.get("chunk_uuid")
            if cu not in global_map:
                n_dropped_invalid_uuid += 1
                continue
            if global_map[cu]["person_id"] != pid:
                n_dropped_person_mismatch += 1
                continue

            chunk_rec = chunk_lookup[pid][cu]
            fallback_iso = chunk_rec["date"] or chunk_rec["visit_datetime"]

            raw_start, raw_end = ev.get("time_start"), ev.get("time_end")
            if raw_start is None or raw_end is None:
                n_time_fallback_used += 1
            t_start, t_end = parse_time(raw_start, fallback_iso), parse_time(raw_end, fallback_iso)
            if t_end < t_start:
                t_start, t_end = t_end, t_start

            event_type, was_canon = canonicalize_event_type(ev.get("event_type", "other"))
            if was_canon:
                n_event_type_canonicalized += 1

            patient_events = events_by_patient.setdefault(pid, {})
            if cu in patient_events:
                n_dedup_merged += 1
                continue

            patient_events[cu] = {
                "node_id": cu, "chunk_uuid": cu, "person_id": int(pid), "event_type": event_type,
                "time_start": t_start.isoformat(), "time_end": t_end.isoformat(),
                "source_window_idx": r["window_idx"], "_t_start": t_start, "_t_end": t_end,
            }

    all_nodes, all_edges, per_patient_stats = [], [], {}
    for pid, ev_map in events_by_patient.items():
        evs = sorted(ev_map.values(), key=lambda e: (e["_t_start"], e["_t_end"]))
        n = len(evs)
        edges = []
        # (a) consecutive-in-time chain edges: keeps the graph traversable as a timeline
        for i in range(n - 1):
            rel = allen_relation((evs[i]["_t_start"], evs[i]["_t_end"]), (evs[i + 1]["_t_start"], evs[i + 1]["_t_end"]))
            edges.append({"source": evs[i]["node_id"], "target": evs[i + 1]["node_id"], "relation": rel})
        # (b) all non-adjacent OVERLAPPING pairs (captures during/contains/overlaps/equals)
        for i in range(n):
            for j in range(i + 2, n):
                ai, aj = evs[i], evs[j]
                if aj["_t_start"] > ai["_t_end"]:
                    break
                rel = allen_relation((ai["_t_start"], ai["_t_end"]), (aj["_t_start"], aj["_t_end"]))
                if rel not in ("before", "after"):
                    edges.append({"source": ai["node_id"], "target": aj["node_id"], "relation": rel})

        for e in evs:
            all_nodes.append({k: v for k, v in e.items() if not k.startswith("_")})
        all_edges.extend(edges)
        per_patient_stats[pid] = {"n_events": n, "n_edges": len(edges)}

    # Sanity: no clinical text ever leaks into a node.
    for node in all_nodes:
        assert {"text", "raw_text", "section"}.isdisjoint(node.keys()), \
            f"Clinical text field leaked into node: {node.keys()}"

    graph = {
        "metadata": {
            "model": "Qwen/Qwen2.5-32B-Instruct",
            "extraction_method": "windowed (30-visit, stride-25) event extraction via vLLM; "
                                  "Allen's 13 interval relations computed deterministically "
                                  "post-hoc, NOT generated by the LLM.",
            "n_windows_total": raw["n_windows"], "n_windows_parse_failed": raw["n_parse_fail"],
            "n_events_raw_extracted": n_events_raw,
            "n_events_dropped_invalid_uuid": n_dropped_invalid_uuid,
            "n_events_dropped_person_mismatch": n_dropped_person_mismatch,
            "n_events_time_fallback_used": n_time_fallback_used,
            "n_events_deduped_across_overlapping_windows": n_dedup_merged,
            "n_event_type_canonicalized_to_controlled_vocab": n_event_type_canonicalized,
            "n_final_nodes": len(all_nodes), "n_final_edges": len(all_edges),
            "event_type_controlled_vocabulary": sorted(EVENT_TYPE_VOCAB),
            "allen_relation_vocabulary": ALLEN_RELATIONS,
            "node_schema": "{node_id, chunk_uuid, person_id, event_type, time_start, time_end, "
                            "source_window_idx} -- NO clinical text fields.",
            "per_patient": per_patient_stats,
        },
        "nodes": all_nodes,
        "edges": all_edges,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(graph, f, indent=2)
    print(f"[DONE] nodes={len(all_nodes)} edges={len(all_edges)} -> {OUT_PATH}")
    print(json.dumps(graph["metadata"], indent=2, default=str)[:2000])


if __name__ == "__main__":
    main()
