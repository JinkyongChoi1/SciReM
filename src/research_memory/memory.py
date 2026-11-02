"""Shared evidence memory. Execute indexing/retrieval only on approved compute hosts."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import uuid
from datetime import datetime, timezone


def _tokens(text):
    return re.findall(r"[\w-]+", text.casefold())


def _dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("IDs must be short safe identifiers")
    return value


def _validate(record):
    record = dict(record)
    for key in ("memory_id", "agent_id", "track_id"):
        _identifier(record.get(key))
    for key in ("text", "source"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"Missing {key}: every memory requires source provenance")
    if not isinstance(record.get("aliases", []), list) or any(not isinstance(a, str) or not a.strip() for a in record.get("aliases", [])):
        raise ValueError("aliases must be nonempty strings")
    for edge in record.get("edges", []):
        _identifier(edge.get("target"))
        if not isinstance(edge.get("relation"), str) or not edge["relation"].strip():
            raise ValueError("edge requires relation")
        if not isinstance(edge.get("source"), str) or not edge["source"].strip():
            raise ValueError("edge requires its own provenance")
    record.setdefault("memory_type", "evidence")
    record.setdefault("status", "proposed")
    record.setdefault("aliases", [])
    record.setdefault("edges", [])
    record.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    return record


def submit_candidate(workspace, record):
    """Atomically publish an immutable per-agent candidate; never overwrite an ID."""
    record = _validate(record)
    Path(workspace).mkdir(parents=True, exist_ok=True, mode=0o700)
    folder = Path(workspace) / "candidates" / record["agent_id"]
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (record["memory_id"] + ".json")
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def _normalize(vector):
    values = [float(x) for x in vector]
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("Vectors must contain finite numbers")
    norm = math.sqrt(sum(x * x for x in values))
    return [x / norm for x in values] if norm else values


def _lexical(text, vocabulary):
    counts = {}
    for token in _tokens(text):
        counts[token] = counts.get(token, 0) + 1
    return _normalize([counts.get(word, 0) for word in vocabulary])


def _local_model(workspace, model_name):
    # Each embedding allocation uses one CPU; do not create a host-sized Rayon pool.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=model_name, cache_dir=str(Path(workspace) / "models"), threads=1)


def _algorithm_label(backend, strategy):
    base = "lexical-RAG (not semantic)" if backend == "lexical" else "vector-RAG"
    if strategy == "ppr":
        return base + "+anchors+PPR"
    return base + (" [lexical similarity only]" if backend == "lexical" else " [dense-only]")


def build_snapshot(workspace, embedding_backend="lexical", model_name=None):
    """Single writer builds a consistent immutable snapshot then atomically publishes CURRENT."""
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = root / ".index-lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError("Index writer already active; inspect a stale lock manually") from exc
    try:
        records = [_validate(_read(p)) for p in sorted((root / "candidates").glob("*/*.json"))]
        ids = [r["memory_id"] for r in records]
        if len(ids) != len(set(ids)):
            raise ValueError("memory_id must be globally unique across agents")
        edges = []
        for record in records:
            for edge in record["edges"]:
                if edge["target"] not in ids:
                    raise ValueError("Edge target is absent from snapshot")
                edges.append(dict(edge, origin=record["memory_id"]))
        vocabulary = sorted(set(t for r in records for t in _tokens(r["text"])))
        if embedding_backend == "lexical":
            vectors = [_lexical(r["text"], vocabulary) for r in records]
            model_version = "lexical-token-count-v1-not-semantic"
        elif embedding_backend == "external":
            if not model_name:
                raise ValueError("External vectors require explicit model_name/version")
            vectors = [_normalize(r.get("vector", [])) for r in records]
            model_version = model_name
        elif embedding_backend == "fastembed":
            model_version = model_name or "BAAI/bge-small-en-v1.5"
            vectors = [_normalize(v) for v in _local_model(root, model_version).passage_embed([r["text"] for r in records])] if records else []
        else:
            raise ValueError("Unknown embedding backend")
        if vectors and len({len(v) for v in vectors}) != 1:
            raise ValueError("Inconsistent embedding dimensions")
        # shortcut: full deterministic rebuild; add content-hash embedding cache for large corpora.
        snapshot_id = "s-" + uuid.uuid4().hex
        snapshots = root / "snapshots"
        snapshots.mkdir(exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".building-", dir=snapshots))
        _dump(staging / "graph.json", {"nodes": records, "edges": edges})
        _dump(staging / "vectors.json", {"ids": ids, "values": vectors, "vocabulary": vocabulary})
        manifest = {
            "snapshot_id": snapshot_id, "created_at": datetime.now(timezone.utc).isoformat(),
            "backend": embedding_backend, "model_version": model_version,
            "node_count": len(records), "edge_count": len(edges),
            "transition_policy": {"forward": 1.0, "reverse": 0.5, "damping": 0.85},
            "files": {name: _digest(staging / name) for name in ("graph.json", "vectors.json")},
        }
        _dump(staging / "manifest.json", manifest)
        staging.rename(snapshots / snapshot_id)
        fd, temporary = tempfile.mkstemp(prefix=".current-", dir=root)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(snapshot_id + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / "CURRENT")
        return manifest
    finally:
        lock.rmdir()


def _pagerank(ids, edges, seeds, damping=0.85, tolerance=1e-10, max_iterations=200):
    adjacency = {key: {} for key in ids}
    for edge in edges:
        a, b = edge["origin"], edge["target"]
        # Reverse retrieval traversal does not assert a reverse semantic/causal claim.
        adjacency[a][b] = adjacency[a].get(b, 0) + 1.0
        adjacency[b][a] = adjacency[b].get(a, 0) + 0.5
    scores = dict(seeds)
    for iteration in range(max_iterations):
        updated = {key: (1 - damping) * seeds[key] for key in ids}
        for origin, targets in adjacency.items():
            if targets:
                total = sum(targets.values())
                for target, weight in targets.items():
                    updated[target] += damping * scores[origin] * weight / total
            else:
                for target in ids:
                    updated[target] += damping * scores[origin] * seeds[target]
        error = sum(abs(updated[key] - scores[key]) for key in ids)
        scores = updated
        if error < tolerance:
            return scores, iteration + 1, True
    return scores, max_iterations, False


def retrieve(workspace, query, track_id, top_k=5, snapshot_id=None, query_vector=None, strategy="ppr"):
    """Query-grounded RAG + anchors + PPR; returns source excerpts, never authority."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be nonempty")
    _identifier(track_id)
    if strategy not in ("ppr", "dense", "similarity"):
        raise ValueError("strategy must be ppr, dense or similarity")
    if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
        raise ValueError("top_k must be 1..100")
    root = Path(workspace)
    snapshot_id = _identifier(snapshot_id or (root / "CURRENT").read_text().strip())
    folder = root / "snapshots" / snapshot_id
    manifest = _read(folder / "manifest.json")
    if manifest["snapshot_id"] != snapshot_id or any(_digest(folder / name) != digest for name, digest in manifest["files"].items()):
        raise ValueError("Inconsistent or corrupted snapshot")
    graph, vectors = _read(folder / "graph.json"), _read(folder / "vectors.json")
    nodes = [{key: value for key, value in r.items() if key not in ("edges", "vector")} for r in graph["nodes"] if r["track_id"] == track_id and r["status"] != "invalid"]
    ids = [r["memory_id"] for r in nodes]
    edges = [e for e in graph["edges"] if e["origin"] in ids and e["target"] in ids]
    backend = manifest["backend"]
    if not nodes:
        return {"request_id": "r-" + uuid.uuid4().hex, "query": query, "track_id": track_id,
                "snapshot_id": snapshot_id, "backend": backend, "model_version": manifest["model_version"],
                "algorithm": _algorithm_label(backend, strategy), "strategy": strategy,
                "nodes": [], "edges": [], "anchors": [], "selected_ids": [], "ppr": {},
                "brief": "# Research memory evidence packet\n\nNo memory exists in this project snapshot.\n",
                "converged": True, "iterations": 0, "damping": 0.85, "outcome": "no-match",
                "transition_policy": manifest["transition_policy"]}
    if backend == "lexical":
        query_values = _lexical(query, vectors["vocabulary"])
    elif backend == "external":
        if query_vector is None:
            raise ValueError("External vector snapshot requires a query_vector from the same model")
        query_values = _normalize(query_vector)
    elif backend == "fastembed":
        query_values = _normalize(next(_local_model(root, manifest["model_version"]).query_embed(query)))
    else:
        raise ValueError("Unknown snapshot embedding backend")
    if len(query_values) != len(vectors["values"][0]):
        raise ValueError("Query vector dimension mismatch")
    similarities = {key: sum(a * b for a, b in zip(query_values, vector)) for key, vector in zip(vectors["ids"], vectors["values"]) if key in ids}
    query_tokens = _tokens(query)
    explicit_ids = {record["memory_id"] for record in nodes if _tokens(record["memory_id"]) and
                    any(query_tokens[i:i + len(_tokens(record["memory_id"]))] == _tokens(record["memory_id"])
                        for i in range(len(query_tokens)))}
    anchors = []
    for record in nodes:
        if explicit_ids and record["memory_id"] not in explicit_ids:
            continue
        names = [record["memory_id"]] + record["aliases"]
        matched = next((name for name in names if _tokens(name) and any(query_tokens[i:i + len(_tokens(name))] == _tokens(name) for i in range(len(query_tokens)))), None)
        if matched:
            anchors.append({"id": record["memory_id"], "weight": 1.0, "reason": "exact ID/alias: " + matched, "type": record["memory_type"], "source": record["source"]})
    dense = sorted(ids, key=lambda key: (-similarities[key], key))[:max(20, top_k * 4)]
    if not anchors:
        anchors = [{"id": key, "weight": similarities[key], "reason": "lexical overlap seed (not semantic)" if backend == "lexical" else "vector candidate grounded to evidence ID", "type": next(r["memory_type"] for r in nodes if r["memory_id"] == key), "source": next(r["source"] for r in nodes if r["memory_id"] == key)} for key in dense[:3] if similarities[key] > 0]
    total = sum(a["weight"] for a in anchors)
    seeds = {key: 0.0 for key in ids}
    for anchor in anchors:
        anchor["weight"] /= total
        seeds[anchor["id"]] += anchor["weight"]
    scores, iterations, converged = _pagerank(ids, edges, seeds) if anchors and strategy == "ppr" else ({key: 0.0 for key in ids}, 0, True)
    ppr_order = sorted((key for key in ids if scores[key] > 0), key=lambda key: (-scores[key], key))
    dense_order = [key for key in dense if similarities[key] > 0]
    fusion = {}
    for order in ((ppr_order, dense_order) if strategy == "ppr" else (dense_order,)):
        for rank, key in enumerate(order, 1):
            fusion[key] = fusion.get(key, 0) + 1 / (60 + rank)
    selected, seen_results = [], set()
    by_id = {node["memory_id"]: node for node in nodes}
    for key in sorted(fusion, key=lambda key: (-fusion[key], key)) if anchors else []:
        group = by_id[key].get("result_id", key)
        if group in seen_results:
            continue
        seen_results.add(group)
        selected.append(key)
        if len(selected) == top_k:
            break
    for record in nodes:
        key = record["memory_id"]
        record.update(ppr=scores[key], similarity=similarities[key], final_score=fusion.get(key, 0), included=key in selected, anchor_weight=seeds[key], selection_reason="included by reciprocal rank fusion" if key in selected else "outside context budget" if anchors else "no grounded memory")
    brief = "# Research memory evidence packet\n\nRelevance does not establish truth or permission.\n\n"
    if not selected:
        brief += "No grounded relevant memory found.\n"
    for key in selected:
        record = next(r for r in nodes if r["memory_id"] == key)
        brief += f"- [{key}] ({record['status']}) {record['text'][:1200]}\n  Source: {record['source']}\n"
    return {
        "request_id": "r-" + uuid.uuid4().hex, "query": query, "track_id": track_id,
        "snapshot_id": snapshot_id, "backend": backend, "model_version": manifest["model_version"],
        "algorithm": _algorithm_label(backend, strategy),
        "strategy": strategy,
        "nodes": nodes, "edges": edges, "anchors": anchors, "selected_ids": selected,
        "ppr": scores, "brief": brief, "converged": converged, "iterations": iterations,
        "damping": 0.85, "fusion": "RRF k=60, equal dense/PPR weights" if strategy == "ppr" else "dense ranking only", "outcome": "retrieved" if selected else "no-match",
        "transition_policy": manifest["transition_policy"], "context_char_budget": top_k * 1200,
    }
