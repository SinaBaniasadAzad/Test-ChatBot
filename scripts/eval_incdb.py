"""
ارزیابی دقت روی فایل خام تیکت‌ها (فرمت INC_DB.jsonl).

هر خط:
  {"Key": "INC-20689", "Application": "ERP", "Summary": "...", "Description": "...",
   "Labels": {"layer_1": "Incident", "layer_2": "ERP"}}

- ground truth = Labels.layer_1 / layer_2 (نام نمایشی مثل "Incident"/"ERP").
  نام‌ها خودکار به id داخلی نگاشت می‌شوند و کلیدِ "layer_1" با "layer1"ِ taxonomy
  تطبیق داده می‌شود (حذفِ underscore).
- مدل single-shot اجرا می‌شود (بدون سوال تکمیلی) = «دقتِ خامِ مدل».

این ماژول دو لایه دارد:
  • run_evaluation(...) -> dict     محاسبهٔ متریک‌ها (برای متن و داشبورد)
  • main()                          گزارشِ متنیِ CLI

برای داشبوردِ تصویرِ حرفه‌ای: scripts/report.py

اجرا (از ریشهٔ پروژه):
    python -m scripts.eval_incdb data/INC_DB.jsonl --balanced 75 --workers 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier.classifier import Classifier  # noqa: E402


def _norm_key(k: str) -> str:
    """«layer_1» / «Layer 1» -> «layer1» (فقط حروف و عدد)."""
    return "".join(ch for ch in str(k).lower() if ch.isalnum())


def _cap(name: str) -> str:
    """نام نمایشیِ کوتاهِ انگلیسی: «Type / نوع» -> «Type»."""
    return name.split("/")[0].strip() if "/" in name else name.strip()


def _build_maps(tax):
    """نگاشت‌های نام->id برای لایه‌ها و برچسب‌ها (تحملِ نام نمایشی یا id)."""
    layer_key_map: dict[str, str] = {}
    label_map: dict[str, dict[str, str]] = {}
    for layer in tax.layers:
        layer_key_map[_norm_key(layer.id)] = layer.id
        m: dict[str, str] = {}
        for lbl in layer.labels:
            m[lbl.name.strip().lower()] = lbl.id
            m[lbl.id.strip().lower()] = lbl.id
        label_map[layer.id] = m
    return layer_key_map, label_map


def _gt_label(row: dict, layer, label_map) -> str | None:
    """idِ برچسبِ طلاییِ این لایه از روی Labels (یا None اگر نبود/نگاشت نشد)."""
    labels = row.get("Labels") or {}
    for gk, gv in labels.items():
        if _norm_key(gk) == _norm_key(layer.id):
            return label_map[layer.id].get(str(gv).strip().lower())
    return None


def load_rows(path: Path, limit: int | None, balanced: int | None, tax) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))

    if balanced:
        # حداکثر N نمونه از هر ترکیبِ برچسب تا کلاس‌های نادر هم پوشش بگیرند.
        _, label_map = _build_maps(tax)
        buckets: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            combo = tuple(_gt_label(r, layer, label_map) for layer in tax.layers)
            buckets[combo].append(r)
        rows = []
        for items in buckets.values():
            rows.extend(items[:balanced])

    return rows[:limit] if limit else rows


def _field(row: dict, *names: str) -> str:
    for n in names:
        if row.get(n):
            return str(row[n])
    return ""


def compute_cost(tokens: dict, price_in: float, price_cache: float, price_out: float) -> float:
    """هزینهٔ دلاری از روی توکن‌ها (قیمت‌ها به ازای ۱M)."""
    return (
        tokens.get("cache_hit", 0) * price_cache
        + tokens.get("cache_miss", 0) * price_in
        + tokens.get("completion", 0) * price_out
    ) / 1_000_000


def run_evaluation(
    data_path,
    *,
    limit: int | None = None,
    balanced: int | None = None,
    workers: int = 4,
    out_path=None,
    progress: bool = True,
) -> dict:
    """مدل را روی تیکت‌ها اجرا و متریک‌ها را برمی‌گرداند (بدون نمایش)."""
    clf = Classifier()
    tax = clf.taxonomy
    _, label_map = _build_maps(tax)
    rows = load_rows(Path(data_path), limit, balanced, tax)
    total = len(rows)
    if progress:
        print(f"بارگذاری {total} تیکت از {data_path}", file=sys.stderr)

    per_layer_total: dict[str, int] = defaultdict(int)
    per_layer_correct: dict[str, int] = defaultdict(int)
    cls_total = {l.id: defaultdict(int) for l in tax.layers}
    cls_correct = {l.id: defaultdict(int) for l in tax.layers}
    confusion = {l.id: defaultdict(lambda: defaultdict(int)) for l in tax.layers}
    full_total = full_correct = errors = 0
    tok: dict[str, int] = defaultdict(int)
    model_served = None
    t0 = time.perf_counter()
    out_fh = Path(out_path).open("w", encoding="utf-8") if out_path else None

    def run_one(row: dict):
        try:
            out, meta = clf.classify(_field(row, "Summary", "summary"), _field(row, "Description", "description"))
            return row, out, meta, None
        except Exception as e:  # شکستِ یک تیکت کلِ اجرا را نکُشد
            return row, None, {}, str(e)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row, out, meta, err in pool.map(run_one, rows):
            done += 1
            if err:
                errors += 1
            if meta.get("model"):
                model_served = meta["model"]

            u = meta.get("usage") or {}
            pt, ct = u.get("prompt_tokens", 0) or 0, u.get("completion_tokens", 0) or 0
            hit, miss = u.get("prompt_cache_hit_tokens"), u.get("prompt_cache_miss_tokens")
            if hit is None or miss is None:
                hit, miss = 0, pt
            tok["prompt"] += pt
            tok["cache_hit"] += hit
            tok["cache_miss"] += miss
            tok["completion"] += ct

            row_all_ok = True
            row_has_all = True
            rec = {"Key": row.get("Key"), "true": {}, "pred": {}}
            for layer in tax.layers:
                true_id = _gt_label(row, layer, label_map)
                if true_id is None:
                    row_has_all = False
                    continue
                lo = out.layers.get(layer.id) if out else None
                pred_id = lo.top.label if (lo and lo.top) else None
                per_layer_total[layer.id] += 1
                cls_total[layer.id][true_id] += 1
                confusion[layer.id][true_id][pred_id] += 1
                if pred_id == true_id:
                    per_layer_correct[layer.id] += 1
                    cls_correct[layer.id][true_id] += 1
                else:
                    row_all_ok = False
                rec["true"][layer.id] = true_id
                rec["pred"][layer.id] = pred_id
            if row_has_all:
                full_total += 1
                full_correct += int(row_all_ok)
            if out_fh:
                rec["correct"] = row_all_ok and row_has_all
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if progress and done % 25 == 0:
                print(f"... {done}/{total}", file=sys.stderr)

    if out_fh:
        out_fh.close()
    wall = time.perf_counter() - t0

    layers_out = []
    for layer in tax.layers:
        lid = layer.id
        classes = []
        for cid in layer.label_ids:
            t = cls_total[lid].get(cid, 0)
            c = cls_correct[lid].get(cid, 0)
            lbl = layer.get_label(cid)
            classes.append(
                {"id": cid, "name": lbl.name if lbl else cid, "recall": (c / t if t else 0.0), "correct": c, "total": t}
            )
        tot = per_layer_total[lid]
        layers_out.append(
            {
                "id": lid,
                "name": _cap(layer.name),
                "accuracy": (per_layer_correct[lid] / tot if tot else 0.0),
                "correct": per_layer_correct[lid],
                "total": tot,
                "label_ids": list(layer.label_ids),
                "classes": classes,
                "confusion": {t: dict(confusion[lid][t]) for t in confusion[lid]},
            }
        )

    return {
        "n": total,
        "errors": errors,
        "model": model_served,
        "layers": layers_out,
        "overall": {
            "accuracy": (full_correct / full_total if full_total else 0.0),
            "correct": full_correct,
            "total": full_total,
        },
        "tokens": dict(tok),
        "wall_s": wall,
        "latency_ms_avg": (wall * 1000 / total if total else 0.0),
    }


def print_text_report(res: dict, price_in: float, price_cache: float, price_out: float) -> None:
    cost = compute_cost(res["tokens"], price_in, price_cache, price_out)
    t = res["tokens"]
    print("\n================= نتایج ارزیابی (single-shot) =================")
    print(f"مدل: {res.get('model')}   |   تیکت‌ها: {res['n']}   |   خطای فراخوانی: {res['errors']}")
    print(f"توکن‌ها → prompt={t.get('prompt',0):,} (hit={t.get('cache_hit',0):,}/miss={t.get('cache_miss',0):,})  completion={t.get('completion',0):,}")
    print(f"هزینهٔ تخمینی: ${cost:.4f}  (نرخ/1M: in=${price_in}, hit=${price_cache}, out=${price_out} — نرخِ روز را بررسی کن)")
    print(f"زمان کل: {res['wall_s']:.0f}s  |  میانگین: {res['latency_ms_avg']:.0f} ms/تیکت")

    print("\n— دقتِ هر لایه —")
    for L in res["layers"]:
        print(f"  {L['id']} ({L['name']}): {L['accuracy']:.1%}  ({L['correct']}/{L['total']})")
    print(f"\n— دقتِ کلی (هر دو لایه هم‌زمان درست) —\n  {res['overall']['accuracy']:.1%}  ({res['overall']['correct']}/{res['overall']['total']})")

    print("\n— recall هر کلاس —")
    for L in res["layers"]:
        print(f"  {L['id']}:")
        for c in L["classes"]:
            print(f"    {c['name']:<18} {c['recall']:6.1%}  ({c['correct']}/{c['total']})")

    for L in res["layers"]:
        ids = L["label_ids"]
        name = {c["id"]: c["name"] for c in L["classes"]}
        print(f"\n— Confusion «{L['id']}» (سطر=واقعی، ستون=پیش‌بینی؛ ⌀=بدون پیش‌بینی) —")
        print("true\\pred".ljust(18) + "".join(name[p][:16].ljust(18) for p in ids) + "⌀")
        for tr in ids:
            cells = L["confusion"].get(tr, {})
            print(name[tr].ljust(18) + "".join(str(cells.get(p, 0)).ljust(18) for p in ids) + str(cells.get(None, 0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_path", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--balanced", type=int, default=None, help="حداکثر N نمونه از هر ترکیب برچسب")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None, help="ذخیرهٔ پیش‌بینی‌ها (JSONL)")
    ap.add_argument("--price-in", type=float, default=0.27)
    ap.add_argument("--price-cache", type=float, default=0.07)
    ap.add_argument("--price-out", type=float, default=1.10)
    args = ap.parse_args()

    res = run_evaluation(
        args.data_path, limit=args.limit, balanced=args.balanced, workers=args.workers, out_path=args.out
    )
    print_text_report(res, args.price_in, args.price_cache, args.price_out)


if __name__ == "__main__":
    main()
