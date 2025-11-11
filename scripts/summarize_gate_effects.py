#!/usr/bin/env python3
import re, sys, json, os

def parse_line(pattern, text, flags=0):
    m = re.search(pattern, text, flags)
    return m.groupdict() if m else None

def main():
    if len(sys.argv) < 2:
        print("Usage: summarize_gate_effects.py <log_path> [out_json]")
        sys.exit(1)
    log_path = sys.argv[1]
    out_json = sys.argv[2] if len(sys.argv) > 2 else ''
    if not os.path.exists(log_path):
        print(f"Log not found: {log_path}")
        sys.exit(2)

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        txt = f.read()

    # Initialize report
    report = {
        'easy_lock': {'locked': None, 'total': None},
        'meta_gate': {'applied': None, 'total': None, 'entropy_thr': None, 'margin_thr': None},
        'keyword_routing': {
            'near_candidates': None, 'locked': None, 'eligible': None,
            'routed_total': None, 'routed_by_hits': None, 'routed_by_tie': None,
            'pairs': None, 'delta_thr': None, 'conf_thr': None,
            'min_hits': None, 'min_hits_wide': None
        },
        'text_gate': {'adjusted': None, 'delta_thr': None, 'pairs': None},
        'pair_gate': {'applied': None}
    }

    # Easy-Lock example: [Easy-Lock] Locked 1180/3140 easy samples (conf >= thresholds)
    m = parse_line(r"\[Easy-Lock\]\s+Locked\s+(?P<locked>\d+)/(\s?)(?P<total>\d+)", txt)
    if m:
        report['easy_lock']['locked'] = int(m['locked'])
        report['easy_lock']['total'] = int(m['total'])

    # Meta-Gate example: [Meta-Gate] Applied meta to 36/3140 uncertain samples (entropy>1.4, margin<0.1)
    m = parse_line(r"\[Meta-Gate\]\s+Applied meta to\s+(?P<applied>\d+)/(\s?)(?P<total>\d+).*entropy>(?P<ent>[0-9.]+), margin<(?P<mar>[0-9.]+)\)", txt)
    if m:
        report['meta_gate']['applied'] = int(m['applied'])
        report['meta_gate']['total'] = int(m['total'])
        report['meta_gate']['entropy_thr'] = float(m['ent'])
        report['meta_gate']['margin_thr'] = float(m['mar'])

    # KeywordRouting Near-boundary line: [KeywordRouting] Near-boundary candidates: X, locked: Y, eligible: Z
    m = parse_line(r"\[KeywordRouting\]\s+Near-boundary candidates:\s+(?P<near>\d+),\s+locked:\s+(?P<locked>\d+),\s+eligible:\s+(?P<elig>\d+)", txt)
    if m:
        report['keyword_routing']['near_candidates'] = int(m['near'])
        report['keyword_routing']['locked'] = int(m['locked'])
        report['keyword_routing']['eligible'] = int(m['elig'])

    # KeywordRouting routed line: [KeywordRouting] Routed: R (by hits: H, by tie→top: T), pairs=[[3, 7], [4, 14]], thr(delta<0.12, conf>0.5, min_hits>=1, min_hits_wide>=2)
    m = parse_line(r"\[KeywordRouting\]\s+Routed:\s+(?P<routed>\d+)\s+\(by hits:\s+(?P<hits>\d+),\s+by tie[^:]*:\s+(?P<tie>\d+)\),\s+pairs=(?P<pairs>\[[^\]]+\]),\s+thr\(delta<(?P<delta>[0-9.]+),\s+conf>(?P<conf>[0-9.]+),\s+min_hits>=?(?P<mh>\d+),\s+min_hits_wide>=?(?P<mhw>\d+)\)", txt)
    if m:
        report['keyword_routing']['routed_total'] = int(m['routed'])
        report['keyword_routing']['routed_by_hits'] = int(m['hits'])
        report['keyword_routing']['routed_by_tie'] = int(m['tie'])
        report['keyword_routing']['pairs'] = m['pairs']
        report['keyword_routing']['delta_thr'] = float(m['delta'])
        report['keyword_routing']['conf_thr'] = float(m['conf'])
        report['keyword_routing']['min_hits'] = int(m['mh'])
        report['keyword_routing']['min_hits_wide'] = int(m['mhw'])

    # Text-Gate example: [Text-Gate] Adjusted 13 near-boundary samples (delta<0.08) for pairs [[3, 7], [4, 14]]
    m = parse_line(r"\[Text-Gate\]\s+Adjusted\s+(?P<adj>\d+)\s+near-boundary samples \(delta<(?P<delta>[0-9.]+)\) for pairs (?P<pairs>\[[^\]]+\])", txt)
    if m:
        report['text_gate']['adjusted'] = int(m['adj'])
        report['text_gate']['delta_thr'] = float(m['delta'])
        report['text_gate']['pairs'] = m['pairs']

    # Pair-Gate: not explicitly logged in predict.py; keep as None

    if out_json:
        os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Saved summary → {out_json}")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
