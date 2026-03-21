#!/usr/bin/env python3
"""ROC/PR analysis of QKI benchmark: rMATS vs BRAID on 105 RT-PCR events."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from pathlib import Path

# ── Load data ────────────────────────────────────────────────────────────────
with open('/home/keunsoo/projects/24_BRAID/benchmarks/results/qki_rmats_benchmark_results.json') as f:
    data = json.load(f)

# Only use matched events (ones that have both rMATS and BRAID scores)
validated_matched = [e for e in data['validated'] if e.get('matched', False)]
failed_matched = [e for e in data['failed'] if e.get('matched', False)]

# Unmatched validated = no rMATS/BRAID hit → treat as undetected positives (score=0)
validated_unmatched = [e for e in data['validated'] if not e.get('matched', False)]
failed_unmatched = [e for e in data['failed'] if not e.get('matched', False)]

print(f"Validated: {len(data['validated'])} total, {len(validated_matched)} matched, {len(validated_unmatched)} unmatched")
print(f"Failed:    {len(data['failed'])} total, {len(failed_matched)} matched, {len(failed_unmatched)} unmatched")

# Build arrays: label=1 for validated (true positive), label=0 for failed (true negative)
# Include ALL 105 events; unmatched get score=0
all_events = []
for e in validated_matched:
    all_events.append({
        'label': 1,
        'rmats_fdr': e['rmats_fdr'],
        'rmats_dpsi': e['rmats_dpsi'],
        'braid_prob': e['braid_dpsi_prob_abs_ge_cutoff'],
        'braid_dpsi': e['braid_dpsi'],
        'significant': e.get('significant', False),
        'high_confidence': e.get('high_confidence', False),
        'supported_differential': e.get('supported_differential', False),
        'near_strict': e.get('near_strict', False),
        'gene': e['gene'],
    })
for e in validated_unmatched:
    all_events.append({
        'label': 1,
        'rmats_fdr': 1.0,  # not detected → worst score
        'rmats_dpsi': 0.0,
        'braid_prob': 0.0,
        'braid_dpsi': 0.0,
        'significant': False,
        'high_confidence': False,
        'supported_differential': False,
        'near_strict': False,
        'gene': e['gene'],
    })
for e in failed_matched:
    all_events.append({
        'label': 0,
        'rmats_fdr': e['rmats_fdr'],
        'rmats_dpsi': e['rmats_dpsi'],
        'braid_prob': e['braid_dpsi_prob_abs_ge_cutoff'],
        'braid_dpsi': e['braid_dpsi'],
        'significant': e.get('significant', False),
        'high_confidence': e.get('high_confidence', False),
        'supported_differential': e.get('supported_differential', False),
        'near_strict': e.get('near_strict', False),
        'gene': e['gene'],
    })
for e in failed_unmatched:
    all_events.append({
        'label': 0,
        'rmats_fdr': 1.0,
        'rmats_dpsi': 0.0,
        'braid_prob': 0.0,
        'braid_dpsi': 0.0,
        'significant': False,
        'high_confidence': False,
        'supported_differential': False,
        'near_strict': False,
        'gene': e.get('gene', 'unknown'),
    })

print(f"Total events for analysis: {len(all_events)} (should be 105)")

labels = np.array([e['label'] for e in all_events])
n_pos = labels.sum()
n_neg = len(labels) - n_pos
print(f"Positives (validated): {n_pos}, Negatives (failed): {n_neg}")

# ── Compute scores for 3 methods ────────────────────────────────────────────
# Method 1: rMATS FDR only → score = 1 - FDR
score_rmats_fdr = np.array([1.0 - e['rmats_fdr'] for e in all_events])

# Method 2: rMATS FDR + |dPSI| combined → score = (1 - FDR) * |dPSI|
score_rmats_combined = np.array([
    (1.0 - e['rmats_fdr']) * abs(e['rmats_dpsi']) for e in all_events
])

# Method 3: BRAID posterior P(|dPSI| >= 0.1)
score_braid = np.array([e['braid_prob'] for e in all_events])

methods = {
    'rMATS FDR': score_rmats_fdr,
    'rMATS FDR×|dPSI|': score_rmats_combined,
    'BRAID P(|dΨ|≥0.1)': score_braid,
}

# ── 1. ROC Curves ───────────────────────────────────────────────────────────
print("\n" + "="*70)
print("1. ROC ANALYSIS")
print("="*70)

roc_results = {}
for name, scores in methods.items():
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    roc_results[name] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': roc_auc}
    print(f"  {name:25s}  AUC = {roc_auc:.4f}")

# ── 2. Precision-Recall Curves ──────────────────────────────────────────────
print("\n" + "="*70)
print("2. PRECISION-RECALL ANALYSIS")
print("="*70)

pr_results = {}
for name, scores in methods.items():
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    pr_auc = average_precision_score(labels, scores)
    pr_results[name] = {'precision': precision, 'recall': recall, 'thresholds': thresholds, 'auc': pr_auc}
    print(f"  {name:25s}  PR-AUC = {pr_auc:.4f}")

# ── 3. dPSI Threshold Table ─────────────────────────────────────────────────
print("\n" + "="*70)
print("3. dPSI THRESHOLD TABLE")
print("="*70)

def count_filter(events, filter_fn):
    """Count TP, FP, FN, TN for a given filter."""
    tp = sum(1 for e in events if e['label'] == 1 and filter_fn(e))
    fp = sum(1 for e in events if e['label'] == 0 and filter_fn(e))
    fn = sum(1 for e in events if e['label'] == 1 and not filter_fn(e))
    tn = sum(1 for e in events if e['label'] == 0 and not filter_fn(e))
    return tp, fp, fn, tn

filters = {
    'FDR<0.05': lambda e: e['rmats_fdr'] < 0.05,
    'FDR<0.05 + |dPSI|>0.1': lambda e: e['rmats_fdr'] < 0.05 and abs(e['rmats_dpsi']) > 0.1,
    'FDR<0.05 + |dPSI|>0.2': lambda e: e['rmats_fdr'] < 0.05 and abs(e['rmats_dpsi']) > 0.2,
    'FDR<0.05 + |dPSI|>0.3': lambda e: e['rmats_fdr'] < 0.05 and abs(e['rmats_dpsi']) > 0.3,
    'BRAID supported': lambda e: e['supported_differential'],
    'BRAID high-confidence': lambda e: e['high_confidence'],
    'BRAID near-strict': lambda e: e['near_strict'],
}

threshold_table = []
print(f"\n  {'Filter':<30s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'TN':>4s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'FPR':>7s}")
print("  " + "-"*90)
for name, fn in filters.items():
    tp, fp, fn_count, tn = count_filter(all_events, fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn_count) if (tp + fn_count) > 0 else 0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"  {name:<30s} {tp:4d} {fp:4d} {fn_count:4d} {tn:4d} {prec:7.3f} {recall:7.3f} {f1:7.3f} {fpr_val:7.3f}")
    threshold_table.append({
        'filter': name, 'tp': tp, 'fp': fp, 'fn': fn_count, 'tn': tn,
        'precision': round(prec, 4), 'recall': round(recall, 4),
        'f1': round(f1, 4), 'fpr': round(fpr_val, 4),
    })

# ── 4. NNV (Number Needed to Validate) ──────────────────────────────────────
print("\n" + "="*70)
print("4. NUMBER NEEDED TO VALIDATE (NNV)")
print("="*70)
print("  NNV = (TP + FP) / TP  (lower is better, 1.0 = perfect)")
print()

nnv_results = []
for row in threshold_table:
    tp, fp = row['tp'], row['fp']
    nnv = (tp + fp) / tp if tp > 0 else float('inf')
    print(f"  {row['filter']:<30s}  NNV = {nnv:.3f}  ({tp+fp} calls, {tp} true)")
    nnv_results.append({'filter': row['filter'], 'nnv': round(nnv, 4), 'calls': tp + fp, 'true_positives': tp})

# ── 5. Save results JSON ────────────────────────────────────────────────────
output = {
    'metadata': {
        'total_events': len(all_events),
        'positives_validated': int(n_pos),
        'negatives_failed': int(n_neg),
        'validated_matched': len(validated_matched),
        'validated_unmatched': len(validated_unmatched),
        'failed_matched': len(failed_matched),
        'failed_unmatched': len(failed_unmatched),
    },
    'roc_auc': {name: round(r['auc'], 4) for name, r in roc_results.items()},
    'pr_auc': {name: round(r['auc'], 4) for name, r in pr_results.items()},
    'threshold_table': threshold_table,
    'nnv': nnv_results,
}

out_path = '/home/keunsoo/projects/24_BRAID/benchmarks/results/qki_roc_analysis.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {out_path}")

# ── 6. Generate figure ──────────────────────────────────────────────────────
colors = {'rMATS FDR': '#1f77b4', 'rMATS FDR×|dPSI|': '#ff7f0e', 'BRAID P(|dΨ|≥0.1)': '#2ca02c'}

fig, axes = plt.subplots(1, 2, figsize=(18/2.54, 7/2.54), dpi=300)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 7

# Panel (a): ROC
ax = axes[0]
for name in methods:
    r = roc_results[name]
    label = f"{name} (AUC={r['auc']:.3f})"
    ax.plot(r['fpr'], r['tpr'], color=colors[name], linewidth=1.2, label=label)
ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=7)
ax.set_ylabel('True Positive Rate', fontsize=7)
ax.set_title('ROC Curve', fontsize=8, fontweight='bold')
ax.legend(fontsize=5.5, loc='lower right', frameon=True, fancybox=False, edgecolor='gray')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.tick_params(labelsize=6)
ax.text(-0.15, 1.05, 'a', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

# Panel (b): PR
ax = axes[1]
for name in methods:
    r = pr_results[name]
    label = f"{name} (AP={r['auc']:.3f})"
    ax.plot(r['recall'], r['precision'], color=colors[name], linewidth=1.2, label=label)
baseline = n_pos / (n_pos + n_neg)
ax.axhline(y=baseline, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
ax.set_xlabel('Recall', fontsize=7)
ax.set_ylabel('Precision', fontsize=7)
ax.set_title('Precision-Recall Curve', fontsize=8, fontweight='bold')
ax.legend(fontsize=5.5, loc='lower left', frameon=True, fancybox=False, edgecolor='gray')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.tick_params(labelsize=6)
ax.text(-0.15, 1.05, 'b', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

plt.tight_layout()

fig_dir = Path('/home/keunsoo/projects/24_BRAID/paper/figures')
for ext in ['pdf', 'png', 'jpg']:
    fig.savefig(fig_dir / f'fig_roc_qki.{ext}', dpi=300, bbox_inches='tight')
    print(f"Saved {fig_dir / f'fig_roc_qki.{ext}'}")

plt.close()
print("\nDone.")
