"""Supplementary Code S1. Recomputes every appraisal count quoted in Sections 2.4, 5.2 and 7.1
from Supplementary Data S1 (Table 1 in machine-readable form). Standard library only.
Usage:  python Supplementary_Code_S1_appraisal_counts.py Supplementary_Data_S1_appraisal.csv"""
import csv, sys
path = sys.argv[1] if len(sys.argv) > 1 else 'Supplementary_Data_S1_appraisal.csv'
rows = list(csv.DictReader(open(path, encoding='utf-8')))
crit = ['T1_provenance', 'T2_generalisation', 'T3_uncertainty', 'T4_verification', 'T5_system_relevance']
fit = [r for r in rows if r['study_type'] == 'fitted-model']
phy = [r for r in rows if r['study_type'] == 'physics-based']
refs = lambda rs: sorted(int(r['ref']) for r in rs)
grouped = [r for r in fit if r['T2_generalisation'] == 'met']
lab_fit = [r for r in fit if r['verification_level'] in ('D2', 'D3')]
lab_phy = [r for r in phy if r['verification_level'] in ('D2', 'D3')]
uq = [r for r in fit if r['per_prediction_uncertainty'] == 'yes']
all5 = [r for r in fit if all(r[c] == 'met' for c in crit)]
d3 = [r for r in rows if r['verification_level'] == 'D3']
print(f'Studies: {len(rows)} ({len(fit)} fitted-model, {len(phy)} physics-based)')
print(f'Fitted-model studies with grouped accuracy (T2 met): {len(grouped)} {refs(grouped)}')
print(f'Fitted-model studies that synthesised and tested alloys (D2/D3): {len(lab_fit)} {refs(lab_fit)}')
print(f'Physics-based studies that synthesised and tested alloys (D2/D3): {len(lab_phy)} {refs(lab_phy)}')
print(f'Overlap of grouped and laboratory-tested fitted-model studies: {sorted(set(refs(grouped)) & set(refs(lab_fit)))}')
print(f'Fitted-model studies with per-prediction uncertainty: {len(uq)} {refs(uq)}')
print(f'Fitted-model studies meeting all five criteria: {len(all5)}')
print(f'Laboratory-tested studies reporting reversible capacity or cycling (D3): {len(d3)} {refs(d3)}')
print('\nCriterion   met  partly  not met  n/a   (all 20 studies)')
for c in crit:
    v = [r[c] for r in rows]
    print(f"{c.split('_')[0]:10s}{v.count('met'):5d}{v.count('partly met'):8d}{v.count('not met'):9d}{v.count('not applicable'):5d}")
