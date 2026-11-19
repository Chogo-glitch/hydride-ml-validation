# hydride-ml-validation

Code and data for: *From validation score to usable capacity: a critical review of machine learning for hydrogen storage in metal hydrides* (Ogunnigbo, Beneke and Oosthuizen, submitted to Digital Discovery).

## Contents
| File | Content |
|---|---|
| Supplementary_Information.docx | Supplementary Notes S1–S5 and Tables S1–S4 |
| Supplementary_Data_S1_appraisal.csv | Table 1 (appraisal of 20 studies) in machine-readable form |
| Supplementary_Code_S1_appraisal_counts.py | Recomputes the appraisal counts (Sections 2.4, 5.2, 7.1) |
| Supplementary_Code_S2_reanalysis.py | Grouped re-analysis of ML-HydPARK v0.0.5 (Section 9, Table 3, Fig. 7) |
| Supplementary_Code_S3_system_units.py | Operating-unit conversions (Figs. 3–4, Section 8) |
| Reanalysis_Colab.ipynb | Runs Code S2 in a web browser (Google Colab) |
| summary_table3.csv, fold_metrics.csv, predictions.csv, coverage_by_family.csv, mpea_entries.csv | Supplementary Data S2: outputs of Code S2 |

## Reproducing the re-analysis
Input data are from https://github.com/mwitman1/HEAhydrideMLv2 (folder `hea_h2storage/ml_data`):

```
python Supplementary_Code_S2_reanalysis.py --data ML-HYDPARK_v0.0.5.csv \
    --features train_fea_v0.0.5.csv --mpea-ref ML-HYDPARK_v0.0.0.csv --out reanalysis_out
```

Requires Python 3 with numpy, pandas, scipy and scikit-learn (tested with scikit-learn 1.9.1). Runtime is about 15–20 minutes on two cores. Codes S1 and S3 need only the Python standard library.

## Licence
Code: MIT. ML-HydPARK data: CC BY 4.0, Witman et al., doi:10.5281/zenodo.10680097.
