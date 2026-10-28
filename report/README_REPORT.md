# Rebuild the report (optional)

The submitted PDF and its figure are already included. These dependencies are
only for rebuilding the document, not training, prediction or scoring.

Use Python 3.12 and install `report/requirements-report.txt` in a separate
environment. From the repository root:

```text
python report/plot_results.py
pdftoppm -singlefile -scale-to 2200 -png code/.environment/report-figure.pdf report/figures/validation_results
python report/build_report.py
```

`pdftoppm` is supplied by Poppler. If the figure is not being rebuilt, the last
command alone uses the already supplied PNG. The report is five A4 pages.
Numerical chart inputs come from `code/evidence/validation_analysis.json`.
