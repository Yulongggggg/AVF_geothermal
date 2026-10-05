Returned six-reservoir NUTS results — current paper revision, 2026-10-05

Scope
Prespecified test_01–test_06; four chains per reservoir; same POD1024+64 checkpoint
ed04348969fe27903bc9bd084781164f65c610c7e46da0ff75bee60553fdc18e.
No historical-model NUTS values are substituted. The 24-reservoir MAP/ES-MDA
comparison remains distinct from this matched six-reservoir comparison.

Pinned primary data
Repository: https://github.com/Yulongggggg/OL-geothermal-final-scheme
Returned result commit: 967b99977660d6b0e6651abc03a51d3ec4c65033
Path: handoffs/workstationd-nuts-20261001-results
Input handoff commit: 927574a (same original frozen bundle also at 137d61a)
Path: handoffs/workstationd-nuts-20261001
Original MANIFEST SHA256:
c35f9dcf57b41e54f914055d01f4ac1a9367b8bbe316bf3387ecd5e71bc88c64
Use Git LFS when downloading the pinned branches. All 177 returned checksum
entries and model/protocol/case/calibration/chain bindings were independently
verified. Full chains and member fields remain in that pinned results repository.

Independent calculations
reproduction/verify_nuts_return_20261005.py recomputes rank-normalized split Rhat,
bulk/tail ESS across all 4096 coordinates from four x 1024 raw retained draws;
field RMSE from posterior mean fields; log-permeability RMSE; predictive P50
absolute error; and empirical P5–P95 width from the returned member arrays.
No reservoirs are dropped. Each forecast uses 64 evenly spaced draws per chain,
256 total. Within-case spatial RMSE is averaged equally across six reservoirs;
sample SD uses ddof=1. Well MAE is the mean of reservoir-level absolute errors.
Timing sums the four sequential chains on one RTX A4000 and includes warmup;
it excludes loading, later forecast generation and idle time before merging.
MAP/ES-MDA use four CPU threads, so this is not a same-resource comparison.

Files
per_reservoir.csv: six rows, primary errors/widths, timing and diagnostics.
summary.json: means, sample SD and SE; diagnostic summaries are for checking,
not additional performance metrics.
coordinate_diagnostics.npz: independently recomputed diagnostic vectors.
test_01_plot_fields.npz: compact posterior/truth arrays used in Fig. 5.
test_01_well_curves.npz: 256 balanced samples x 11 times x 2 channels x 2 blocks.
Temperature and pressure are paired to producer and injector blocks respectively.
Figures plot pressure ensemble means, while well accuracy tables score P50.

Figure reproduction
reproduction/compute_nuts_curve_20261005.py propagates the balanced draws through
all eleven times using the frozen original bundle. The maximum ten-year
per-member discrepancy against returned dense GPU readouts was 5.22414e-5,
below the existing 5e-4 absolute plotting consistency threshold.
reproduction/render_nuts_fig5_20261005.py uses the original Fig. 5 renderer,
pressure panel and zoom routines, with the current prior/MAP/ES-MDA inputs.
Generated adapters and all input hashes are retained in
scientific_audit_20261002/evidence/nuts_20261005/figure/.
The other twenty current figure assets and the Introduction are unchanged.
Scripts retain explicit original workspace roots for the source and HPC inputs;
map those roots to downloaded bundles/local data to run on another host. The
verification and propagation scripts require a finite CPU Slurm allocation.
The small-file integrity check verify_bundle.py does not require HPC inputs.

Limitations retained
All six cases meet registered coordinate thresholds, with zero divergences.
Every retained transition uses 127 potential evaluations at the depth-seven cap;
coordinate thresholds alone do not prove global exploration of the posterior.
The original workstation strict smoke failed on cross-device FP32 numerical
tolerances and bitwise GPU gradient replay. The returned deviation log records
the decision to proceed. This revision does not relabel that gate as passed or
claim bitwise GPU trajectory reproducibility. Raw failure logs, independent
verification and the explicit manuscript changes are retained in evidence/.
No new CRPS, likelihood-score or coverage columns are added to any paper table.

Extension
The remaining test_07–test_24 (72 chains) were separately handed off at commit
137d61a05c174de0a3d5cc1d0b1db28eee6f4cc8, path
handoffs/workstationd-nuts-extension-20261005, with a sibling CONTINUATION.md.
The original six-case bundle is unchanged. The extension is a subsequent cohort
expansion, not an originally prespecified 24-case NUTS experiment. It is not
included in this manuscript's results. Both original and extension manifests
passed an independent GitHub/LFS download check (82 and 117 files respectively).
