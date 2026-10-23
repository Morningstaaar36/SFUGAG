# SFUGAG

Source-free, label-free fair adaptation of a GNN: only a frozen source task encoder and a sensitive head cross from the source graph, and on the target the method reads node features and edges only.

**Paper under review.**

## Layout

| Path | What it does |
|---|---|
| `v8/fair_cta/adapt_v8.py` | The method (final model, v8) |
| `v8/fair_cta/chooser.py` | C1, the proxy screen |
| `v8/fair_cta/common.py` | File-access guard, label-free target loader, encoder definitions |
| `v8/fair_cta/adapt_v8_stress.py`, `check_chooser_source.py`, `ablation_table.py` | Stress test, C1 check on source graphs, ablation table |
| `v8/run_v8.sh` | One dataset x seed run |
| `v8/verify_v4.py` | Checks on the guard log and the saved predictions |
| `v8/score_sandbox.py`, `tools/score.py` | Scoring (ACC, AUC, DP, EO) |
| `tools/make_splits.py` | Per-seed split export |
| `sfugag/` | Reference adapter code |

## Running

The source checkpoints and the splits come from the CELL code and data, which are not included here. Place them at `runs/cell/` and `splits/`, then:

```bash
DS=german v8/run_v8.sh <tag> 1111
python v8/verify_v4.py <tag>
python v8/score_sandbox.py --ds german --tags <tag>
```

Requires Python with torch, torch_geometric, numpy, pandas, scipy and scikit-learn.
