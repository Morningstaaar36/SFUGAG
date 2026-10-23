#!/bin/bash
# One v8 run: copy ONLY CELL's two source checkpoints into the run folder, then the guarded, label-free target
# adaptation (fair_cta/adapt_v8.py). No source-side step: nothing but the encoders crosses over.
# The encoder type and dropout below are the checkpoints' own architecture (CELL config.yaml), not v8 settings.
# usage: [DS=<dataset>] v8/run_v8.sh <tag> <seed> [adapt_v8.py flags...]    DS = pokec|bailA|german|syn
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
tag=$1; seed=$2; shift 2
SB=$(cd "$(dirname "$0")" && pwd); ROOT=$(dirname "$SB")
PY=${PY:-python}
ds=${DS:-pokec}
case $ds in
  pokec)  src=pokec_z;  tgt=pokec_n;  enc=GCN;     drop=0.4;;
  bailA)  src=bailA_2;  tgt=bailA_1;  enc=GCN;     drop=0.5;;
  german) src=german_2; tgt=german_1; enc=vanilla; drop=0.4;;
  syn)    src=syn-2;    tgt=syn-1;    enc=vanilla; drop=0.1;;
  *) echo "unknown DS=$ds"; exit 1;;
esac
d=$SB/runs/$tag/$ds/seed_$seed
if [ -f "$d/DONE" ]; then echo "skip $d (DONE)"; exit 0; fi
case "$d" in "$SB"/runs/*) rm -rf "$d";; *) echo "refusing to clear $d"; exit 1;; esac
mkdir -p "$d/src_model"
for f in inter_train_${enc}_${src}.pth sens_train_MLP_${src}.pth; do
  cp "$ROOT/runs/cell/$ds/seed_$seed/param/$f" "$d/src_model/" || { echo "FAIL: no CELL source model $f"; exit 1; }
done
(cd "$d/src_model" && md5sum *.pth > md5.txt)
mkdir -p "$d/code" && cp "$SB/fair_cta/adapt_v8.py" "$SB/fair_cta/chooser.py" "$SB/fair_cta/common.py" "$d/code/"
echo "--inter_encoder $enc --dropout $drop $*" > "$d/args.txt"
cd "$d"
s=$(date +%s)
"$PY" "$SB/fair_cta/adapt_v8.py" --seed "$seed" --split_dir "$SB/splits/seed_$seed" --source $src --target $tgt \
      --run_dir "$d" --inter_encoder $enc --dropout $drop "$@" > adapt.out 2>&1 \
  || { echo "FAIL $tag $ds seed $seed: $(grep -a -m1 -E 'Error' adapt.out)"; exit 1; }
[ -f pred.npy ] && [ -f source_only.npy ] || { echo "FAIL $tag $ds seed $seed: missing predictions"; exit 1; }
echo $(( $(date +%s) - s )) > seconds.txt; touch DONE
echo "OK $tag $ds seed $seed ($(cat seconds.txt)s) $(python3 -c "import json;d=json.load(open('decision.json'));print(d['mode'],d['choice'],d['chosen'])")"
