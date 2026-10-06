#!/bin/bash
# Usage: ./tools/regression_check.sh <out_dir>
# Runs a short seeded version of every flow and writes the metrics to <out_dir>/metrics.txt.
# Seeded runs are deterministic, so two metrics files can be compared with diff to check
# that a change did not alter the behaviour (see doc/development.md).
# Needs the INCLUDE and INCLUDE_official datasets in HANDCRAFT_DATA (default /disco1/datasets).
set -u
OUT=$(realpath -m "$1"); REPO=$(cd "$(dirname "$0")/.." && pwd); C=$REPO/src/configs/INCLUDE
DATA_DIR=${HANDCRAFT_DATA:-/disco1/datasets}
export WANDB_MODE=offline CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
rm -rf $OUT; mkdir -p $OUT/cfg; cd $REPO
mk(){ sed -E "s/total_steps: [0-9]+/total_steps: $3/; s/synth_total_steps: [0-9]+/synth_total_steps: 1/" $1 > $OUT/cfg/$2.yaml; }
mk $C/ViT/official-nm-nodct.yaml vit_new 2
mk $C/ViT/official-nm-pad64.yaml vit_pad64 2
mk $C/ViT/official-nm-handmask.yaml vit_handmask 2
mk $C/ViT/official-all.yaml vit_all 2
mk $C/stgcn/official-lr5.yaml stgcn 2
mk $C/ViT/original-pad-128x2.yaml vit_old 2
mk $C/conv1d/DCT-depth4-oc-pad.yaml conv1d 2
mk $C/CsiMLPe/depth_big_noise_0.1.yaml gen_fwd 2
mk $C/CsiMLPe/depth_big_noise_0.1-reversed.yaml gen_rev 2
mk $C/ViT/original-pad-synth25-425.yaml vit_synth 3
run(){ # name, data, extra args...
  n=$1; d=$2; shift 2
  python src/main.py -data $d -cfg $OUT/cfg/$n.yaml -save $OUT/$n/ --project ref --num_workers 2 --prefetch_factor 2 -every 1 --print_every 1 -mpc --seed 42 "$@" > $OUT/$n.out 2> $OUT/$n.err
  echo "== $n (exit $?)" >> $OUT/metrics.txt
  grep -E "Test Top 1-acc|Test Loss|Best Top|Best MPJPE|dataset size|Dataset saved" $OUT/$n.out | sed -E 's/^\[INFO\] [0-9-]+ [0-9:]+ > //' >> $OUT/metrics.txt
}
OFF=$DATA_DIR/INCLUDE_official/; ORI=$DATA_DIR/INCLUDE/
for n in vit_new vit_pad64 vit_handmask vit_all stgcn; do run $n $OFF --mode classification -t --test; done
run vit_old $ORI --mode classification -t --test
run conv1d $ORI --mode classification -t --test
run gen_fwd $ORI --mode cond_prediction -t
run gen_rev $ORI --mode cond_prediction -t --reverse
F=$(ls -d $OUT/gen_fwd/checkpoints/*/ | head -1); R=$(ls -d $OUT/gen_rev/checkpoints/*/ | head -1)
n=gen_rev; python src/main.py --mode cond_prediction -sd -data $ORI -cfg $OUT/cfg/gen_rev.yaml -save $OUT/gen_data/ -best --project ref --num_workers 2 --prefetch_factor 2 -mpc --seed 42 -ckpt $F -tg -r_ckpt $R --sd_num 2 > $OUT/gen_data.out 2> $OUT/gen_data.err
echo "== gen_data (exit $?)" >> $OUT/metrics.txt
G=$(ls -d $OUT/gen_data/generated_datasets/*/ 2>/dev/null | head -1)
( cd "$G" 2>/dev/null && echo "files: $(find poses -name '*.npy' | wc -l) rows: $(wc -l < instances.csv)" && find poses -name '*.npy' | sort | head -40 | xargs md5sum | md5sum ) >> $OUT/metrics.txt
run vit_synth $ORI --mode classification -t --test -s_data $G
for f in $OUT/*.err; do grep -qE "Traceback" $f && echo "FAILED: $(basename $f)"; done >> $OUT/metrics.txt
echo "written $OUT/metrics.txt"
