# Three runs back to back: Phase 1 recipe (rank 16, lr 1e-4), then a higher learning rate, then a higher LoRA rank.
# Setup fixed from the first sweep: all four combos were within noise, so 512 px (cheapest) and " A" (the token the model emits).
cd /content
for cfg in "16 1e-4" "16 3e-4" "64 1e-4"; do
  set -- $cfg
  out=/content/phase1_smol/r$1_lr$2
  mkdir -p $out
  OUT_DIR=$out LORA_RANK=$1 LR=$2 IMAGE_SIDE=512 LETTER_STYLE=space python -u phase1_smolvlm.py > $out/run.log 2>&1
  echo "finished $out (exit $?)" >> /content/phase1_smol/progress.txt
done
