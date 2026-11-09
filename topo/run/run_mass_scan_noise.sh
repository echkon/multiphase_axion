mkdir -p ../logs

MASSES=(0.1 0.127 0.162 0.207 0.264 0.336 0.428 0.546 0.695 0.886 1.13 1.44 1.83 2.34 2.98 3.79 4.83 6.16 7.85 10)

for m in "${MASSES[@]}"; do
  for model in KSVZ DFSZ; do
    MASS_UEV=$m MODEL=$model nohup python3 recover_mass_noise.py \
        > ../logs/noise_${model}_m${m}ueV.log 2>&1 &
  done
done