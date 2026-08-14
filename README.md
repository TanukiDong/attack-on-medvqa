# attack-on-medvqa
Attack on MedVQA


Running bias field attack
```bash
# Running locally without environment activated
uv run scripts/bias_field_attack.py --config configs/cps_8_eps_0p3.yaml --start-index 0 --end-index 10
# Running locally with uv environment activated
uv run --active scripts/bias_field_attack.py --config configs/tmp.yaml --start-index 0 --end-index 10 --output-path tmp
```

Combine results from batches
```bash

python HPC/combine_result.py

# Default path : result/MedVLM-R1/bias_field_attack
python combine_result.py path/to/attack/result/folder
```