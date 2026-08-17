# attack-on-medvqa
Attack on MedVQA


Running bias field attack
```bash
# Running locally
uv run scripts/bias_field_attack.py --config configs/cps_8_eps_0p3.yaml --start-index 0 --end-index 10 --output-path tmp --overwrite
```

Validate bias field attack results
```bash
uv run scripts/validate_answer.py --result-directory result/MedVLM-R1/bias_field_attack/cps_8_eps_0p1/batch_0/ --overwrite
```

Combine results from batches
```bash

python HPC/combine_result.py

# Default path : result/MedVLM-R1/bias_field_attack
python combine_result.py path/to/attack/result/folder
```