# attack-on-medvqa
Attack on MedVQA


Running bias field attack
```bash
# Local
uv run scripts/bias_field_attack.py --config cps_8_eps_0p3 --start-index 0 --end-index 10 --output-path tmp --overwrite

# HPC
bash HPC/batch_bias_field_attack.sh
```

Validate bias field attack results
```bash
# Local
uv run scripts/validate_answer.py --config cps_8_eps_0p1 --overwrite

# HPC
bash HPC/batch_validate_answer.sh
```

Combine results from batches
```bash

python HPC/combine_result.py

# Default path : result/MedVLM-R1/bias_field_attack
python combine_result.py path/to/attack/result/folder
```