# attack-on-medvqa
Attack on MedVQA

Extract samples
```bash
# Local
uv run scripts/extract_samples.py --modality mri --samples 10 --overwrite

# HPC
sbatch HPC/extract_samples.sh mri 10
bash HPC/batch_extract_samples.sh
```

Running bias field attack
```bash
# Local
uv run scripts/bias_field_attack.py --config cps_8_eps_0p3 --modality mri --start-index 0 --end-index 10 --output-path tmp --overwrite

# HPC
sbatch HPC/bias_field_attack.sh cps_8_eps_0p3 mri
bash HPC/batch_bias_field_attack.sh
```

Validate bias field attack results
```bash
# Local
uv run scripts/validate_answer.py --config cps_8_eps_0p3 --modality mri --overwrite

# HPC
sbatch HPC/validate_answer.sh cps_8_eps_0p3 mri
bash HPC/batch_validate_answer.sh
```

Combine results from batches
```bash

python HPC/combine_result.py

# Default path : result/MedVLM-R1/bias_field_attack
python combine_result.py path/to/attack/result/folder
```