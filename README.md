## MTuner-Megatron

### Env

```bash
# use original env
source ~/.venv/fthub/bin/activate
# or build a new env
python -m ~/.venv/finetune
source ~/.venv/finetune/bin/activate
pip install -r requirements.txt
```

### Exp1: overall results (Figure 8)


Step 1: Run and collect all the results
```bash
# below scripts can run in parallel (on different machines using Slurm)
# Cost about 1 hour in total
bash scripts/run_megatron.sh # results saved to megatron_output
bash scripts/run_flux.sh # results saved to flux_output
bash scripts/run_mtuner.sh # results saved to mtuner_output

```

Step 2: plot the figure using `plot_figure8.ipynb`


### Exp2: overhead analysis (Table 2)

```bash
# Use profile data (stored in logs) to search for memory plan
bash scripts/run_dp.sh
# search time is shown as "strategy dump to impls-70-8192.pkl, time cost 86.36314463615417"
```
