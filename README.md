## MTuner-Megatron


### Exp1: overall results (Figure 8)


Step 1: Run and collect all the results
```
# below scripts can run in parallel (on different machines using Slurm)
# Cost about 1 hour in total
bash scripts/run_megatron.sh # results saved to megatron_output
bash scripts/run_flux.sh # results saved to flux_output
bash scripts/run_mtuner.sh # results saved to mtuner_output

```

Step 2: plot the figure using `plot_figure8.ipynb`


### Exp2: overhead analysis (Table 2)

```
# Use profile data (stored in logs) to search for memory plan
bash scripts/run_dp.sh
# search time is shown as "strategy dump to impls-70-8192.pkl, time cost 86.36314463615417"
```