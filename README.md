# TC_study

## Installation
After having downloaded the code, in a virtual environment type
```bash
pip install -e .[tf]
``` 
for non-GPU architecture or
```bash
pip install -e .[tf-gpu]
``` 
if you want to use a GPU.
You now need to download the pre-trained models and dataset that are used in the experiment:
```bash
tcs_download models <model_path>
tcs_download dataset <data_path>
```
Once this is done, if you choose `<data_path>` to be different from your current directory,
define a new environment variable `DISENTANGLEMENT_LIB_DATA` so that disentanglement lib can access
the dataset.

## Reproducing experiments

### Computing the passive, mixed and active variables indexes using:
**This needs to be done first** as other parts of the experiment relies on these results.
```bash
tcs_experiment fv <model_path>
```
where `<model_path>` is the absolute path to the folder where you have downloaded the models.

### Computing unsupervised scores
You can reproduce the unsupervised metrics scores with
```bash
tcs_experiment um <model_path> {mean,sampled}[--overwrite]
```
For example, to reproduce the complete experiment:
```bash
tcs_experiment um <model_path> mean
tcs_experiment um <model_path> sampled
```

### Computing downstream tasks
You can reproduce the downstream tasks results with
```bash
tcs_experiment dt <model_path> {mean,sampled} {logistic_regression_cv,gradient_boosting_classifier} [--overwrite]
```
For example, to reproduce the complete experiment:
```bash
tcs_experiment dt <model_path> mean logistic_regression_cv
tcs_experiment dt <model_path> sampled logistic_regression_cv
```

### Aggregating the results
Before doing any visualization, the results needs to be aggregated using
```bash
tcs_aggregate_results <model_path> <output_path>
```
where `<output_path>` is the folder in which your aggregated results will be stored.

### Computing the figures
Once the results aggregated, the figures given in the paper can be reproduced with
`tcs_visualize_results`.

#### Visualizing passive variable relationship with unsupervised scores
```bash
tcs_visualize_results pv <results_path> <output_path> <metric> 
```
where `<results_path>` is the path to your aggregated results, `<output_path>` the path where the
figures will be stored and `<metric>` the metric to visualize. The metric can
be:
- gaussian_total_correlation
- mutual_info_score
For example to see the relationship between total correlation and passive variables:
```bash
tcs_visualize_results pv <results_path> <output_path> gaussian_total_correlation
```  

#### Visualizing the impact of truncated representations on unsupervised scores
```bash
tcs_visualize_results ts <results_path> <output_path> <metric> 
```
where `<results_path>` is the path to your aggregated results, `<output_path>` the path where the
figures will be stored and `<metric>` the metric to visualize. The metric can
be:
- gaussian_total_correlation
- mutual_info_score
For example to see the impact of truncated representations on total correlation:
```bash
tcs_visualize_results ts <results_path> <output_path> gaussian_total_correlation
```

#### Visualizing the impact of truncated representations on downstream tasks
```bash
tcs_visualize_results dt <results_path> <output_path>
```
where `<results_path>` is the path to your aggregated results, and `<output_path>` the path where the
figures will be stored.

#### Observing correlation of passive variables

First the model need to be retrained so that it is saved at multiple timesteps

```bash
tcs_experiment tr <output_path> <model_num>
```
where output_path is the path where the model will be saved and model_num the ID of the model
to train. Please refer to disentanglement lib to get the model ids.

Once the model has been retrained, the correlation and covariance scores can be retrieved
as follows:

```Python
import glob
import json
import numpy as np

files = glob.glob("<output_path>/<model_num>/*/metrics/mean/truncated_unsupervised/results/aggregate/evaluation.json")
corrs = []
covars = []
for file in files:
    with open(file) as f:
        res = json.load(f)
        corrs.append(np.array(res["evaluation_results.correlation_matrix"]))
        covars.append(np.array(res["evaluation_results.covariance_matrix"]))
```
where model_num and output_path are the parameters you choose during the training step.

To observe factor to factor relationship over time one can transform the existing numpy arrays using:
```Python
import pandas as pd

df = []

for i in range(10):
    for j in range(10):
        df += [{"factor_1": i, "factor_2": j, "correlation": corrs[n][i, j], "covariance":covars[n][i, j], 
                "step": n} for n in range(300)]
df = pd.DataFrame(df)
```