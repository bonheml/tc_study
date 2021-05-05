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

### Computing the passive variables using:
**This needs to be done first** as other parts of the experiment relies on these results.
```bash
tcs_experiment pv <model_path>
```
where `<model_path>` is the absolute path to the folder where you have downloaded the models.

### Computing unsupervised scores
You can reproduce the unsupervised metrics scores with
```bash
tcs_experiment um <model_path> {mean,sampled} [--truncate][--overwrite]
```
For example, to reproduce the complete experiment:
```bash
tcs_experiment um <model_path> mean
tcs_experiment um <model_path> mean -t
tcs_experiment um <model_path> sampled
tcs_experiment um <model_path> sampled -t
```

### Computing downstream tasks
You can reproduce the downstream tasks results with
```bash
tcs_experiment dt <model_path> {mean,sampled} {logistic_regression_cv,gradient_boosting_classifier} [--truncate][--overwrite]
```
For example, to reproduce the complete experiment:
```bash
tcs_experiment dt <model_path> mean logistic_regression_cv
tcs_experiment dt <model_path> mean logistic_regression_cv -t
tcs_experiment dt <model_path> sampled logistic_regression_cv
tcs_experiment dt <model_path> sampled logistic_regression_cv -t
tcs_experiment dt <model_path> mean gradient_boosting_classifier
tcs_experiment dt <model_path> mean gradient_boosting_classifier -t
tcs_experiment dt <model_path> sampled gradient_boosting_classifier
tcs_experiment dt <model_path> sampled gradient_boosting_classifier -t
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
- gaussian_wasserstein_correlation
- gaussian_wasserstein_correlation_norm
- mutual_info_score
- adjusted_mutual_info_score
- norm_mutual_info_score
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
- gaussian_wasserstein_correlation
- gaussian_wasserstein_correlation_norm
- mutual_info_score
- adjusted_mutual_info_score
- norm_mutual_info_score
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