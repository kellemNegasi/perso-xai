## Notes on how the evaluators work
## Lime explainer
Lime explainer is implemented locally by taking a given instance, generate gausian noise based on std of the data, kernel width and noise scale. Basically give more weight for closer perturbations than to farther nones. `weights = np.exp(-(distances ** 2) / (kernel_width ** 2 + 1e-12)).`. kernel width controls how quickly the weights decay. Run predictions on the perturbed instances, train a ridge regressor on the dataset and use that regressor to explain the predictions. 

### Compactness
- Take the absolute values of importances and compute how many of them fall under the zero tolerance. 
```python
sparsity = 1.0 - (non_zero / n_features)
```
Then computes the normalized proportion of the top-k feature's contributions and also effective features coverage

```python
top5 = float(np.sum(normalized[: min(5, n_features)]))
top10 = float(np.sum(normalized[: min(10, n_features)]))

prob_dist = imp / total # each feature importance devided by total. (normalize it to sum=1)
participation = float(np.sum(prob_dist ** 2))
effective_count = 1.0 / participation if participation > 0.0 else float(n_features)
```
### Completness
- Take explanation that exceed a given threshold (support features)
- Mask n_features in these indexes
- Replace those features with base line and comupute prediction drop normalized to the original predictions
- Sample sampel trials of the same size and mask and compute the drop as well
- Completenss = max(0,normal_drop, random_drop)

