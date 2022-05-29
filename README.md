# 2021-Mathor-Cup-A

 The answer of  Question A in "2021 Mathor Cup" Competition
 
 ![](https://shields.io/badge/language-Chinese%20simplified-orange)
 ![](https://shields.io/badge/dependencies-Python%203.9-blue)

## Question

The language of this competition is Chinese simplified.

[2021年MathorCup大数据挑战赛-赛道A.pdf](https://github.com/cloudy-sfu/2021-Mathor-Cup-A/files/7881846/2021.MathorCup.-.A.pdf)

## Warning

**There are some scientific errors in this program:**

We should not use correlation matrix to filter variables; instead, we should use vairance inflection factors (VIF).

The "ANOVA" part is not ANOVA in fect; we should ignore it before using this program.

## Usage

The scripts use features of Jetbrains PyCharm scientific mode. You should run a script section-by-section, where sections are splitted by `# %%` symbol.

For problem 1:

1. `Q1_description_report.py` generates description statistical reports for raw and pre-processed data.
2. `Q1_pre_processing.py` performs data pre-processing, where `pre_processor.py` provides transformers in some of steps.
3. `Q1_correlation.py` performs further pre-processing steps, based on correlation between variables.
4. `Q1_adaboost.py, Q1_random_forest.py, Q1_xgboost.py` are regression models.
5. `Q1_stacking.py` fits a stacking model on k-fold dataset, based on models configured in step (4).
6. `Q1_refit_stacking.py` fits a stacking model on all samples.
7. `Q1_app_stacking.py` applies the model trained in step (6), and predicts the dependent variables on test set.

For problem 2:

1. `Q2_pre_processing.py` performs data pre-processing, where `pre_processor.py, pre_processor_extended.py` both provide transformers in some of steps.
2. `Q2_random_forest.py` builds a random forest model on k-fold dataset, and outputs feature importances.
3. `Q2_anova.py` performs a statistics test, researching whether heteroscedasticity exists.

For problem 3:

I research on the periodicity of traded amount over time.

1. `Q3_pre_processing.py` performs pre-processing for time series data.
2. `Q3_arima_get_orders.py` calculates orders for ARIMA model.
3. `Q3_arima.py` builds an ARIMA model and outputs statistical tests on residual and periodicity.
