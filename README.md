## Dependencies

For a straight-forward use of INAFEN, you can install the required libraries from *requirements.txt*: ` pip install -r requirements_attdmm.txt`

## Dataset

We provide the raw data used in this paper in *./Data/Raw_data*. If you want to use your own dataset, you can process it into the required format. 

## Example Usage

1. To run the baseline models in a specific dataset: `python main_baselines.py --dataset_name some_dataset --baseline_model some_baseline_model`

   The *dataset_name* can be chosen from `[adult, churn, credit-card-econometrics, credit-Japan, dataset_29_credit-a, dataset_31_credit-g, dataset_37_diabetes, dataset_heart-failure, gmsc, South-German-Credit-Prediction]` and *some_baseline_model* can be chosen from `[XGB, RF, LR, DT, KNN, SVM]`. 

   For example, you can run the experiment with *logistic regression* on the *adult* dataset with `python main_baselines.py --dataset_name adult --baseline_model LR`.

   The generated results can be found in *./Results/dataset_name/some_baseline_model*. The results are in *.txt* format and named by the AUROC score on the validation dataset and the corresponding hyperparameters of baseline models.

2. Then, you can run the INAFEN with a specific parameter setting with `python main_INAFEN.py --dataset_name dataset_31_credit-g.csv --DTFT True --ARFC True --BMKD True --FC_min_support 0.2 --FC_min_confidence 0.8`. The teacher model can be chosen by modifying the code `teacher_model = XGBClassifier(n_estimators=100, max_depth=3, gamma=1)` in *main_INAFEN.py* with the best teacher model according some metric on the validation dataset. 

   The generated results can be found in  *./Results/dataset_name/some_setting_of_INAFEN*, in which *some_setting_of_INAFEN* can be *DTFT[True]-ARFC[True]-BMKD[True]*, for example.
