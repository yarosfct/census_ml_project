# Machine Learning Project – First Milestone (Application Route)

## 1. Problem Definition and Dataset Overview

We aim to predict whether an individual's annual income exceeds USD 50,000 using demographic and employment-related attributes. This is formulated as a binary classification task: given a feature vector describing a person (age, work class, education level, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week and country of origin), the goal is to output one of two labels: “≤50K” or “>50K”.

We will use the **Adult Census Income** dataset from the UCI Machine Learning Repository. The dataset contains 48,842 instances and 14 predictive features with a mix of numerical and categorical variables. Some features (e.g., workclass, occupation, native-country) include missing values indicated by a question mark. The target variable indicates whether the person’s income exceeds USD 50,000. The dataset is imbalanced, with the majority of individuals earning ≤50K.

*Key characteristics*
- **Instances:** 48,842.
- **Features:** 14 (8 categorical and 6 numerical).
- **Target:** Binary (>50K vs ≤50K).
- **Missing values:** Present in some categorical attributes.
- **Imbalance:** The >50K class is the minority.

This problem is relevant for socio‑economic analysis and has been widely used as a benchmark in machine learning research. Challenges include handling mixed data types, dealing with missing values and class imbalance, and addressing potential fairness concerns (the dataset includes sensitive attributes like race and sex).

## 2. Research Aim and Objectives

Our primary objective is to evaluate how different **preprocessing strategies** and **hyperparameter tuning** affect the performance of classical machine‑learning algorithms on the Adult dataset. Many previous studies compare various algorithms on this dataset but often fix preprocessing choices or hyperparameters. We want to systematically examine these design choices under a unified experimental protocol.

Our main goal is to apply and compare multiple classical machine-learning models on the Adult Income dataset. In addition to testing different prediction algorithms, we want to see how common preprocessing steps and basic hyperparameter tuning affect their performance.

**Research question:**

> *How much do preprocessing choices and hyperparameter tuning influence the predictive performance of classical machine‑learning models on the Adult income prediction task?*

To address this question, our project will:

1. **Baseline pipeline:** Build a standard preprocessing and modelling workflow with minimal tuning.
2. **Enhanced pipeline:** Incorporate additional preprocessing methods (handling imbalance via sampling or class weights, feature selection) and perform systematic hyperparameter tuning.
3. **Compare algorithms:** Evaluate several common classification models to understand which benefit most from tuning and preprocessing.
4. **Statistical analysis:** Report metrics across cross‑validation folds and apply statistical tests to compare models.

## 3. Literature Review

The Adult dataset has been extensively studied. We review existing work focusing on their data preprocessing, modelling methods, evaluation protocols and reported performance. Below, we group the literature into themes and summarise selected papers.

### 3.1 Classical baseline comparisons

**Chen (2021)** explored descriptive statistics of the Adult dataset, handled missing values, and compared logistic regression, discriminant analysis, support vector machines, random forests and boosting models. The study reported metrics such as ROC curves, accuracy, recall and F‑measure. The **boosting** model achieved the highest AUC and accuracy, identifying three variables as most influential relationship
, 
native.country 
and 
capital.gain .

**Chakrabarty & Biswas (2018)** treated income prediction as a classification problem and applied a **Gradient Boosting classifier** to the Adult dataset. They highlighted the economic motivation of reducing income inequality and reported that gradient boosting achieved about **88 % accuracy**, surpassing previous benchmarks. The authors reviewed earlier works using logistic regression, Naive Bayes, decision trees, extra trees, k‑NN and SVMs, and discussed using extra trees to compute feature importance for selection.

**Kumar (2024)** clustered individuals using **KMeans** to group similar records, then applied **Random Forest** and **XGBoost** models. Data preprocessing steps (handling missing values, encoding categorical variables) were emphasised as critical for model performance. Among all models tested, **Random Forest** delivered the highest accuracy.

**Islam et al. (2023)** used a modified version of the Adult dataset with 16 columns and compared **eleven models**, including logistic regression, Naive Bayes (standard and Gaussian), k‑NN, SVM, decision trees, random forests, XGBoost and artificial neural networks. They followed guidelines for handling categorical data and missing values and performed optimisation and evaluation. Their study, presented at the 2023 ICCCIS conference and later released online, noted that XGBoost and neural network ensembles achieved the highest accuracy (~87 %), stressing the importance of preprocessing and parameter tuning.

### 3.2 Advanced models and explainability

**Özkurt et al. (2025)** applied the **LightGBM** algorithm to predict income levels and addressed class imbalance using **SMOTE** oversampling. They reported an accuracy of about **87.5 %** with high precision, recall and F1‑score. The authors also used **SHAP** and **LIME** to interpret feature contributions, highlighting the value of explainable AI techniques alongside model performance.

**Mothilal et al. (2019)** introduced the concept of generating **diverse counterfactual explanations** to make classifier decisions more interpretable. Though not specific to the Adult dataset, their methodology emphasises the importance of explainability and fairness in machine‑learning models.

### 3.3 Other related works

**Srivastava et al. (2019)** developed **PUTWorkbench**, an open‑source tool for balancing privacy and utility in AI‑intensive systems. The authors evaluated the tool on two versions of the UCI Adult dataset. The first, called **adult‑iyengar**, is a subset containing only eight attributes and the class label, a setup that has become a de facto benchmark in privacy research. The second, **adult‑complete**, includes all 14 attributes; duplicate and conflicting rows with missing values were removed to yield a clean dataset of 45,175 instances. Experiments using J48 and Naïve Bayes classifiers varied the number of attributes fed to the models. Results showed that classification accuracy remained within reasonable limits when at least three attributes were used. The tool also offers one‑click options to handle missing values or duplicate instances and illustrates how privacy‑preserving attribute selection can maintain utility.

Several additional studies (Thapa 2023; Zhu 2016; Lemon et al.) apply multiple machine‑learning algorithms to the Adult dataset or related demographic datasets. Most use standard preprocessing (encoding categorical features, scaling numerical ones) and compare classifiers such as logistic regression, Naive Bayes, k‑NN, decision trees, random forests, SVMs and gradient boosting. Reported metrics include accuracy, precision, recall, F1‑score and ROC‑AUC. Many studies select models based on accuracy without conducting statistical tests or exploring the impact of preprocessing choices.


### 3.4 Summary of gaps in the literature

- **Limited analysis of preprocessing choices:** Few studies systematically compare different imputation strategies, encoding schemes or sampling methods.
- **Sparse use of statistical tests:** Many papers report raw metrics but rarely conduct statistical significance testing across models.
- **Hyperparameter tuning is often ad‑hoc:** Some works mention grid search but do not specify parameter grids or evaluation protocols.
- **Explainability is underexplored:** Except for recent work using SHAP or counterfactual explanations, interpretability is largely ignored.

Our project addresses these gaps by explicitly focusing on preprocessing methods and systematic hyperparameter tuning under a uniform experimental framework.

## 4. Proposed Methodology

### 4.1 Preprocessing Pipelines

We plan to implement two preprocessing pipelines:

1. **Baseline pipeline**
   - **Missing values:** Replace missing categorical entries (marked with `?`) with a dedicated “Missing” category. Impute missing numerical values using the median.
   - **Encoding:** Apply one‑hot encoding to all categorical features. Label‑encode the target variable (≤50K → 0, >50K → 1).
   - **Scaling:** Use **Min‑Max scaling** on numerical variables to normalise them to [0, 1].

2. **Enhanced pipeline**
   - Start with the baseline steps.
   - **Class imbalance handling:** Experiment with techniques such as **SMOTE** oversampling or incorporating **class weights** in the classifiers.
   - **Feature selection:** Use filter methods (e.g., mutual information or chi‑square scores) or model‑based methods (e.g., feature importances from Random Forest) to select the most informative features.
   - **Parameter tuning:** Perform systematic hyperparameter search (grid or random search) for selected models.

### 4.2 Prediction Algorithms

We will compare several classical machine‑learning algorithms:

- **Logistic Regression:** A linear model serving as a strong baseline for binary classification.
- **Naïve Bayes:** A simple probabilistic classifier to provide a generative baseline.
- **k‑Nearest Neighbours (k‑NN):** An instance‑based method sensitive to scaling; useful for illustrating the importance of preprocessing.
- **Support Vector Machine (SVM):** A margin‑based classifier that can capture nonlinear boundaries with appropriate kernels.
- **Random Forest:** An ensemble of decision trees that handles nonlinearities and feature interactions.
- **XGBoost (or LightGBM):** Gradient boosting algorithms known for strong performance on tabular data.

Due to time constraints, full hyperparameter tuning may focus on Logistic Regression, Random Forest, SVM and XGBoost/LightGBM. k‑NN and Naïve Bayes will serve mainly as baselines.

### 4.3 Experimental Protocol

- **Data splitting:** Use **stratified 5‑fold cross‑validation** to ensure each fold reflects the class distribution.
- **Model training:** Within each fold, fit the preprocessing pipeline using the training partition and transform both training and test partitions.
- **Hyperparameter tuning:** For tunable models, perform grid or random search on the training set (either using nested cross‑validation or a validation split) to select the best hyperparameters based on F1‑score.
- **Evaluation metrics:** Report **accuracy**, **precision**, **recall**, **F1‑score** and **ROC‑AUC** for each model. Given class imbalance, F1‑score and recall for the minority class (>50K) will be emphasised.
  - **Statistical tests:** To assess whether differences in model performance across cross‑validation folds are statistically significant, we will use **paired tests** on the fold‑wise scores. When normality assumptions hold we will employ paired **t‑tests**; otherwise we will use the non‑parametric **Wilcoxon signed‑rank test**. For comparing more than two models simultaneously, we may apply the **Friedman test** followed by a post‑hoc Nemenyi test.
- **Additional analyses:**
  - Compare the baseline and enhanced pipelines to quantify the impact of imbalance handling and feature selection.
  - For the best‑performing model(s), compute feature importances and, if time permits, use explainability techniques (e.g., SHAP) to interpret predictions.
  - Conduct sensitivity analysis on hyperparameters to understand their effect on performance.


## 7. References (to be completed)

1. Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository.
2. Chen, L.‑P. (2021). **Supervised learning for binary classification on US adult income.** *Journal of Modeling and Optimization*, 13(2), 80–91.
3. Chakrabarty, N., & Biswas, S. (2018). **A statistical approach to adult census income level prediction.** 2018 International Conference on Advances in Computing, Communication Control and Networking.
4. Kumar, V. (2024). **Income prediction using machine learning.** *Soft Computing Fusion with Applications*, 1(3), 150–159.
5. Islam, M. A., et al. (2023). **An investigation into the prediction of annual income levels through the utilization of demographic features employing the modified UCI Adult dataset.** 2023 International Conference on Computing, Communication and Intelligent Systems (ICCCIS). *(Paper presented in November 2023 and made available online in early 2024.)*
6. Özkurt, C., et al. (2025). **Income Level Estimation with Light‑GBM: Understanding Model Decisions with Explainable AI Techniques Shap and Lime.** *Artificial Intelligence in Applied Sciences*, 1(1), 7–12.
7. Mothilal, R., et al. (2019). **Explaining machine learning classifiers through diverse counterfactual explanations.** Proceedings of the Conference on Fairness, Accountability and Transparency.
8. Thapa, S. (2023). **Adult income prediction using various ML algorithms.** SSRN Electronic Journal. **TODO** – need details.
9. Zhu, H. (2016). **Predicting earning potential using the Adult dataset.** Retrieved December 5, 2016. **TODO** – need details.
10. Lemon, C., Zelazo, C., & Mulakaluri, K. (Year?). **Predicting if income exceeds $50,000 per year based on 1994 US Census Data with simple classification techniques.** **TODO** – need details.

11. Srivastava, S., Namboodiri, V. P., & Prabhakar, T. V. (2019). **PUTWorkbench: Analysing Privacy in AI‑intensive Systems.** *arXiv preprint* arXiv:1902.01580.

