# Evaluating the Impact of Undersampling Techniques on Imbalanced Healthcare Data: A Case Study on Stroke Prediction
## Introduction
In healthcare, predictive models can significantly improve patient outcomes by enabling early intervention and personalized treatment plans. However, many healthcare datasets, such as those predicting stroke occurrences, suffer from class imbalance. This study evaluates the effectiveness of various undersampling techniques on an imbalanced healthcare dataset related to stroke prediction.

## Dataset
The dataset used in this project contains information necessary to predict the occurrence of a stroke. Each row in the dataset represents a patient, and the dataset includes the following attributes:

- **id:** Unique identifier
- **gender:** "Male", "Female", or "Other"
- **age:** Age of the patient
- **hypertension:** 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
- **heart_disease:** 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
- **ever_married:** "No" or "Yes"
- **work_type:** "Children", "Govt_job", "Never_worked", "Private", or "Self-employed"
- **Residence_type:** "Rural" or "Urban"
- **avg_glucose_level:** Average glucose level in the blood
- **bmi:** Body mass index
- **smoking_status:** "Formerly smoked", "Never smoked", "Smokes", or "Unknown"
- **stroke:** 1 if the patient had a stroke, 0 if not

## Problem Definition
Many healthcare datasets are imbalanced, leading to biased models that perform poorly in identifying the minority class. This study aims to:

1. **Assess Model Performance on Imbalanced Data:** Understand model performance when trained on imbalanced data without adjustments.
2. **Implement Oversampling Techniques:** Apply various undersampling methods to balance the dataset.
3. **Comparative Analysis:** Compare model performance on original vs. undersampled datasets using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.
4. **Insights and Recommendations:** Identify which undersampling techniques yield the most significant improvements and provide recommendations for handling class imbalance in similar datasets.

## Methods
1. **Data Preprocessing:** Cleaning and preparing the data for analysis.
2. **Baseline Model:** Training a baseline model on the original imbalanced dataset.
3. **Undersampling Techniques:** Implementing the following undersampling techniques:
   - **Cluster Centroids:** Reduces the dataset by synthesizing centroids from clusters using K-means, rather than selecting individual samples.
   - **Random Undersampling:** Randomly selects a subset of data from the targeted classes.
   - **NearMiss:** Uses three different rules based on nearest neighbors to select samples:
     - **NearMiss-1:** Chooses samples with the smallest average distance to the closest samples of the minority class.
     - **NearMiss-2:** Chooses samples with the smallest average distance to the farthest samples of the minority class.
     - **NearMiss-3:** Keeps samples based on a two-step nearest-neighbor selection process.
   - **Tomek Links:** Removes pairs of samples from different classes that are each other's closest neighbors.
   - **Edited Nearest Neighbors (ENN):** Removes majority samples if any or most neighbors belong to a different class.
   - **Repeated Edited Nearest Neighbors (RENN):** Extends ENN by repeating the process multiple times.
   - **All KNN:** Increases the neighborhood size in each iteration of ENN to progressively clean the dataset.
   - **Condensed Nearest Neighbor (CNN):**  Adds samples that are difficult to classify based on a 1-nearest neighbor rule.
   - **One-Sided Selection (OSS):** Combines CNN and Tomek Links to remove noisy or difficult-to-classify majority samples.
   - **Neighborhood Cleaning Rule (NCR):** Uses a combination of ENN and nearest-neighbor classification to remove noisy samples.
   - **Instance Hardness Threshold:**  Retains samples that are easier to classify by removing instances classified with low probability, based on a specified classifier.
   
     **Source:** https://imbalanced-learn.org/stable/under_sampling.html

4. **Model Training:** Training models on both the original and undersampled datasets.
5. **Evaluation:** Comparing model performance using accuracy, precision, recall, F1-score, and AUC-ROC.

## Models
The following machine learning models were used in this study:
1. **Logistic Regression:** A linear model for binary classification that estimates probabilities using the logistic function. It's commonly used for classification tasks due to its simplicity and interpretability.
2. **Random Forest:** An ensemble learning method that creates multiple decision trees using random subsets of data and features, then averages their predictions to improve accuracy and control overfitting.
3. **Support Vector Machine (SVM):** A classifier that finds the optimal hyperplane to separate classes, maximizing the margin between the nearest data points of each class (support vectors). It can also provide probability estimates for classification tasks.
4. **Gradient Boosting:** An ensemble technique that builds models sequentially, with each model correcting the errors of the previous one. It often yields high accuracy, especially in structured data tasks.
5. **AdaBoost:** Another ensemble method, AdaBoost combines multiple weak classifiers to form a strong classifier, giving more weight to instances that were incorrectly classified in previous iterations. This approach is particularly useful for focusing on harder-to-classify instances.
6. **XGBoost:** An optimized gradient boosting framework that is highly efficient and scalable, widely used in competitions and industry applications. It includes regularization to prevent overfitting, making it suitable for both classification and regression.
7. **LightGBM:** A gradient boosting framework developed by Microsoft, optimized for speed and efficiency. It uses a novel tree-building strategy called "leaf-wise" growth and is known for handling large datasets well.
8. **CatBoost:** A gradient boosting library developed by Yandex, designed to handle categorical features efficiently without extensive preprocessing. It performs well on structured data and is robust against overfitting.
9. **k-Nearest Neighbors (k-NN):** A non-parametric, instance-based learning algorithm that classifies instances based on the majority class of their k nearest neighbors. It’s simple and effective for smaller datasets but can be computationally expensive with larger data.
10. **Decision Tree:** A non-parametric model that splits data based on feature values to form a tree-like structure. It’s interpretable and useful for both classification and regression tasks, though prone to overfitting.
11. **Naive Bayes:** A probabilistic classifier based on Bayes' theorem, with the assumption that features are independent. It’s efficient and performs well on high-dimensional data, especially with text classification.
12. **Linear Discriminant Analysis (LDA):** A linear classifier that projects data onto a lower-dimensional space, maximizing class separability. LDA is suitable for classification when the classes are linearly separable.
13. **Quadratic Discriminant Analysis (QDA):** An extension of LDA that models quadratic decision boundaries, allowing it to capture non-linear relationships between features and class labels.
14. **Extra Trees (Extremely Randomized Trees):** An ensemble method similar to Random Forest, but with more randomization in the tree-building process. It often improves generalization and is faster than Random Forest due to the reduced complexity in splitting nodes.

**Sources (APA Style):**
- Logistic Regression
  - Cox, D. R. (1958). The regression analysis of binary sequences. *Journal of the Royal Statistical Society: Series B (Methodological)*, 20(2), 215–242.
  - Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- Random Forest
  - Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324
- Support Vector Machine (SVM)
  - Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning, 20*(3), 273–297. https://doi.org/10.1007/BF00994018
- Gradient Boosting
  - Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics, 29*(5), 1189–1232. https://doi.org/10.1214/aos/1013203451
- AdaBoost
  - Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. *Journal of Computer and System Sciences, 55*(1), 119–139. https://doi.org/10.1006/jcss.1997.1504
- XGBoost
  - Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). https://doi.org/10.1145/2939672.2939785
- LightGBM
  - Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems* (Vol. 30). https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree
- CatBoost
  - Prokhorenkova, L. O., Gusev, G. V., Vorobev, A. A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. In *Advances in Neural Information Processing Systems* (Vol. 31). https://arxiv.org/abs/1706.09516
- k-Nearest Neighbors (k-NN)
  - Cover, T. M., & Hart, P. E. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory, 13*(1), 21–27. https://doi.org/10.1109/TIT.1967.1053964
- Decision Tree
  - Quinlan, J. R. (1986). Induction of decision trees. *Machine Learning, 1*, 81–106. https://doi.org/10.1007/BF00116251
- Naive Bayes
  - McCallum, A., & Nigam, K. (1998). A comparison of event models for Naive Bayes text classification. In *AAAI-98 Workshop on Learning for Text Categorization* (pp. 41–48).
- Linear Discriminant Analysis (LDA)
  - Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics, 7*(2), 179–188. https://doi.org/10.1111/j.1469-1809.1936.tb02137.x
- Quadratic Discriminant Analysis (QDA)
  - Loog, M., Duin, R. P. W., & Haeb-Umbach, R. (2001). Multiclass linear and quadratic discriminant analysis revisited. In *Pattern Recognition* (pp. 999–1002). Springer, Berlin, Heidelberg.
- Extra Trees (Extremely Randomized Trees)
  - Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. *Machine Learning, 63*(1), 3–42. https://doi.org/10.1007/s10994-006-6226-1

## Evaluation Metrics
The following evaluation metrics were used in this study:
- **Accuracy:** The ratio of correctly predicted instances to the total instances. Measures the overall correctness of the model. However, it can be misleading in imbalanced datasets, as it might reflect high values even if the model fails to predict the minority class correctly.
- **Precision:** The ratio of correctly predicted positive instances to the total predicted positives. Indicates the accuracy of the positive predictions made by the model. High precision means that there are fewer false positives.
- **Recall (Sensitivity or True Positive Rate):** The ratio of correctly predicted positive instances to all actual positives. Measures the model's ability to identify all relevant instances. High recall means that there are fewer false negatives.
- **F1-Score:** The harmonic mean of precision and recall. Provides a single metric that balances both precision and recall. It is particularly useful when the class distribution is imbalanced.
- **ROC AUC Score:** The area under the Receiver Operating Characteristic (ROC) curve.
   - The ROC curve plots the true positive rate (recall) against the false positive rate (1 - specificity). The AUC score indicates how well the model distinguishes between the classes. A score of 1 indicates perfect discrimination, while a score of 0.5 indicates no discrimination (random guessing).

**Sources (APA Style):**
- Accuracy
  - Powers, D. M. W. (2011). Evaluation: From precision, recall, and F-measure to ROC, informedness, markedness, and correlation. *Journal of Machine Learning Technologies, 2*(1), 37–63. https://www.researchgate.net/publication/281917226
- Precision, Recall, and F1-Score
  - Powers, D. M. W. (2011). Evaluation: From precision, recall, and F-measure to ROC, informedness, markedness, and correlation. *Journal of Machine Learning Technologies, 2*(1), 37–63. https://www.researchgate.net/publication/281917226
- ROC AUC Score
  - Hanley, J. A., & McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology, 143*(1), 29–36. https://doi.org/10.1148/radiology.143.1.7063747
  - Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010

## Results
The results section will detail the performance of models on both the imbalanced and oversampled datasets.
### Performance on Imbalanced Data
The table below summarizes the performance of various machine learning models when trained on the imbalanced stroke dataset:

| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-score (0) | F1-score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|----------------|
| Logistic Regression          | 0.94    | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.851          |
| Random Forest                | 0.94    | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.797          |
| Support Vector Machine       | 0.94    | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.628          |
| Gradient Boosting            | 0.94    | 0.94          | 0.33          | 1.00       | 0.02       | 0.97         | 0.03         | 0.835          |
| XGBoost                      | 0.94     | 0.94          | 0.50          | 0.99       | 0.10       | 0.97         | 0.16         |  0.796         |
| AdaBoost                     | 0.94     | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.793          |
| LightGBM                     | 0.94     | 0.94          | 0.33          | 0.99       | 0.06       | 0.97         | 0.11         | 0.817          |
| CatBoost                     | 0.94     |0.94           |0.67           |1.00        |0.03        |0.97          |0.06          | 0.819          |
| K-Nearest Neighbors          | 0.94    | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.647          |
| Decision Tree                | 0.92    | 0.95          | 0.27          | 0.97       | 0.19       | 0.96         | 0.23         | 0.580          |
| Naive Bayes                  | 0.87    | 0.96          | 0.22          | 0.89       | 0.47       | 0.93         | 0.30         | 0.829          |
| Linear Discriminant Analysis | 0.93    | 0.94          | 0.27          | 0.99       | 0.05       | 0.97         | 0.08         | 0.842          |
| Quadratic Discriminant Analysis | 0.88 | 0.96          | 0.24          | 0.91       | 0.45       | 0.93         | 0.31         | 0.830          |
|  Extra Trees                 | 0.94     | 0.94          | 0.29          | 0.99       | 0.03       | 0.97         | 0.06         | 0.776          |

**Observations of Performance on Imbalanced Data:**

Most models achieve high accuracy (around 93-94%). However, accuracy alone is misleading for imbalanced data, as it can be high even when the model fails to predict the minority class (stroke cases) effectively.

Precision (0) and Recall (0) are consistently high across all models, close to or at 1.00. This indicates that the models are highly accurate in identifying non-stroke cases and have a very low rate of false positives for the majority class.Precision (1) and Recall (1) vary significantly across models, generally indicating poor performance in identifying stroke cases.

Many models, such as Logistic Regression, Random Forest, SVM, and AdaBoost, have Precision (1) and Recall (1) of 0.00. This means these models are entirely missing the minority class, predicting no stroke cases.Some models like XGBoost, LightGBM, CatBoost, Decision Tree, and Naive Bayes show slightly better performance on stroke cases but still have low recall, indicating high false negatives.

F1-scores for class 1 are very low across all models, indicating an overall weak performance in capturing stroke cases. This low F1-score shows that even when some models have non-zero precision and recall for stroke, their balance between precision and recall is still poor. 

The ROC AUC score varies significantly, with Logistic Regression achieving the highest score (0.851), indicating relatively better discrimination between the classes than other models.Models like Support Vector Machine, Decision Tree, K-Nearest Neighbors, and Extra Trees have notably lower AUC scores, suggesting limited ability to distinguish between stroke and non-stroke cases.

**Summary of Performance on Imbalanced Data:**
- High Bias towards the Majority Class: Most models have high accuracy and perfect recall for non-stroke cases but perform poorly for stroke prediction, highlighting the challenge of imbalanced data.
- Poor Recall for Stroke (Class 1): Low recall for the minority class (stroke) in most models indicates a high number of false negatives, meaning many true stroke cases are not identified by the models.
- Best Models for Minority Class (Class 1): Among all models, CatBoost and XGBoost have the highest Precision (1) values (0.67 and 0.50, respectively), suggesting they are the least likely to falsely predict stroke cases. Naive Bayes and Quadratic Discriminant Analysis also show better performance with slightly higher recall and F1-scores for class 1.
- Recommendation: For a more effective stroke prediction model, focus on models like Logistic Regression and CatBoost, which show relatively higher AUC scores. However, further adjustments, such as using undersampling class balancing techniques, are recommended to improve recall for stroke cases.

### Performce on undersampled datasets
Below are the results of each undersampling technique:
#### ClusterCentroids
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-score (0) | F1-score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          | 0.744    | 0.98          | 0.16          | 0.74       | 0.73       | 0.85         | 0.26         | 0.831         |
| Random Forest                | 0.440    | 0.99          | 0.09          | 0.41       | 0.95       | 0.58         | 0.17         | 0.792         |
| Support Vector Machine       | 0.733    | 0.98          | 0.15          | 0.73       | 0.73       | 0.84         | 0.25         | 0.815         |
| Gradient Boosting            | 0.321    | 1.00          | 0.08          | 0.28       | 1.00       | 0.43         | 0.15         | 0.809         |
| AdaBoost                     | 0.355    | 1.00          | 0.09          | 0.31       | 1.00       | 0.48         | 0.16         | 0.815         |
| XGBoost                      | 0.424    | 1.00          | 0.09          | 0.39       | 0.98       | 0.56         | 0.17         | 0.806         |
| LightGBM                     | 0.343    | 1.00          | 0.08          | 0.30       | 1.00       | 0.46         | 0.16         | 0.809         |
| CatBoost                     | 0.403    | 1.00          | 0.09          | 0.36       | 1.00       | 0.53         | 0.17         | 0.814         |
| k-Nearest Neighbors          | 0.746    | 0.97          | 0.15          | 0.75       | 0.68       | 0.85         | 0.24         | 0.745         |
| Decision Tree                | 0.392    | 0.99          | 0.08          | 0.36       | 0.92       | 0.53         | 0.16         | 0.639         |
| Naive Bayes                  | 0.682    | 0.98          | 0.14          | 0.67       | 0.84       | 0.80         | 0.24         | 0.836         |
| Linear Discriminant Analysis | 0.732    | 0.98          | 0.15          | 0.73       | 0.76       | 0.84         | 0.26         | 0.830         |
| Quadratic Discriminant Analysis | 0.684 | 0.99          | 0.14          | 0.67       | 0.85       | 0.80         | 0.25         | 0.830         |
| Extra Trees                  | 0.545    | 0.99          | 0.11          | 0.52       | 0.90       | 0.68         | 0.19         | 0.796         |

**Observations of Performance on each model after ClusterCentroids:**

The accuracy scores are highly variable, ranging from 32% (Gradient Boosting) to 75% (k-Nearest Neighbors).While Logistic Regression, Support Vector Machine, k-Nearest Neighbors, and Naive Bayes show relatively higher accuracy (~70% or more), accuracy alone can be misleading in imbalanced data, as it might not reflect the model’s ability to identify the minority class (stroke cases).

Most models show very high precision for the non-stroke class, close to 1.00. This means that when the model predicts a case as non-stroke, it is usually correct.
Models like Gradient Boosting, AdaBoost, XGBoost, LightGBM, and CatBoost achieve a perfect 1.00 precision for Class 0, indicating a strong ability to correctly identify non-stroke cases with minimal false positives.Precision for the minority class (stroke) is generally low across models, with only Logistic Regression (0.16), Naive Bayes (0.14), and Quadratic Discriminant Analysis (0.14) showing slightly higher values.The low precision in most models (often around 0.08 to 0.15) indicates that when these models predict a stroke, many of those predictions are false positives, reducing the model’s reliability for stroke prediction.

Recall values for Class 0 are consistently high, especially for Logistic Regression, SVM, and Naive Bayes, indicating that these models are very effective in capturing non-stroke cases.The recall for non-stroke cases reaches 0.98 to 1.00 in most models, showing that non-stroke cases are almost always identified correctly. Logistic Regression, Naive Bayes, and Quadratic Discriminant Analysis achieve higher recall values for stroke cases (around 0.73 to 0.85), indicating that these models are better at identifying actual stroke cases compared to other models.However, models like Gradient Boosting, AdaBoost, and LightGBM reach 1.00 recall for stroke cases, but this is likely due to predicting nearly all cases as strokes, which explains their low accuracy.

The F1-scores for the non-stroke class are consistently high, with most models achieving around 0.80 to 0.85, indicating balanced precision and recall for this class. The F1-score for the stroke class is low across all models, with Logistic Regression, SVM, Naive Bayes, and Quadratic Discriminant Analysis showing slightly higher F1-scores (~0.24 to 0.26).These low F1-scores indicate that despite some improvements in recall, the models still struggle to achieve a good balance between precision and recall for predicting strokes.

ROC AUC scores range between 0.638 (Decision Tree) and 0.836 (Naive Bayes). This metric indicates how well the model distinguishes between classes, with higher scores showing better class separation. Logistic Regression, Naive Bayes, and Quadratic Discriminant Analysis have relatively higher ROC AUC scores (above 0.82), suggesting they are slightly better at distinguishing stroke cases from non-stroke cases.

**Summary of Performance  on each model after ClusterCentroids:**

- Logistic Regression, Naive Bayes, and Quadratic Discriminant Analysis show relatively better performance after applying ClusterCentroids, with higher recall for the stroke class and balanced performance across metrics. Gradient Boosting, AdaBoost, and LightGBM achieve high recall for stroke cases but at the expense of accuracy and precision, suggesting they may over-predict strokes.
- The ClusterCentroids undersampling technique helps achieve balanced recall across classes but often reduces precision for the minority class (stroke).
Overall, Naive Bayes and Logistic Regression show the best balance between capturing stroke cases and maintaining a reasonable ROC AUC, making them potentially more suitable for this imbalanced dataset.

#### Random Undersampling:
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-score (0) | F1-score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          | 0.728    | 0.98          | 0.15          | 0.72       | 0.77       | 0.83         | 0.26         | 0.846         |
| Random Forest                | 0.713    | 0.98          | 0.15          | 0.71       | 0.79       | 0.82         | 0.25         | 0.824         |
| Support Vector Machine       | 0.696    | 0.98          | 0.14          | 0.69       | 0.81       | 0.81         | 0.24         | 0.824         |
| Gradient Boosting            | 0.703    | 0.97          | 0.13          | 0.70       | 0.69       | 0.82         | 0.22         | 0.813         |
| AdaBoost                     | 0.713    | 0.98          | 0.14          | 0.71       | 0.73       | 0.82         | 0.23         | 0.764         |
| XGBoost                      | 0.710    | 0.98          | 0.15          | 0.71       | 0.79       | 0.82         | 0.25         | 0.816         |
| LightGBM                     | 0.704    | 0.98          | 0.14          | 0.70       | 0.74       | 0.82         | 0.23         | 0.821         |
| CatBoost                     | 0.725    | 0.98          | 0.15          | 0.72       | 0.79       | 0.83         | 0.26         | 0.842         |
| k-Nearest Neighbors          | 0.678    | 0.98          | 0.14          | 0.67       | 0.81       | 0.80         | 0.23         | 0.778         |
| Decision Tree                | 0.680    | 0.98          | 0.13          | 0.68       | 0.74       | 0.80         | 0.22         | 0.709         |
| Naive Bayes                  | 0.725    | 0.98          | 0.15          | 0.72       | 0.77       | 0.83         | 0.25         | 0.832         |
| Linear Discriminant Analysis | 0.725    | 0.98          | 0.15          | 0.72       | 0.79       | 0.83         | 0.26         | 0.848         |
| Quadratic Discriminant Analysis | 0.729 | 0.98          | 0.15          | 0.73       | 0.77       | 0.83         | 0.26         | 0.833         |
| Extra Trees                  | 0.697    | 0.98          | 0.14          | 0.69       | 0.81       | 0.81         | 0.24         | 0.804         |

**Observations of Performance on each model after Random Undersampling:**

Accuracy values range from 67.8% (k-Nearest Neighbors) to 72.9% (Quadratic Discriminant Analysis), with most models achieving around 70-73%. However, since accuracy is influenced by the dominant class, it’s not the most reliable indicator for imbalanced datasets.

Precision for the non-stroke class (0) is consistently high across models, close to 0.98. This indicates that when these models predict a non-stroke case, they are usually correct, reflecting the strong performance of models on the majority class. Precision for the stroke class (1) is relatively low across all models, around 0.13-0.15. This low precision indicates a high rate of false positives when predicting strokes, meaning models often incorrectly classify non-stroke cases as strokes.

Recall for non-stroke cases is fairly high across all models, ranging from 0.67 to 0.73. This suggests that most non-stroke cases are correctly identified, though a few false negatives still occur. Recall for the stroke class (1) is higher than precision, with most models achieving around 0.73 to 0.81. This higher recall suggests that the models are identifying a majority of true stroke cases but with some false positives, leading to lower precision.
Quadratic Discriminant Analysis, Logistic Regression, and CatBoost achieved the highest recall scores (~0.77-0.81), making them slightly better for stroke detection compared to other models.

The F1-scores for non-stroke cases are consistently high (around 0.80 to 0.83), indicating a good balance between precision and recall for this class. The F1-scores for stroke cases are low across all models (around 0.22 to 0.26). This low F1-score for the minority class suggests that, despite improvements in recall, models still struggle to achieve high precision and recall simultaneously for stroke cases.

The ROC AUC scores range from 0.709 (Decision Tree) to 0.848 (Linear Discriminant Analysis), indicating variability in the models' ability to distinguish between stroke and non-stroke cases. Logistic Regression, CatBoost, Linear Discriminant Analysis, and Quadratic Discriminant Analysis exhibit higher ROC AUC scores (around 0.83 to 0.85), making them the most effective at differentiating between stroke and non-stroke cases after RandomUnderSampler.

**Summary of Performance  on each model after Random Undersampling:**
- Logistic Regression, Linear Discriminant Analysis, Quadratic Discriminant Analysis, and CatBoost show relatively balanced performance after RandomUnderSampler, achieving higher accuracy, recall, and ROC AUC scores for stroke cases than other models.
- Low Precision for Stroke Prediction: While recall for stroke cases has improved, precision remains low, suggesting that models are prone to false positives. This trade-off indicates that the models are tuned towards identifying stroke cases (high recall) but struggle with false alarms.
- Overall Effectiveness of RandomUnderSampler: RandomUnderSampler has helped models achieve better recall and AUC scores for stroke cases, yet further improvements are needed to increase precision. This method can serve as a preliminary technique for balancing data, but additional fine-tuning, possibly combining it with other sampling techniques or model adjustments, would enhance stroke detection.

#### NearMiss-1:
#### NearMiss-2:
#### NearMiss-3:
#### Tomek Links:
#### Edited Nearest Neighbors (ENN):
#### Repeated Edited Nearest Neighbors (RENN):
#### All KNN:
#### Condensed Nearest Neighbor (CNN):
#### One-Sided Selection (OSS):
#### Neighborhood Cleaning Rule (NCR):
#### Instance Hardness Threshold:

## Conclusion

## Discussion

## Future Work
