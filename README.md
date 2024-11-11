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

## Evaluation Metrics
The following evaluation metrics were used in this study:
- **Accuracy:** The ratio of correctly predicted instances to the total instances. Measures the overall correctness of the model. However, it can be misleading in imbalanced datasets, as it might reflect high values even if the model fails to predict the minority class correctly.
- **Precision:** The ratio of correctly predicted positive instances to the total predicted positives. Indicates the accuracy of the positive predictions made by the model. High precision means that there are fewer false positives.
- **Recall (Sensitivity or True Positive Rate):** The ratio of correctly predicted positive instances to all actual positives. Measures the model's ability to identify all relevant instances. High recall means that there are fewer false negatives.
- **F1-Score:** The harmonic mean of precision and recall. Provides a single metric that balances both precision and recall. It is particularly useful when the class distribution is imbalanced.
- **ROC AUC Score:** The area under the Receiver Operating Characteristic (ROC) curve.
   - The ROC curve plots the true positive rate (recall) against the false positive rate (1 - specificity). The AUC score indicates how well the model distinguishes between the classes. A score of 1 indicates perfect discrimination, while a score of 0.5 indicates no discrimination (random guessing).

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

**Summary of Performance on each model after Random Undersampling:**
- Logistic Regression, Linear Discriminant Analysis, Quadratic Discriminant Analysis, and CatBoost show relatively balanced performance after RandomUnderSampler, achieving higher accuracy, recall, and ROC AUC scores for stroke cases than other models.
- Low Precision for Stroke Prediction: While recall for stroke cases has improved, precision remains low, suggesting that models are prone to false positives. This trade-off indicates that the models are tuned towards identifying stroke cases (high recall) but struggle with false alarms.
- Overall Effectiveness of RandomUnderSampler: RandomUnderSampler has helped models achieve better recall and AUC scores for stroke cases, yet further improvements are needed to increase precision. This method can serve as a preliminary technique for balancing data, but additional fine-tuning, possibly combining it with other sampling techniques or model adjustments, would enhance stroke detection.

#### NearMiss-1:
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-score (0) | F1-score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          | 0.469    | 0.95          | 0.07          | 0.46       | 0.63       | 0.62         | 0.13         | 0.631         |
| Random Forest                | 0.252    | 0.93          | 0.06          | 0.22       | 0.74       | 0.36         | 0.11         | 0.529         |
| Support Vector Machine       | 0.359    | 0.94          | 0.06          | 0.34       | 0.66       | 0.50         | 0.11         | 0.522         |
| Gradient Boosting            | 0.285    | 0.95          | 0.06          | 0.25       | 0.77       | 0.40         | 0.12         | 0.458         |
| AdaBoost                     | 0.283    | 0.93          | 0.06          | 0.26       | 0.71       | 0.40         | 0.11         | 0.395         |
| XGBoost                      | 0.310    | 0.95          | 0.06          | 0.28       | 0.76       | 0.43         | 0.12         | 0.566         |
| LightGBM                     | 0.355    | 0.95          | 0.07          | 0.33       | 0.73       | 0.49         | 0.12         | 0.569         |
| CatBoost                     | 0.287    | 0.94          | 0.06          | 0.26       | 0.74       | 0.40         | 0.11         | 0.555         |
| k-Nearest Neighbors          | 0.480    | 0.94          | 0.06          | 0.47       | 0.56       | 0.63         | 0.12         | 0.581         |
| Decision Tree                | 0.196    | 0.91          | 0.06          | 0.16       | 0.76       | 0.27         | 0.10         | 0.459         |
| Naive Bayes                  | 0.504    | 0.94          | 0.06          | 0.51       | 0.48       | 0.66         | 0.11         | 0.510         |
| Linear Discriminant Analysis | 0.622    | 0.96          | 0.10          | 0.62       | 0.61       | 0.76         | 0.16         | 0.668         |
| Quadratic Discriminant Analysis | 0.517 | 0.94          | 0.06          | 0.52       | 0.48       | 0.67         | 0.11         | 0.500         |
| Extra Trees                  | 0.281    | 0.94          | 0.06          | 0.25       | 0.74       | 0.40         | 0.11         | 0.517         |

**Observations of Performance on each model after NearMiss-1:**

The accuracy scores are relatively low, ranging from 19.6% (Decision Tree) to 62.2% (Linear Discriminant Analysis). This low accuracy reflects the effect of undersampling the majority class, making it more challenging for models to achieve high overall correctness when both classes are roughly balanced.

Precision for the non-stroke class (0) is relatively high across all models, usually above 0.91. This indicates that the models generally make fewer false positives when predicting non-stroke cases. The highest precision for Class 0 (around 0.96) is achieved by Linear Discriminant Analysis, showing that this model predicts non-stroke cases with fewer errors compared to other models.

Precision for the stroke class (1) is very low across all models, typically around 0.06-0.07 and reaching a maximum of 0.10 in Linear Discriminant Analysis. This low precision indicates a high number of false positives for the stroke class, meaning that many cases predicted as stroke are actually non-stroke cases.

Recall for Class 0 is generally low, with most models achieving between 0.16 (Decision Tree) and 0.62 (Linear Discriminant Analysis). Linear Discriminant Analysis achieves the highest recall for non-stroke cases, suggesting it can capture a larger portion of true non-stroke cases compared to other models.

Recall for the stroke class is higher than precision, with scores between 0.48 and 0.77. This reflects the models’ ability to capture true stroke cases, albeit with a high number of false positives. Gradient Boosting, CatBoost, and Decision Tree achieve some of the highest recall values (~0.74-0.77), indicating these models are better at identifying true stroke cases, even though their accuracy and precision are low.

The F1-scores for Class 0 are relatively low, ranging from 0.27 (Decision Tree) to 0.76 (Linear Discriminant Analysis). This suggests that while these models maintain high precision, their recall for non-stroke cases is not as strong. The F1-scores for stroke prediction are very low across all models (around 0.10-0.16). This low F1-score indicates that despite relatively higher recall, the models struggle with low precision, making it difficult to achieve balanced performance in stroke prediction.

ROC AUC scores range from 0.394 (AdaBoost) to 0.668 (Linear Discriminant Analysis), indicating the variability in the models’ ability to distinguish between stroke and non-stroke cases. Linear Discriminant Analysis has the highest ROC AUC score, suggesting it is better at differentiating between stroke and non-stroke cases than other models.

**Summary of Performance on each model after NearMiss-1:**
- Linear Discriminant Analysis achieves the best overall performance after applying NearMiss-1, with the highest accuracy, recall, and ROC AUC score for both classes, indicating it is better at handling balanced class distributions.
- High Recall but Low Precision for Stroke (Class 1): Most models exhibit higher recall than precision for stroke cases, meaning they can identify a majority of true stroke cases but at the expense of high false positives.
- Effectiveness of NearMiss-1: NearMiss-1 improves the models’ ability to capture stroke cases (higher recall) but compromises precision, leading to lower overall accuracy and high false positives for the minority class.
  
#### NearMiss-2:
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-score (0) | F1-score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          | 0.520    | 0.96          | 0.08          | 0.51       | 0.65       | 0.67         | 0.14         | 0.595         |
| Random Forest                | 0.210    | 0.94          | 0.06          | 0.17       | 0.84       | 0.29         | 0.11         | 0.468         |
| Support Vector Machine       | 0.219    | 0.95          | 0.06          | 0.18       | 0.85       | 0.30         | 0.12         | 0.460         |
| Gradient Boosting            | 0.228    | 0.95          | 0.06          | 0.19       | 0.85       | 0.31         | 0.12         | 0.467         |
| AdaBoost                     | 0.265    | 0.94          | 0.06          | 0.23       | 0.76       | 0.37         | 0.11         | 0.547         |
| XGBoost                      | 0.227    | 0.96          | 0.06          | 0.19       | 0.87       | 0.31         | 0.12         | 0.478         |
| LightGBM                     | 0.224    | 0.95          | 0.06          | 0.18       | 0.85       | 0.31         | 0.12         | 0.457         |
| CatBoost                     | 0.220    | 0.94          | 0.06          | 0.18       | 0.81       | 0.31         | 0.11         | 0.496         |
| k-Nearest Neighbors          | 0.391    | 0.93          | 0.06          | 0.38       | 0.56       | 0.54         | 0.10         | 0.452         |
| Decision Tree                | 0.214    | 0.94          | 0.06          | 0.17       | 0.84       | 0.29         | 0.11         | 0.506         |
| Naive Bayes                  | 0.374    | 0.93          | 0.05          | 0.36       | 0.55       | 0.52         | 0.10         | 0.392         |
| Linear Discriminant Analysis | 0.573    | 0.96          | 0.09          | 0.57       | 0.63       | 0.72         | 0.15         | 0.622         |
| Quadratic Discriminant Analysis | 0.295 | 0.94          | 0.06          | 0.27       | 0.74       | 0.41         | 0.11         | 0.403         |
| Extra Trees                  | 0.204    | 0.96          | 0.06          | 0.16       | 0.89       | 0.27         | 0.12         | 0.434         |

**Observations of Performance on each model after NearMiss-2:**
Accuracy scores are generally low across most models, ranging from 20.3% (Extra Trees) to 57.3% (Linear Discriminant Analysis). Linear Discriminant Analysis achieves the highest accuracy, but overall, the accuracy for most models is low due to NearMiss-2 undersampling, which disrupts the majority class's influence.

Precision for the non-stroke class (0) remains high across all models, typically above 0.93. This suggests that when these models predict a non-stroke case, they are generally correct, although this high precision is skewed by the undersampling technique, which reduces the dataset’s variability.

Precision for the stroke class (1) is very low, typically around 0.05-0.09, indicating a high number of false positives.The low precision means that models often incorrectly predict stroke cases, resulting in many non-stroke cases being classified as stroke.

Recall for the non-stroke class is low to moderate across most models, with values between 0.16 (Extra Trees) and 0.57 (Linear Discriminant Analysis). Linear Discriminant Analysis achieves the highest recall (0.57), which suggests it can identify a larger portion of true non-stroke cases compared to other models.

Recall for the stroke class is relatively high, ranging from 0.55 (Naive Bayes) to 0.89 (Extra Trees). This indicates that the models capture a large proportion of actual stroke cases. The high recall but low precision indicates that the models are biased toward identifying stroke cases, likely due to NearMiss-2's focus on farthest samples, which increases true positive rates but also introduces more false positives.

The F1-scores for Class 0 vary but remain relatively high, with Linear Discriminant Analysis achieving the highest (0.72). This suggests that, while accuracy and recall for the majority class remain balanced, the NearMiss-2 technique compromises the models' ability to correctly classify non-stroke cases consistently.

The F1-scores for the stroke class are very low, reflecting a struggle to balance precision and recall for the minority class. This low F1-score, generally around 0.10-0.15, highlights the difficulty in achieving high performance for stroke cases due to high false positives.

ROC AUC scores are low to moderate, ranging from 0.391 (Naive Bayes) to 0.622 (Linear Discriminant Analysis).The Linear Discriminant Analysis model achieves the highest ROC AUC score, indicating it is relatively better at distinguishing between stroke and non-stroke cases compared to other models.

**Summary of Performance on each model after NearMiss-2:**
- Linear Discriminant Analysis (LDA) demonstrates the best overall performance after NearMiss-2 undersampling, with the highest accuracy, recall, and ROC AUC score for both classes, making it the most robust model under these conditions.
- High Recall but Low Precision for Stroke Class (Class 1): Most models show high recall and low precision for stroke cases, leading to many false positives. This is likely due to NearMiss-2's strategy, which prioritizes capturing farthest minority samples and skews models toward predicting more positive cases.
- Effectiveness of NearMiss-2: NearMiss-2 increases the sensitivity (recall) for the minority class but at a trade-off in precision and overall accuracy. This can be useful for applications where identifying all possible stroke cases is critical, even at the cost of more false positives.

#### NearMiss-3:
| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-score (0) | F1-score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression          | 0.569    | 0.97          | 0.09          | 0.56       | 0.71       | 0.71         | 0.17         | 0.683         |
| Random Forest                | 0.558    | 0.96          | 0.09          | 0.55       | 0.66       | 0.70         | 0.15         | 0.661         |
| Support Vector Machine       | 0.531    | 0.95          | 0.07          | 0.53       | 0.58       | 0.68         | 0.13         | 0.610         |
| Gradient Boosting            | 0.524    | 0.96          | 0.09          | 0.51       | 0.71       | 0.67         | 0.15         | 0.636         |
| AdaBoost                     | 0.563    | 0.95          | 0.08          | 0.56       | 0.55       | 0.71         | 0.13         | 0.621         |
| XGBoost                      | 0.548    | 0.97          | 0.09          | 0.54       | 0.73       | 0.69         | 0.16         | 0.655         |
| LightGBM                     | 0.569    | 0.97          | 0.09          | 0.56       | 0.71       | 0.71         | 0.17         | 0.656         |
| CatBoost                     | 0.541    | 0.96          | 0.08          | 0.53       | 0.66       | 0.69         | 0.15         | 0.679         |
| k-Nearest Neighbors          | 0.491    | 0.96          | 0.08          | 0.48       | 0.66       | 0.64         | 0.14         | 0.564         |
| Decision Tree                | 0.368    | 0.94          | 0.06          | 0.35       | 0.66       | 0.51         | 0.11         | 0.505         |
| Naive Bayes                  | 0.578    | 0.97          | 0.10          | 0.57       | 0.73       | 0.72         | 0.17         | 0.706         |
| Linear Discriminant Analysis | 0.583    | 0.97          | 0.10          | 0.57       | 0.71       | 0.72         | 0.17         | 0.691         |
| Quadratic Discriminant Analysis | 0.544 | 0.97          | 0.09          | 0.53       | 0.74       | 0.69         | 0.16         | 0.699         |
| Extra Trees                  | 0.505    | 0.97          | 0.08          | 0.49       | 0.73       | 0.65         | 0.15         | 0.651         |

**Observations of Performance on each model after NearMiss-3:**
Accuracy values are moderate across most models, ranging from 36.8% (Decision Tree) to 58.3% (Linear Discriminant Analysis). Linear Discriminant Analysis (LDA) and Naive Bayes achieve the highest accuracy, indicating they manage NearMiss-3 data better than other models.

Precision for the non-stroke class (0) remains high across all models, typically above 0.94, reflecting the models' ability to correctly classify the majority class. Logistic Regression, Naive Bayes, Linear Discriminant Analysis, and Quadratic Discriminant Analysis achieve the highest precision (0.97), indicating consistent performance in predicting non-stroke cases correctly.

Precision for the stroke class (1) is low for all models, generally around 0.06-0.10. This indicates a high rate of false positives, where models incorrectly classify non-stroke cases as strokes. This low precision is expected with NearMiss-3, as it increases the emphasis on samples near decision boundaries, which can lead to higher false positives.

Recall for non-stroke cases varies between models, with values ranging from 0.35 (Decision Tree) to 0.57 (LDA and Naive Bayes). LDA and Naive Bayes achieve the highest recall for the majority class, reflecting a better balance in capturing actual non-stroke cases without being overly biased by the undersampling technique.

Recall for stroke cases is relatively high for most models, ranging from 0.55 (AdaBoost) to 0.74 (Quadratic Discriminant Analysis). High recall with low precision for the stroke class suggests models can identify most stroke cases but misclassify non-stroke cases as strokes. NearMiss-3 achieves high recall by increasing focus on samples near boundaries, but this causes some overlap between classes.

The F1-scores for Class 0 (non-stroke) vary, with Naive Bayes and LDA having the highest F1-scores at 0.72. High F1-scores in these models suggest a balance between precision and recall for the majority class, especially in Naive Bayes and LDA.

The F1-scores for Class 1 are low across all models, generally ranging from 0.11 to 0.17, due to high false positives. Naive Bayes, LDA, and Logistic Regression achieve the highest F1-scores (around 0.17), highlighting that these models maintain a slightly better balance between recall and precision for stroke cases than others.

ROC AUC scores range from 0.505 (Decision Tree) to 0.706 (Naive Bayes). Naive Bayes, LDA, and Quadratic Discriminant Analysis achieve the highest ROC AUC scores, indicating that they handle the near-boundary samples well, distinguishing between the classes more effectively than other models.

**Summary of Performance on each model after NearMiss-3:**
- Naive Bayes and LDA Performance: These two models achieve the highest accuracy, recall, and ROC AUC scores, suggesting they are better suited for NearMiss-3 undersampled data. They manage to identify stroke cases with decent recall without overly sacrificing the non-stroke classification.
- High Recall but Low Precision for Stroke Cases: Most models show high recall and low precision for stroke cases, meaning they tend to identify most stroke cases at the expense of increased false positives. This is consistent with NearMiss-3’s emphasis on selecting samples near the decision boundary.
- Trade-Off in Accuracy and Recall: While models achieve moderate accuracy, there’s a clear trade-off between identifying minority class cases and maintaining precision for the majority class.
  
#### Tomek Links:
| Model                         | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-score (0) | F1-score (1) | ROC AUC Score |
|-------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression           | 0.939    | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.851         |
| Random Forest                 | 0.940    | 0.94          | 1.00          | 1.00       | 0.02       | 0.97         | 0.03         | 0.816         |
| SVM                           | 0.939    | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.629         |
| Gradient Boosting             | 0.940    | 0.94          | 1.00          | 1.00       | 0.02       | 0.97         | 0.03         | 0.843         |
| AdaBoost                      | 0.940    | 0.94          | 0.67          | 1.00       | 0.03       | 0.97         | 0.06         | 0.792         |
| XGBoost                       | 0.934    | 0.94          | 0.35          | 0.99       | 0.10       | 0.97         | 0.15         | 0.805         |
| LightGBM                      | 0.938    | 0.94          | 0.46          | 0.99       | 0.10       | 0.97         | 0.16         | 0.822         |
| CatBoost                      | 0.938    | 0.94          | 0.40          | 1.00       | 0.03       | 0.97         | 0.06         | 0.818         |
| k-Nearest Neighbors           | 0.940    | 0.94          | 1.00          | 1.00       | 0.02       | 0.97         | 0.03         | 0.659         |
| Decision Tree                 | 0.903    | 0.95          | 0.16          | 0.95       | 0.15       | 0.95         | 0.15         | 0.549         |
| Naive Bayes                   | 0.864    | 0.96          | 0.21          | 0.89       | 0.47       | 0.92         | 0.29         | 0.828         |
| Linear Discriminant Analysis  | 0.931    | 0.94          | 0.24          | 0.99       | 0.06       | 0.96         | 0.10         | 0.841         |
| Quadratic Discriminant Analysis | 0.877 | 0.97          | 0.25          | 0.90       | 0.50       | 0.93         | 0.33         | 0.830         |
| Extra Trees                   | 0.932    | 0.94          | 0.11          | 0.99       | 0.02       | 0.97         | 0.03         | 0.765         |

**Observations of Performance on each model after Tomek Links:**

Most models achieve high accuracy, with values around 94%. Logistic Regression, Random Forest, SVM, and Gradient Boosting models all display accuracy close to 94%, showing that they are generally accurate in classifying the majority (non-stroke) class. The Decision Tree model has a slightly lower accuracy of 90.3%, indicating that it struggles more with distinguishing classes even after TomekLinks.

Precision for the non-stroke class (0) remains high across all models, generally above 0.94. This shows that these models effectively predict non-stroke cases accurately without frequent false positives. Models such as Logistic Regression, Random Forest, and LightGBM achieve some of the highest precision values for class 0, indicating they are better at minimizing false positives in the majority class.

Precision for the stroke class (1) is low across most models, generally near 0.10 or lower, indicating a high rate of false positives. LightGBM and XGBoost achieve slightly higher precision for the minority class, around 0.35 to 0.46, showing some improvement in identifying stroke cases, though still limited.

Recall values for the non-stroke class are near 1.00 for most models, reflecting their ability to correctly identify nearly all non-stroke cases. This high recall for non-stroke cases suggests that the models tend to favor the majority class, which aligns with the high accuracy scores observed.

Recall for the stroke class is generally poor, with models like Logistic Regression, SVM, and CatBoost failing to detect any stroke cases (recall close to 0).
Models like Naive Bayes and Quadratic Discriminant Analysis have recall values around 0.47–0.50 for stroke cases, meaning they are slightly better at identifying stroke instances but still limited.

The F1-score for the non-stroke class remains high, averaging around 0.95 across most models, indicating a good balance between precision and recall for the majority class. This metric further emphasizes the models’ strong performance in correctly identifying non-stroke cases.

The F1-scores for stroke cases are very low, with most models scoring below 0.20, due to both low recall and precision. Naive Bayes and Quadratic Discriminant Analysis show slightly better F1-scores around 0.29–0.33, reflecting a modest balance between precision and recall.

ROC AUC scores vary from 0.55 to 0.85, with Logistic Regression achieving the highest AUC score at 0.85, suggesting better discrimination between classes than other models. Naive Bayes and Quadratic Discriminant Analysis also achieve AUC scores around 0.83, showing that they are relatively better at differentiating stroke cases.

**Summary of Performance on each model after Tomek Links:**
- **Effectiveness of TomekLinks:** While TomekLinks successfully cleans noise from overlapping majority samples, it does not significantly improve the model’s ability to predict the minority (stroke) class. Most models still struggle with detecting stroke cases, shown by low recall and precision values.
- **High Precision and Recall for Non-Stroke Cases:** Models demonstrate consistently high performance for non-stroke cases in both precision and recall, resulting in high accuracy. This is a result of the dataset still being imbalanced after TomekLinks is applied, leading models to prioritize the majority class.
- **Naive Bayes and Quadratic Discriminant Analysis Outperform in Minority Class Detection:** These models exhibit slightly higher recall and AUC scores for the minority class, suggesting they may better handle imbalanced data. However, precision for stroke cases is still low, indicating that these models, while more sensitive, are not without limitations.

#### Edited Nearest Neighbors (ENN)::
**Observations of Performance on each model after Edited Nearest Neighbors (ENN):**

**Summary of Performance on each model after Edited Nearest Neighbors (ENN):**

#### Repeated Edited Nearest Neighbors (RENN):
**Observations of Performance on each model after :**
**Summary of Performance on each model after :**

#### All KNN:
**Observations of Performance on each model after :**
**Summary of Performance on each model after :**

#### Condensed Nearest Neighbor (CNN):
**Observations of Performance on each model after :**
**Summary of Performance on each model after :**

#### One-Sided Selection (OSS):
**Observations of Performance on each model after :**
**Summary of Performance on each model after :**

#### Neighborhood Cleaning Rule (NCR):
**Observations of Performance on each model after :**
**Summary of Performance on each model after :**

#### Instance Hardness Threshold:
**Observations of Performance on each model after :**
**Summary of Performance on each model after :**


## Conclusion

## Discussion

## Future Work
