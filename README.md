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
The results section will detail the performance of models on both the imbalanced and undersampled datasets.
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

The **Performance on Imbalanced Data** highlights the challenges models face when the dataset is highly skewed toward the majority class. In this context, models like **Logistic Regression**, **Support Vector Machine (SVM)**, **Random Forest**, and **K-Nearest Neighbors (K-NN)** achieved high overall accuracy (0.94) due to their dominance in predicting the majority class accurately. However, their precision and recall scores for the minority class are consistently low or even zero, indicating that these models failed to effectively capture the minority class instances, reflecting their limitations in handling class imbalance.

Ensemble models like **Gradient Boosting**, **XGBoost**, and **CatBoost** slightly improved minority class performance. For example, **Gradient Boosting** and **LightGBM** achieved moderate precision (0.33 for Gradient Boosting and LightGBM) and recall (0.02 and 0.06, respectively) for the minority class, resulting in ROC AUC scores above 0.8. **CatBoost** reached a minority precision of 0.67, though with a recall of just 0.03. These results suggest that ensemble models show a slight advantage in identifying minority instances, albeit insufficient to meaningfully improve minority class performance.

The **Decision Tree** model performed reasonably well, achieving an accuracy of 0.92 and a more balanced ROC AUC of 0.580. However, the model’s recall for the minority class remains low at 0.19, demonstrating a need for further tuning or rebalancing techniques to effectively manage imbalanced data. Similarly, **Extra Trees** achieved high accuracy (0.94) but low minority class recall (0.03), highlighting that individual tree-based models often struggle to identify minority classes without sampling techniques.

**Naive Bayes** and **Quadratic Discriminant Analysis (QDA)** offered a comparatively better balance between classes, with QDA achieving an accuracy of 0.88 and a minority class recall of 0.45. These probabilistic models managed to identify more minority instances than other models, showing some resilience to imbalance. **Naive Bayes** achieved a moderate ROC AUC score of 0.829, showing better performance in classifying both classes compared to deterministic models like SVM and K-NN, which struggled.

**Linear Discriminant Analysis (LDA)** displayed good overall performance with an accuracy of 0.93 and a ROC AUC of 0.842, slightly better than other linear models but still limited in minority recall. The **Quadratic Discriminant Analysis** achieved a similar ROC AUC (0.830) and demonstrated moderate precision for the minority class, reflecting the model’s ability to handle class overlap moderately well in an imbalanced scenario.

In summary, on imbalanced data, models tended to prioritize the majority class at the expense of the minority class, as shown by consistently high precision for the majority class and low recall for the minority class. **Ensemble models** like **XGBoost** and **CatBoost** slightly outperformed others, while **probabilistic models** like **Naive Bayes** and **QDA** demonstrated a relatively better balance. However, most models showed significant limitations in minority class detection, underscoring the importance of using rebalancing techniques to achieve a more equitable model performance on imbalanced datasets.

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

ClusterCentroids undersampling proved beneficial for some models while less effective for others, particularly ensemble models. **Logistic Regression** maintained good overall accuracy (0.744) and ROC AUC (0.831), balancing recall and precision between classes. This suggests it adapted well to the simplified dataset without heavily relying on data complexity. Similarly, **Support Vector Machine (SVM)** showed comparable performance, with an accuracy of 0.733 and a high ROC AUC score of 0.815. These metrics indicate that SVM managed to handle the simplified boundaries provided by centroids efficiently. **Naive Bayes** and **Linear Discriminant Analysis (LDA)** also excelled in this setup, achieving balanced accuracy and high ROC AUC scores of 0.836 and 0.830, respectively. Both models demonstrated good adaptability to the centroid-based data structure, making them suitable choices for this type of undersampling.

In contrast, several ensemble models, including **Random Forest**, **Gradient Boosting**, **AdaBoost**, **XGBoost**, **LightGBM**, and **CatBoost**, faced challenges with ClusterCentroids undersampling. Random Forest, for example, suffered a significant drop in accuracy (0.440), as did Gradient Boosting (0.321) and AdaBoost (0.355), indicating a reliance on finer data details, which were diminished with centroids. Although some of these models achieved high minority recall, they struggled with minority precision, impacting their F1 scores and indicating difficulty in distinguishing between classes with limited data. These ensemble models appear to rely more on nuanced boundary details that were lost in centroid sampling, leading to lower effectiveness.

**k-Nearest Neighbors (k-NN)** adapted reasonably well to the undersampling approach, achieving an accuracy of 0.746 and an ROC AUC of 0.745. Though its minority class precision was slightly lower, k-NN managed to perform with balance, showing adaptability to the centroid undersampling. Among the tree-based models, **Decision Tree** struggled the most, with a low accuracy of 0.392 and a poor ROC AUC score (0.639). **Extra Trees** performed somewhat better than Decision Tree with an accuracy of 0.545 and ROC AUC of 0.796, but both models displayed reduced effectiveness with ClusterCentroids undersampling, indicating that tree-based methods may not be optimal for this data reduction approach.

In summary, simpler models such as **Logistic Regression**, **Naive Bayes**, **SVM**, and **LDA** benefitted the most from ClusterCentroids undersampling, maintaining consistent performance across key metrics. These models demonstrated resilience to the simplified class boundaries introduced by the centroid-based approach. Conversely, ensemble models like **Random Forest** and **Gradient Boosting** and tree-based models like **Decision Tree** struggled with the reduced data complexity, showing that they may require more granular boundary details to perform optimally. Overall, ClusterCentroids undersampling is best suited for models that can effectively handle simplified boundaries, while more complex models may suffer performance reductions with this technique.

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

Random Undersampling demonstrated stable and balanced performance across various models, particularly in handling imbalanced classes. **Logistic Regression** maintained a moderate accuracy (0.728) and an improved ROC AUC score (0.846), achieving balanced recall and precision values across both classes. This suggests that Logistic Regression can manage class imbalance well with this straightforward undersampling method. **Support Vector Machine (SVM)** also achieved similar results, with an accuracy of 0.696 and a high ROC AUC score (0.824), indicating its ability to effectively separate the classes using fewer samples. The high precision and recall for the majority class suggest that SVM remained consistent in classifying the more frequent class, while still achieving moderate minority class recall.

Among the ensemble models, **Random Forest**, **Gradient Boosting**, **AdaBoost**, **XGBoost**, and **CatBoost** also performed relatively well under Random Undersampling. These models maintained accuracies between 0.703 and 0.725, with Random Forest and CatBoost achieving the highest ROC AUC scores of 0.824 and 0.842, respectively. While the recall for the minority class was lower than for the majority, these models demonstrated adaptability to the undersampled dataset, retaining fairly balanced F1-scores across classes. However, lower recall for the minority class indicates that while these ensemble models are flexible with undersampling, they may still prioritize the majority class without more advanced handling techniques.

**k-Nearest Neighbors (k-NN)** showed balanced but lower overall performance, with an accuracy of 0.678 and an ROC AUC of 0.778. Despite this, k-NN managed to maintain high precision and recall for the majority class, though the minority class recall was slightly lower. Tree-based models like **Decision Tree** and **Extra Trees** demonstrated decent adaptability, with Decision Tree achieving an accuracy of 0.680 and Extra Trees 0.697. The F1-scores and ROC AUC scores suggest that these models can still perform reasonably well with Random Undersampling, although precision and recall for the minority class were somewhat limited.

**Naive Bayes**, **Linear Discriminant Analysis (LDA)**, and **Quadratic Discriminant Analysis (QDA)** showed the strongest results with Random Undersampling, highlighting their resilience to class imbalance reduction techniques. Naive Bayes achieved an accuracy of 0.725 and ROC AUC of 0.832, while LDA and QDA achieved high ROC AUC scores of 0.848 and 0.833, respectively. These models showed balanced metrics, which implies that they adapted well to the reduced sample size and managed to achieve good classification for both classes, making them excellent candidates for datasets processed with Random Undersampling.

In summary, Random Undersampling proved to be an effective method for models that can handle class imbalance reduction well, such as **Logistic Regression**, **Naive Bayes**, **SVM**, and **LDA/QDA**. These models maintained balanced performance metrics and demonstrated adaptability to the undersampled data. Ensemble models like **Random Forest** and **CatBoost** performed moderately well, though their sensitivity to minority class detection was limited. Tree-based models like **Decision Tree** and **Extra Trees** showed average performance, suggesting they may not be ideal for undersampled data without additional handling. Overall, Random Undersampling is suitable for models that can work with simplified data distributions without losing classification performance, particularly for balancing class metrics.

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

The NearMiss-1 undersampling technique presented unique challenges for various models due to its selection of samples closest to the decision boundary, which aims to increase the minority class’s importance in the model training process. **Logistic Regression** showed moderate performance with an accuracy of 0.469 and an ROC AUC score of 0.631, benefiting somewhat from the emphasis on boundary samples. Despite a balanced recall across classes, minority precision remained low, suggesting that while NearMiss-1 improved boundary clarity, Logistic Regression struggled with distinguishing minority cases accurately.

**Support Vector Machine (SVM)** exhibited similar limitations, achieving an accuracy of 0.359 and an ROC AUC score of 0.522. SVM’s performance indicates its capability to capture boundary-based class separations but shows its susceptibility to overfitting with fewer data points, particularly when the minority precision remained low. Both **Random Forest** and **Gradient Boosting** faced challenges, with Random Forest reaching an accuracy of only 0.252 and an ROC AUC of 0.529, and Gradient Boosting at 0.285 accuracy with a lower ROC AUC (0.458). This suggests that ensemble methods relying on intricate data distributions and feature interdependencies may find NearMiss-1 undersampling less effective, as it removes contextually relevant samples that aid in their decision-making processes.

Among ensemble models, **AdaBoost** and **XGBoost** similarly struggled, each presenting low accuracy and ROC AUC scores. AdaBoost achieved an accuracy of 0.283 with an ROC AUC of 0.395, while XGBoost performed marginally better with an accuracy of 0.310 and an ROC AUC of 0.566. This further highlights that NearMiss-1’s focus on simplifying boundaries may not support ensemble models' depth of learning.

**Naive Bayes** and **Linear Discriminant Analysis (LDA)** displayed better resilience to this undersampling, achieving accuracies of 0.504 and 0.622, respectively, and higher ROC AUC scores of 0.510 for Naive Bayes and 0.668 for LDA. LDA, in particular, handled the simplified boundaries with balanced recall and precision, making it a suitable model for datasets processed with NearMiss-1. **Quadratic Discriminant Analysis (QDA)** also performed reasonably well with a 0.517 accuracy and a 0.500 ROC AUC, demonstrating some adaptability to boundary-focused undersampling.

**k-Nearest Neighbors (k-NN)** adapted moderately well, achieving a 0.480 accuracy and an ROC AUC of 0.581. While k-NN showed balanced precision and recall for the majority class, it struggled with minority precision, suggesting that the model had difficulty with class separation under simplified data structures. **Decision Tree** and **Extra Trees** were less effective, as evidenced by their low accuracies (0.196 for Decision Tree and 0.281 for Extra Trees) and low ROC AUC scores, highlighting that tree-based methods may not be optimal for NearMiss-1, as they rely on more granular data features that are diminished with this undersampling method.

In summary, **Naive Bayes**, **LDA**, and **QDA** benefitted most from NearMiss-1 undersampling, maintaining consistent performance in ROC AUC scores. Models such as **Logistic Regression** and **k-NN** adapted moderately well but showed limited minority precision, reflecting the trade-offs of boundary-based undersampling. Ensemble and tree-based models, including **Random Forest**, **Gradient Boosting**, **AdaBoost**, and **Extra Trees**, encountered significant performance drops, indicating that NearMiss-1 undersampling may not align with the complex decision-making requirements of these models. Overall, NearMiss-1 is best suited for simpler models that can adapt to clear boundary distinctions without heavily relying on extensive data complexity.

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

The NearMiss-2 undersampling technique, which further reduces the majority class by focusing on samples near the minority class, presented distinct challenges for various models. **Logistic Regression** achieved an accuracy of 0.520 with an ROC AUC score of 0.595, reflecting a reasonable balance in recall across classes, particularly for minority recall. Although the minority precision remained low, Logistic Regression demonstrated a moderate ability to handle the boundary-focused data structure, making it relatively adaptable to the undersampled dataset.

**Support Vector Machine (SVM)** and **Random Forest** both struggled under NearMiss-2, with SVM obtaining an accuracy of 0.219 and an ROC AUC of 0.460, and Random Forest an even lower accuracy of 0.210 and ROC AUC of 0.468. These results indicate that both models faced challenges in distinguishing between classes effectively. SVM’s low minority precision and Random Forest’s limited recall highlight that both models may require a more detailed data structure to perform optimally.

Ensemble methods, including **Gradient Boosting**, **AdaBoost**, **XGBoost**, and **LightGBM**, faced significant performance reductions with NearMiss-2. Gradient Boosting and AdaBoost both showed limited minority recall, with accuracies of 0.228 and 0.265, respectively, and similarly low ROC AUC scores (0.467 for Gradient Boosting and 0.547 for AdaBoost). XGBoost and LightGBM, with accuracies of 0.227 and 0.224, respectively, also exhibited reduced effectiveness. These results indicate that ensemble models struggled with the simplified boundaries produced by NearMiss-2, as they generally benefit from richer data complexity and inter-feature relationships.

**Naive Bayes** and **Linear Discriminant Analysis (LDA)** displayed stronger adaptability, with Naive Bayes reaching an accuracy of 0.374 and an ROC AUC of 0.392, while LDA performed the best among all models with an accuracy of 0.573 and an ROC AUC of 0.622. LDA, with balanced recall and precision for both classes, managed to handle the simplified boundaries more effectively than other models, suggesting it as a suitable choice for data preprocessed with NearMiss-2.

**Quadratic Discriminant Analysis (QDA)** performed moderately well with an accuracy of 0.295 and an ROC AUC score of 0.403, suggesting that, while not as robust as LDA, it still managed the simplified dataset structure better than ensemble methods. **k-Nearest Neighbors (k-NN)** and **Decision Tree** also showed moderate performance, achieving accuracies of 0.391 and 0.214, respectively. These models demonstrated limited minority recall, but k-NN particularly showed some adaptability due to its reliance on nearest neighbors for classification.

In summary, **LDA** and **Logistic Regression** emerged as the most effective models for handling NearMiss-2 undersampling, as they adapted well to the data's simplified boundaries, maintaining a reasonable balance across classes. Ensemble models such as **Random Forest**, **Gradient Boosting**, **AdaBoost**, and **XGBoost** showed significant reductions in performance, suggesting that these models may require a more nuanced data structure to perform optimally. **Naive Bayes** and **QDA** also showed moderate adaptability but struggled with low precision for the minority class, indicating that NearMiss-2 undersampling is better suited for simpler models that can operate effectively with basic boundary distinctions.

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

The NearMiss-3 undersampling technique, which further emphasizes samples from the majority class closest to the minority class, led to moderate performance among the models tested. **Logistic Regression** demonstrated reasonable adaptability to this undersampling approach with an accuracy of 0.569 and a ROC AUC score of 0.683. With balanced recall across classes (0.56 for class 0 and 0.71 for class 1), Logistic Regression maintained acceptable performance, though it struggled with precision for the minority class, indicating that it was moderately successful at identifying minority samples in this data setup.

**Random Forest** also performed relatively well under NearMiss-3, achieving an accuracy of 0.558 and a ROC AUC score of 0.661. Its recall for the minority class was reasonable (0.66), but similar to Logistic Regression, it experienced low precision for the minority class. This trend suggests that Random Forest could handle the simplified boundaries created by NearMiss-3 but struggled to reliably classify the minority class with high confidence.

Among the ensemble models, **Gradient Boosting**, **AdaBoost**, **XGBoost**, and **LightGBM** showed varied results. Gradient Boosting reached an accuracy of 0.524, and XGBoost performed slightly better with an accuracy of 0.548. Both models achieved similar minority recall scores (0.71 for Gradient Boosting and 0.73 for XGBoost), indicating reasonable adaptability to the undersampled data. However, minority precision remained low for both models, which impacted their overall effectiveness in distinguishing the classes accurately. LightGBM had an accuracy of 0.569 and an ROC AUC score of 0.656, aligning closely with Logistic Regression’s performance, suggesting that LightGBM was one of the more adaptable ensemble models in this setting.

The simpler models, **Naive Bayes** and **Linear Discriminant Analysis (LDA)**, showed stronger performance with NearMiss-3. Naive Bayes achieved an accuracy of 0.578 with an ROC AUC score of 0.706, while LDA performed slightly better, reaching an accuracy of 0.583 and an ROC AUC of 0.691. Both models demonstrated balanced recall and reasonable F1-scores for the minority class, indicating that they were well-suited for the boundaries created by NearMiss-3 undersampling.

**Quadratic Discriminant Analysis (QDA)** and **k-Nearest Neighbors (k-NN)** showed moderate adaptability to NearMiss-3 undersampling. QDA achieved an accuracy of 0.544 and a ROC AUC score of 0.699, indicating some sensitivity to the minority class, although precision remained low. k-NN achieved an accuracy of 0.491 and an ROC AUC of 0.564, performing moderately but facing challenges in minority precision, suggesting a limited capacity for accurately distinguishing classes with the NearMiss-3 undersampling structure.

In summary, simpler models such as **Logistic Regression**, **Naive Bayes**, and **LDA** performed best with the boundaries established by NearMiss-3 undersampling, showing balanced accuracy and sensitivity toward the minority class. Ensemble models like **Random Forest** and **LightGBM** adapted moderately well, though they required more data complexity to perform optimally. **Gradient Boosting** and **XGBoost** maintained reasonable recall but faced limitations in distinguishing between classes due to lower minority precision. Overall, NearMiss-3 undersampling appears better suited for simpler models, as they can handle the clear but minimal boundaries without a significant drop in performance.
  
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

The **Tomek Links** undersampling method, which removes majority samples that are closest to minority samples, showed mixed effectiveness across the models. This technique largely retained the dataset’s balance, leading to high accuracy across many models but often at the cost of low recall for the minority class. **Logistic Regression** achieved a high accuracy of 0.939 and an ROC AUC score of 0.851. However, its minority recall and precision dropped to zero, indicating that the model was unable to identify any instances from the minority class. This trend suggests that Tomek Links helped create cleaner class boundaries but did not improve the model’s sensitivity to minority samples.

Ensemble models like **Random Forest**, **Gradient Boosting**, **AdaBoost**, and **CatBoost** demonstrated strong accuracy, typically around 0.940, but faced significant challenges with minority recall. For instance, Random Forest achieved an accuracy of 0.940 and an F1-score of 0.03 for the minority class, which highlights its limited ability to capture minority instances accurately. Both Gradient Boosting and CatBoost performed similarly, achieving high accuracy but struggling to identify the minority class effectively. These results suggest that while ensemble methods maintain overall accuracy with Tomek Links, their high precision for the majority class overshadows minority class identification.

**Support Vector Machine (SVM)**, with an accuracy of 0.939 and a ROC AUC of 0.629, showed minimal effectiveness in capturing minority class data, performing only slightly better than random chance. This indicates that SVM might be highly sensitive to the majority-minority imbalance despite the Tomek Links' attempt at boundary cleaning. **k-Nearest Neighbors (k-NN)** also faced similar issues, with an accuracy of 0.940 but limited capacity to recognize minority samples.

On the other hand, **Naive Bayes** and **Quadratic Discriminant Analysis (QDA)** models showed relatively better performance with Tomek Links undersampling. Naive Bayes had a slightly lower accuracy of 0.864 but displayed a stronger minority class recall (0.47), resulting in a relatively balanced ROC AUC score of 0.828. QDA similarly achieved a reasonable accuracy of 0.877 and an ROC AUC score of 0.830, suggesting that these models were somewhat better suited to the undersampling approach, maintaining sensitivity to the minority class while balancing majority precision.

In summary, **Tomek Links undersampling** resulted in high overall accuracy for most models, indicating a cleaner boundary between classes. However, models like **Logistic Regression**, **SVM**, and **ensemble methods** struggled to recognize the minority class, with zero or near-zero recall. **Naive Bayes** and **QDA** showed better adaptability by achieving a modest balance between accuracy and sensitivity toward the minority class, suggesting they may be more appropriate choices when using Tomek Links undersampling in highly imbalanced datasets.

#### Edited Nearest Neighbors (ENN):
| Model                         | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-score (0) | F1-score (1) | ROC AUC Score |
|-------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression           | 0.937    | 0.94          | 0.25          | 1.00       | 0.02       | 0.97         | 0.03         | 0.852         |
| Random Forest                 | 0.938    | 0.94          | 0.45          | 0.99       | 0.08       | 0.97         | 0.14         | 0.813         |
| SVM                           | 0.937    | 0.94          | 0.25          | 1.00       | 0.02       | 0.97         | 0.03         | 0.709         |
| Gradient Boosting             | 0.937    | 0.95          | 0.44          | 0.99       | 0.11       | 0.97         | 0.18         | 0.851         |
| AdaBoost                      | 0.937    | 0.94          | 0.40          | 0.99       | 0.06       | 0.97         | 0.11         | 0.839         |
| XGBoost                       | 0.928    | 0.95          | 0.32          | 0.98       | 0.18       | 0.96         | 0.23         | 0.808         |
| LightGBM                      | 0.931    | 0.94          | 0.30          | 0.98       | 0.11       | 0.96         | 0.16         | 0.840         |
| CatBoost                      | 0.932    | 0.95          | 0.35          | 0.98       | 0.13       | 0.96         | 0.19         | 0.836         |
| k-Nearest Neighbors           | 0.935    | 0.95          | 0.40          | 0.99       | 0.13       | 0.97         | 0.20         | 0.674         |
| Decision Tree                 | 0.886    | 0.95          | 0.19          | 0.93       | 0.26       | 0.94         | 0.22         | 0.593         |
| Naive Bayes                   | 0.845    | 0.97          | 0.20          | 0.87       | 0.53       | 0.91         | 0.29         | 0.825         |
| Linear Discriminant Analysis  | 0.925    | 0.95          | 0.35          | 0.97       | 0.27       | 0.96         | 0.31         | 0.837         |
| Quadratic Discriminant Analysis | 0.850  | 0.97          | 0.21          | 0.87       | 0.53       | 0.92         | 0.30         | 0.826         |
| Extra Trees                   | 0.927    | 0.95          | 0.29          | 0.98       | 0.15       | 0.96         | 0.19         | 0.787         |

The **Edited Nearest Neighbors (ENN)** undersampling method, which removes samples from the majority class that differ from their neighbors, achieved mixed results across the models. **Logistic Regression** maintained high accuracy at 0.937 with a relatively strong ROC AUC score of 0.852, but its ability to identify the minority class was still limited, evidenced by low minority recall and precision values. This result suggests that while ENN improved boundary separation, it didn’t enhance the model's sensitivity toward the minority class.

**Random Forest**, **Gradient Boosting**, and **AdaBoost** showed high overall accuracy (around 0.937) and slightly improved minority recall compared to other undersampling techniques, albeit still limited. Random Forest, for instance, achieved a minority recall of 0.08 and an F1-score of 0.14 for the minority class, which, while better than some other techniques, indicates a persistent struggle in distinguishing minority samples effectively. Gradient Boosting exhibited a similar pattern with high accuracy and slightly better recall, suggesting that these ensemble models were able to partially adapt to the cleaner class boundaries provided by ENN.

**Support Vector Machine (SVM)** demonstrated high accuracy (0.937) but, like Logistic Regression, showed minimal sensitivity toward the minority class, with recall remaining at 0.02 and an ROC AUC score of 0.709. This suggests that, for SVM, ENN was not significantly effective in improving minority class detection. **k-Nearest Neighbors (k-NN)**, however, managed a higher minority recall of 0.13 and an F1-score of 0.20, with an accuracy of 0.935, showing that k-NN benefitted somewhat from the neighborhood-based undersampling of ENN.

**Naive Bayes** and **Quadratic Discriminant Analysis (QDA)** had slightly lower overall accuracy (0.845 and 0.850, respectively) but demonstrated better adaptability to minority detection, achieving higher minority recall values (0.53) and relatively balanced ROC AUC scores (0.825 and 0.826). These metrics indicate that probabilistic models like Naive Bayes and QDA may handle the more balanced class distribution introduced by ENN better than other models, as they could more effectively capture minority patterns while maintaining majority class performance.

**Linear Discriminant Analysis (LDA)** also showed relatively balanced performance, with an accuracy of 0.925 and a modest minority recall of 0.27, achieving a fair ROC AUC score of 0.837. This balance suggests that LDA benefitted from the boundary-focused undersampling approach, resulting in a model that could perform well for both classes to some degree.

In summary, **ENN undersampling** provided high overall accuracy across most models while yielding modest improvements in minority recall, particularly for ensemble methods like Random Forest and Gradient Boosting. However, models such as **Naive Bayes**, **QDA**, and **LDA** demonstrated the most balanced performance, indicating that probabilistic or linear models might be better suited to ENN undersampling. On the other hand, **SVM** and **Logistic Regression** continued to struggle with minority class detection, showing that ENN’s boundary refinement alone may not be sufficient to enhance minority sensitivity in certain models.

#### Repeated Edited Nearest Neighbors (RENN):
| Model                           | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-score (0) | F1-score (1) | ROC AUC Score |
|---------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression             | 0.934    | 0.95          | 0.41          | 0.98       | 0.18       | 0.97         | 0.25         | 0.849         |
| Random Forest                   | 0.924    | 0.95          | 0.33          | 0.97       | 0.24       | 0.96         | 0.28         | 0.809         |
| SVM                             | 0.931    | 0.95          | 0.34          | 0.98       | 0.16       | 0.96         | 0.22         | 0.733         |
| Gradient Boosting               | 0.917    | 0.95          | 0.27          | 0.96       | 0.21       | 0.96         | 0.23         | 0.853         |
| AdaBoost                        | 0.928    | 0.95          | 0.37          | 0.97       | 0.27       | 0.96         | 0.31         | 0.831         |
| XGBoost                         | 0.909    | 0.95          | 0.24          | 0.95       | 0.23       | 0.95         | 0.23         | 0.834         |
| LightGBM                        | 0.919    | 0.95          | 0.29          | 0.96       | 0.23       | 0.96         | 0.25         | 0.842         |
| CatBoost                        | 0.920    | 0.96          | 0.33          | 0.96       | 0.32       | 0.96         | 0.33         | 0.839         |
| k-Nearest Neighbors             | 0.920    | 0.95          | 0.31          | 0.96       | 0.26       | 0.96         | 0.28         | 0.694         |
| Decision Tree                   | 0.867    | 0.95          | 0.15          | 0.91       | 0.26       | 0.93         | 0.19         | 0.582         |
| Naive Bayes                     | 0.828    | 0.97          | 0.19          | 0.84       | 0.56       | 0.90         | 0.28         | 0.823         |
| Linear Discriminant Analysis    | 0.904    | 0.96          | 0.30          | 0.93       | 0.44       | 0.95         | 0.36         | 0.830         |
| Quadratic Discriminant Analysis | 0.832    | 0.97          | 0.19          | 0.85       | 0.53       | 0.90         | 0.28         | 0.823         |
| Extra Trees                     | 0.898    | 0.96          | 0.24          | 0.94       | 0.32       | 0.95         | 0.28         | 0.777         |

The **Repeated Edited Nearest Neighbors (RENN)** undersampling technique, which iteratively removes misclassified majority class samples, demonstrated significant improvements in minority class performance across various models. **Logistic Regression** maintained high accuracy (0.934) and displayed better minority class recall (0.18) and precision (0.41) compared to simpler undersampling methods, leading to a balanced F1-score of 0.25 for the minority class and an ROC AUC of 0.849. These metrics suggest that Logistic Regression was able to leverage the cleaned data more effectively, although it still showed some limitations in sensitivity to the minority class.

Among ensemble models, **Random Forest**, **AdaBoost**, and **CatBoost** benefited notably from the RENN technique, achieving higher minority recall values. Random Forest, for example, achieved an accuracy of 0.924 with a minority recall of 0.24, and AdaBoost performed similarly with an accuracy of 0.928 and minority recall of 0.27. CatBoost achieved a particularly balanced performance, with both precision (0.33) and recall (0.32) for the minority class, indicating effective use of the iterative data cleaning approach. This balance is reflected in its strong F1-score of 0.33 for the minority class and an ROC AUC score of 0.839. These results indicate that ensemble methods, especially CatBoost, adapted well to the iterative nature of RENN, managing to balance the recall and precision across both classes.

**Support Vector Machine (SVM)** also showed reasonable performance improvements with RENN, achieving an accuracy of 0.931 and an ROC AUC of 0.733, though its minority recall remained relatively low at 0.16. This suggests that SVM struggled to leverage the minority samples effectively, likely due to its reliance on clearer boundaries. **k-Nearest Neighbors (k-NN)**, with an accuracy of 0.920 and a moderate minority recall (0.26), was better suited to RENN, achieving balanced precision and recall for both classes, making it relatively effective in this setting.

Probabilistic models like **Naive Bayes** and **Quadratic Discriminant Analysis (QDA)** displayed more balanced performance with RENN, showing improved minority recall values of 0.56 and 0.53, respectively, although their overall accuracy (0.828 for Naive Bayes and 0.832 for QDA) was somewhat lower than other models. These models achieved higher ROC AUC scores of around 0.823, suggesting they could better identify minority instances in the cleaned dataset.

**Linear Discriminant Analysis (LDA)** achieved notable success with RENN, maintaining a high accuracy (0.904) and a better balance between classes, with a minority recall of 0.44 and an ROC AUC of 0.830. This improvement shows that LDA could handle the more balanced data created by RENN while maintaining overall performance.

In summary, **RENN undersampling** enhanced the performance of most models in detecting minority classes, particularly for ensemble methods like **Random Forest**, **AdaBoost**, and **CatBoost**, which demonstrated improvements in both minority recall and precision. Probabilistic models such as **Naive Bayes**, **QDA**, and **LDA** also performed well, showing balanced metrics that reflect adaptability to this undersampling approach. However, **SVM** and **Logistic Regression** still displayed limitations in handling minority class sensitivity, although they achieved high overall accuracy. Overall, RENN proved to be effective for models that benefit from iterative data cleaning, enabling better representation of minority classes.

#### All KNN:
| Model                           | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-score (0) | F1-score (1) | ROC AUC Score |
|---------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression             | 0.934    | 0.94          | 0.31          | 0.99       | 0.06       | 0.97         | 0.11         | 0.851         |
| Random Forest                   | 0.930    | 0.95          | 0.31          | 0.98       | 0.13       | 0.96         | 0.18         | 0.816         |
| SVM                             | 0.936    | 0.94          | 0.38          | 0.99       | 0.08       | 0.97         | 0.13         | 0.733         |
| Gradient Boosting               | 0.931    | 0.95          | 0.35          | 0.98       | 0.18       | 0.96         | 0.24         | 0.847         |
| AdaBoost                        | 0.932    | 0.95          | 0.33          | 0.99       | 0.11       | 0.96         | 0.17         | 0.831         |
| XGBoost                         | 0.928    | 0.95          | 0.35          | 0.97       | 0.23       | 0.96         | 0.27         | 0.823         |
| LightGBM                        | 0.924    | 0.95          | 0.29          | 0.97       | 0.18       | 0.96         | 0.22         | 0.842         |
| CatBoost                        | 0.928    | 0.95          | 0.31          | 0.98       | 0.16       | 0.96         | 0.21         | 0.835         |
| k-Nearest Neighbors             | 0.933    | 0.95          | 0.40          | 0.98       | 0.19       | 0.97         | 0.26         | 0.676         |
| Decision Tree                   | 0.882    | 0.95          | 0.18          | 0.92       | 0.26       | 0.94         | 0.21         | 0.590         |
| Naive Bayes                     | 0.839    | 0.97          | 0.20          | 0.86       | 0.55       | 0.91         | 0.29         | 0.825         |
| Linear Discriminant Analysis    | 0.923    | 0.96          | 0.36          | 0.96       | 0.34       | 0.96         | 0.35         | 0.836         |
| Quadratic Discriminant Analysis | 0.844    | 0.97          | 0.20          | 0.86       | 0.53       | 0.91         | 0.29         | 0.825         |
| Extra Trees                     | 0.915    | 0.95          | 0.21          | 0.96       | 0.15       | 0.96         | 0.17         | 0.773         |

The **All KNN** undersampling technique, which removes majority class samples that do not align with the majority vote within their neighborhood, proved effective for certain models, particularly those that can leverage simplified decision boundaries. **Logistic Regression** maintained high accuracy (0.934) with strong precision for the majority class (0.94), although it showed limited effectiveness for the minority class, with a minority recall of just 0.06 and an F1-score of 0.11. The model's ROC AUC of 0.851 suggests reasonable overall performance but highlights some limitations in distinguishing the minority class accurately.

For ensemble models, **Random Forest** and **Gradient Boosting** both performed relatively well, achieving accuracies of 0.930 and 0.931, respectively. Random Forest displayed balanced precision for both classes and a minority recall of 0.13, while Gradient Boosting showed slightly better sensitivity with a minority recall of 0.18 and an ROC AUC of 0.847. **AdaBoost** and **CatBoost** also adapted well, with AdaBoost reaching an accuracy of 0.932 and CatBoost 0.928, each showing moderate precision and recall for the minority class, suggesting that these models could leverage the neighborhood-based undersampling effectively.

**Support Vector Machine (SVM)** achieved an accuracy of 0.936 and an ROC AUC of 0.733, performing better in terms of minority precision (0.38) than recall (0.08). This indicates that SVM, although capable of creating clearer boundaries with All KNN, struggled to capture enough of the minority class effectively. Similarly, **k-Nearest Neighbors (k-NN)** achieved an accuracy of 0.933, with a minority recall of 0.19, suggesting that it can leverage the nearest neighbor approach well with this technique. However, its ROC AUC of 0.676 indicates limited capability in identifying minority instances across the dataset.

**Naive Bayes** and **Quadratic Discriminant Analysis (QDA)** displayed balanced but lower overall performance, with accuracies of 0.839 and 0.844, respectively, and ROC AUC scores around 0.825. Both models showed reasonable minority recall (0.55 for Naive Bayes and 0.53 for QDA), highlighting their ability to benefit from All KNN undersampling by maintaining balanced representation across classes. **Linear Discriminant Analysis (LDA)**, on the other hand, achieved high accuracy (0.923) and displayed balanced metrics with a minority recall of 0.34 and a minority precision of 0.36, suggesting strong adaptability to this undersampling method.

**Extra Trees** achieved moderate performance with an accuracy of 0.915, but its minority recall was limited at 0.15, and its ROC AUC of 0.773 reflected this limitation. **Decision Tree** struggled more, with an accuracy of 0.882 and a lower ROC AUC of 0.590, indicating that single-tree models may not be as suited to neighborhood-based undersampling.

In summary, **All KNN undersampling** worked effectively with models that could benefit from simplified boundaries, such as **Logistic Regression**, **Random Forest**, and **Gradient Boosting**, which maintained high accuracies and moderate performance across classes. Ensemble models like **AdaBoost** and **CatBoost** also showed balanced outcomes, indicating good adaptability. However, SVM and k-NN displayed limitations in capturing minority recall effectively. Probabilistic models such as **Naive Bayes**, **QDA**, and **LDA** performed well overall, demonstrating that models with neighborhood-focused strategies can leverage All KNN’s data simplification effectively.

#### Condensed Nearest Neighbor (CNN):
| Model                           | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-score (0) | F1-score (1) | ROC AUC Score |
|---------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression             | 0.939    | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.846         |
| Random Forest                   | 0.937    | 0.94          | 0.40          | 0.99       | 0.06       | 0.97         | 0.11         | 0.767         |
| SVM                             | 0.939    | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.364         |
| Gradient Boosting               | 0.906    | 0.94          | 0.10          | 0.96       | 0.06       | 0.95         | 0.08         | 0.765         |
| AdaBoost                        | 0.921    | 0.94          | 0.17          | 0.97       | 0.08       | 0.96         | 0.11         | 0.740         |
| XGBoost                         | 0.899    | 0.95          | 0.22          | 0.94       | 0.26       | 0.95         | 0.24         | 0.716         |
| LightGBM                        | 0.913    | 0.95          | 0.25          | 0.96       | 0.21       | 0.95         | 0.23         | 0.739         |
| CatBoost                        | 0.932    | 0.94          | 0.25          | 0.99       | 0.06       | 0.96         | 0.10         | 0.799         |
| k-Nearest Neighbors             | 0.931    | 0.94          | 0.20          | 0.99       | 0.05       | 0.96         | 0.08         | 0.641         |
| Decision Tree                   | 0.808    | 0.94          | 0.09          | 0.85       | 0.23       | 0.89         | 0.12         | 0.536         |
| Naive Bayes                     | 0.913    | 0.95          | 0.24          | 0.96       | 0.19       | 0.95         | 0.21         | 0.844         |
| Linear Discriminant Analysis    | 0.938    | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.844         |
| Quadratic Discriminant Analysis | 0.921    | 0.95          | 0.27          | 0.97       | 0.18       | 0.96         | 0.21         | 0.838         |
| Extra Trees                     | 0.930    | 0.94          | 0.25          | 0.98       | 0.08       | 0.96         | 0.12         | 0.750         |

The **Condensed Nearest Neighbor (CNN)** undersampling technique, which focuses on retaining a minimal set of samples from the majority class, showed mixed performance across different models. **Logistic Regression** maintained a high accuracy (0.939) with excellent precision for the majority class (0.94) but achieved zero recall for the minority class, highlighting a limitation in detecting minority samples. This outcome, with a ROC AUC of 0.846, suggests that Logistic Regression performs well for majority class predictions but struggles with the minority class in this undersampled dataset.

For ensemble models, **Random Forest** achieved a high accuracy of 0.937, demonstrating its robustness even with reduced data, although minority class recall was low at 0.06. This result reflects Random Forest’s reliance on richer data, which CNN undersampling diminishes. **Gradient Boosting** and **AdaBoost** followed a similar pattern with accuracies of 0.906 and 0.921, respectively. Despite strong performance metrics for the majority class, both models struggled with minority class recall, indicating that CNN undersampling may not offer enough sample diversity for these models to generalize well across classes.

**XGBoost** and **LightGBM** exhibited reasonable adaptability, achieving accuracies of 0.899 and 0.913, respectively, with moderate ROC AUC scores of 0.716 and 0.739. These models showed slightly better sensitivity to the minority class than other ensemble models, suggesting some capacity to learn from condensed samples. **CatBoost** performed particularly well with an accuracy of 0.932 and a ROC AUC score of 0.799, highlighting its adaptability even with reduced data complexity, though minority recall remained low at 0.06.

Among simpler models, **Support Vector Machine (SVM)** achieved a high accuracy of 0.939 but struggled to capture the minority class, resulting in zero recall for that class and a low ROC AUC of 0.364. This pattern indicates that SVM’s reliance on boundary definitions may be challenged by the minimal sample selection of CNN. **k-Nearest Neighbors (k-NN)** showed similar results with a high accuracy (0.931) but very low minority recall (0.05), suggesting that nearest-neighbor models require more data variety to perform optimally.

Probabilistic models like **Naive Bayes** and **Quadratic Discriminant Analysis (QDA)** performed moderately well, with Naive Bayes achieving an accuracy of 0.913 and a balanced ROC AUC score of 0.844. QDA showed a minority class recall of 0.18 and a strong accuracy of 0.921, indicating that these models were better able to adjust to CNN undersampling than other classifiers. **Linear Discriminant Analysis (LDA)**, however, showed zero recall for the minority class, similar to Logistic Regression, due to the limited sample diversity provided by CNN.

**Decision Tree** struggled with CNN undersampling, achieving a lower accuracy of 0.808 and an ROC AUC score of 0.536, indicating limited adaptability when deprived of richer majority data. **Extra Trees** performed somewhat better with an accuracy of 0.930, though minority recall remained low, reflecting the tree-based methods' limitations under condensed sampling.

In summary, **CNN undersampling** favors models with a preference for simple majority class predictions, such as **Logistic Regression**, **Random Forest**, and **CatBoost**, each maintaining high accuracy with moderate minority performance. However, SVM and k-NN struggled to capture the minority class due to limited boundary data. Probabilistic models like **Naive Bayes** and **QDA** showed balanced performance, indicating adaptability to CNN, while tree-based models showed varying degrees of success depending on their reliance on data diversity.

#### One-Sided Selection (OSS):
| Model                           | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-score (0) | F1-score (1) | ROC AUC Score |
|---------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression             | 0.939    | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.851         |
| Random Forest                   | 0.938    | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.825         |
| SVM                             | 0.939    | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.630         |
| Gradient Boosting               | 0.939    | 0.94          | 0.50          | 1.00       | 0.02       | 0.97         | 0.03         | 0.840         |
| AdaBoost                        | 0.938    | 0.94          | 0.40          | 1.00       | 0.03       | 0.97         | 0.06         | 0.796         |
| XGBoost                         | 0.934    | 0.94          | 0.35          | 0.99       | 0.10       | 0.97         | 0.15         | 0.812         |
| LightGBM                        | 0.936    | 0.94          | 0.38          | 0.99       | 0.08       | 0.97         | 0.13         | 0.819         |
| CatBoost                        | 0.939    | 0.94          | 0.50          | 1.00       | 0.05       | 0.97         | 0.09         | 0.820         |
| k-Nearest Neighbors             | 0.940    | 0.94          | 1.00          | 1.00       | 0.02       | 0.97         | 0.03         | 0.659         |
| Decision Tree                   | 0.902    | 0.94          | 0.15          | 0.95       | 0.13       | 0.95         | 0.14         | 0.541         |
| Naive Bayes                     | 0.864    | 0.96          | 0.21          | 0.89       | 0.47       | 0.92         | 0.29         | 0.828         |
| Linear Discriminant Analysis    | 0.932    | 0.94          | 0.25          | 0.99       | 0.06       | 0.96         | 0.10         | 0.841         |
| Quadratic Discriminant Analysis | 0.876    | 0.96          | 0.24          | 0.90       | 0.48       | 0.93         | 0.32         | 0.830         |
| Extra Trees                     | 0.934    | 0.94          | 0.22          | 0.99       | 0.03       | 0.97         | 0.06         | 0.760         |

The **One-Sided Selection (OSS)** undersampling technique, which focuses on removing redundant majority class samples while retaining significant minority samples, demonstrated varied performance across different models. Logistic Regression maintained a high accuracy of 0.939 with excellent precision for the majority class (0.94), but it achieved zero recall for the minority class, highlighting its inability to detect minority samples. This outcome, with a ROC AUC score of 0.851, suggests that while Logistic Regression excels in majority class predictions, it struggles with minority class generalization under OSS.

For ensemble models, **Random Forest** achieved an accuracy of 0.938, demonstrating robustness to reduced majority class data. However, minority class recall was zero, emphasizing its reliance on richer datasets. **Gradient Boosting** and **CatBoost** exhibited slightly better adaptability, achieving accuracies of 0.939 and 0.939, respectively. Gradient Boosting recorded a minority class precision of 0.50, while CatBoost had a precision of 0.50 and a recall of 0.05, reflecting a modest improvement in capturing minority samples. Their ROC AUC scores of 0.840 and 0.820 indicate strong performance overall, although OSS still limits their minority class detection. XGBoost and LightGBM followed a similar trend, with accuracies of 0.934 and 0.936, respectively, but struggled with minority class recall (0.10 for XGBoost and 0.08 for LightGBM), showing limited sensitivity despite their adaptability.

Among simpler models, Support Vector Machine (SVM) achieved an accuracy of 0.939 but exhibited zero recall for the minority class, resulting in a low ROC AUC score of 0.630. This highlights the challenge SVM faces in defining boundaries with minimal minority samples. k-Nearest Neighbors (k-NN) performed slightly better, with a minority class precision of 1.00 but recall at only 0.02, indicating its dependence on sample variety to optimize performance.

Probabilistic models like Naive Bayes and Quadratic Discriminant Analysis (QDA) showed balanced performance, with Naive Bayes achieving an accuracy of 0.864 and a minority class recall of 0.47. Its ROC AUC score of 0.828 reflects adaptability to OSS. QDA achieved similar results, with a minority class recall of 0.48 and a balanced ROC AUC score of 0.830. These results indicate that probabilistic models can adjust better to OSS by leveraging class probabilities effectively. Linear Discriminant Analysis (LDA), however, achieved zero recall for the minority class, similar to Logistic Regression, due to OSS's limited sample diversity.

Tree-based models like Decision Tree and Extra Trees struggled under OSS. Decision Tree had a lower accuracy of 0.902 and a minority class recall of 0.13, highlighting its difficulty adapting to reduced datasets. Extra Trees performed marginally better, with an accuracy of 0.934 and a ROC AUC score of 0.760, but its minority class recall remained low (0.03), reflecting a common limitation among tree-based methods when working with undersampled data.

In summary, OSS favors models that excel in majority class predictions, such as Logistic Regression and Random Forest, but limits their ability to generalize for minority class detection. Ensemble models like Gradient Boosting, CatBoost, and XGBoost show moderate adaptability, though their sensitivity to the minority class remains low. Probabilistic models like Naive Bayes and QDA demonstrate better balance across classes, reflecting greater flexibility under OSS, while simpler and boundary-focused models like SVM and k-NN struggle due to limited sample diversity. Tree-based models reveal mixed success, with performance heavily influenced by their reliance on data variety. 


#### Neighborhood Cleaning Rule (NCR):
| Model                           | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-score (0) | F1-score (1) | ROC AUC Score |
|---------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression             | 0.938    | 0.94          | 0.33          | 1.00       | 0.02       | 0.97         | 0.03         | 0.853         |
| Random Forest                   | 0.938    | 0.94          | 0.44          | 0.99       | 0.06       | 0.97         | 0.11         | 0.809         |
| SVM                             | 0.937    | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.703         |
| Gradient Boosting               | 0.934    | 0.94          | 0.22          | 0.99       | 0.03       | 0.97         | 0.06         | 0.846         |
| AdaBoost                        | 0.936    | 0.94          | 0.36          | 0.99       | 0.06       | 0.97         | 0.11         | 0.831         |
| XGBoost                         | 0.932    | 0.95          | 0.36          | 0.98       | 0.16       | 0.96         | 0.22         | 0.822         |
| LightGBM                        | 0.931    | 0.94          | 0.30          | 0.98       | 0.11       | 0.96         | 0.16         | 0.833         |
| CatBoost                        | 0.933    | 0.94          | 0.29          | 0.99       | 0.06       | 0.97         | 0.11         | 0.829         |
| k-Nearest Neighbors             | 0.934    | 0.95          | 0.37          | 0.99       | 0.11       | 0.97         | 0.17         | 0.680         |
| Decision Tree                   | 0.894    | 0.95          | 0.18          | 0.94       | 0.21       | 0.94         | 0.19         | 0.574         |
| Naive Bayes                     | 0.850    | 0.97          | 0.21          | 0.87       | 0.52       | 0.92         | 0.29         | 0.826         |
| Linear Discriminant Analysis    | 0.928    | 0.95          | 0.37          | 0.97       | 0.27       | 0.96         | 0.31         | 0.838         |
| Quadratic Discriminant Analysis | 0.858    | 0.97          | 0.22          | 0.88       | 0.53       | 0.92         | 0.31         | 0.827         |
| Extra Trees                     | 0.928    | 0.94          | 0.25          | 0.98       | 0.10       | 0.96         | 0.14         | 0.797         |

The **Neighborhood Cleaning Rule (NCR)** undersampling technique, which aims to improve boundary clarity by removing noisy samples, yielded strong performance across multiple models, especially in terms of majority class detection. **Logistic Regression** achieved a high accuracy of 0.938 and an ROC AUC score of 0.853, indicating effective separation of classes, although it had limited success with the minority class, as reflected by its low recall and F1-score for that class.

**Random Forest** showed slightly improved minority precision at 0.44 compared to Logistic Regression, with an accuracy of 0.938 and an ROC AUC score of 0.809. This performance reflects NCR’s benefit for ensemble models in maintaining high precision without sacrificing accuracy. **Gradient Boosting** and **AdaBoost** also performed well, achieving accuracies of 0.934 and 0.936, respectively. However, they displayed limited minority recall, similar to Logistic Regression, which reduced their F1-scores for the minority class.

**XGBoost** and **LightGBM** demonstrated balanced performance, with ROC AUC scores of 0.822 and 0.833, respectively, and improved minority recall compared to other models. **CatBoost** also performed well with an accuracy of 0.933, maintaining high precision while moderately improving recall for the minority class, which suggests it can adapt to the sample refinement introduced by NCR.

**Support Vector Machine (SVM)** continued to struggle with minority class detection, achieving a zero recall, despite maintaining high accuracy and precision for the majority class. **k-Nearest Neighbors (k-NN)**, however, showed better adaptability to NCR, with a high accuracy of 0.934 and improved minority class recall, albeit with a relatively lower ROC AUC score of 0.680. This performance reflects the model's moderate ability to leverage NCR's cleaned boundaries.

Among the probabilistic models, **Naive Bayes** and **Quadratic Discriminant Analysis (QDA)** showed balanced results, with Naive Bayes achieving an accuracy of 0.850 and ROC AUC of 0.826, and QDA achieving 0.858 and 0.827, respectively. Both models displayed decent recall and precision for the minority class, suggesting NCR benefits models that utilize probability distributions for class distinction.

**Linear Discriminant Analysis (LDA)** also benefitted from NCR, achieving a balanced performance with an accuracy of 0.928 and ROC AUC of 0.838, showing its effectiveness in handling both majority and minority classes. **Decision Tree** and **Extra Trees** showed more modest gains, with Decision Tree achieving an accuracy of 0.894 and Extra Trees reaching 0.928, indicating NCR’s potential limitations in optimizing tree-based models for minority class detection.

In summary, **NCR undersampling** helped enhance boundary clarity, favoring models like **Logistic Regression, Random Forest, and CatBoost**, which showed strong majority class performance and moderate improvements in minority class recall. **Naive Bayes, QDA, and LDA** demonstrated good adaptability to the cleaned sample space. However, **SVM** and **Decision Tree** struggled to capture minority instances effectively, suggesting NCR may be less beneficial for models requiring finer data structure details to improve minority class performance.

#### Instance Hardness Threshold:
| Model                              | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-score (0) | F1-score (1) | ROC AUC Score |
|------------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression                | 0.247    | 1.00          | 0.07          | 0.20       | 1.00       | 0.33         | 0.14         | 0.730         |
| Random Forest                      | 0.229    | 1.00          | 0.07          | 0.18       | 1.00       | 0.30         | 0.14         | 0.682         |
| SVM                                | 0.242    | 1.00          | 0.07          | 0.19       | 1.00       | 0.32         | 0.14         | 0.672         |
| Gradient Boosting                  | 0.209    | 1.00          | 0.07          | 0.16       | 1.00       | 0.27         | 0.13         | 0.627         |
| AdaBoost                           | 0.195    | 1.00          | 0.07          | 0.14       | 1.00       | 0.25         | 0.13         | 0.595         |
| XGBoost                            | 0.321    | 1.00          | 0.08          | 0.28       | 1.00       | 0.43         | 0.15         | 0.677         |
| LightGBM                           | 0.204    | 1.00          | 0.07          | 0.15       | 1.00       | 0.26         | 0.13         | 0.639         |
| CatBoost                           | 0.205    | 1.00          | 0.07          | 0.15       | 1.00       | 0.27         | 0.13         | 0.730         |
| k-Nearest Neighbors                | 0.276    | 1.00          | 0.08          | 0.23       | 1.00       | 0.37         | 0.14         | 0.641         |
| Decision Tree                      | 0.196    | 1.00          | 0.07          | 0.14       | 1.00       | 0.25         | 0.13         | 0.572         |
| Naive Bayes                        | 0.205    | 1.00          | 0.07          | 0.15       | 1.00       | 0.27         | 0.13         | 0.577         |
| Linear Discriminant Analysis       | 0.320    | 1.00          | 0.08          | 0.28       | 1.00       | 0.43         | 0.15         | 0.792         |
| Quadratic Discriminant Analysis    | 0.101    | 1.00          | 0.06          | 0.04       | 1.00       | 0.08         | 0.12         | 0.524         |
| Extra Trees                        | 0.211    | 1.00          | 0.07          | 0.16       | 1.00       | 0.28         | 0.13         | 0.722         |

The **Instance Hardness Threshold (IHT)** undersampling technique, which removes instances deemed "hard" to classify to reduce noise, significantly affected model performance. Notably, **Logistic Regression** achieved a low accuracy of 0.247 but showed high precision for the majority class (1.00) and a modest ROC AUC of 0.730. This result suggests that while the model correctly classified majority instances, it struggled with the minority class, yielding low F1-scores for it.

**Random Forest** and **SVM** performed similarly, with accuracies of 0.229 and 0.242, respectively, while maintaining perfect precision for the majority class but poor recall for the minority class. This indicates that these models were unable to generalize well to the IHT-filtered dataset, likely due to an over-reliance on the majority class with minimal improvement in distinguishing minority samples.

**Gradient Boosting**, **AdaBoost**, and **LightGBM** also performed poorly, with low accuracies around 0.209, 0.195, and 0.204, respectively, highlighting their struggle with IHT’s removal of difficult instances. These ensemble models showed limited ability to handle the reduced dataset effectively, reflected in low minority recall and F1-scores for both classes.

**XGBoost** and **k-Nearest Neighbors (k-NN)** slightly outperformed other models, achieving higher accuracies of 0.321 and 0.276, respectively. XGBoost's ROC AUC of 0.677 and k-NN's of 0.641 demonstrate that they managed marginally better balance in classifying both classes under IHT. However, the improvement was minor, and the models still exhibited poor recall for the minority class.

Probabilistic models like **Naive Bayes** and **Quadratic Discriminant Analysis (QDA)** struggled considerably with IHT, with QDA achieving an accuracy as low as 0.101. Both models managed perfect precision for the majority class but failed to classify the minority class effectively. The low ROC AUC of 0.524 for QDA reflects the severe limitations of these models in leveraging the data structure created by IHT.

**Linear Discriminant Analysis (LDA)** fared slightly better, reaching an accuracy of 0.320 and a ROC AUC score of 0.792, indicating a more balanced performance compared to other models. However, minority class recall remained low, limiting its effectiveness in real-world applications.

Among tree-based models, **Decision Tree** and **Extra Trees** showed minimal improvements with IHT, achieving accuracies of 0.196 and 0.211, respectively, along with low F1-scores and minority recall. This performance implies that tree-based models also struggled to adapt to IHT’s undersampling, likely due to the lack of nuanced boundary information in the reduced dataset.

In summary, **Instance Hardness Threshold undersampling** significantly impacted model performance, generally favoring the majority class at the expense of the minority class. Models like **Logistic Regression, Random Forest, and SVM** maintained high precision for the majority class but failed to classify the minority class accurately. Ensemble models like **XGBoost** and **k-NN** displayed slightly better balance but remained limited in minority detection. Probabilistic models, including **Naive Bayes** and **QDA**, were particularly disadvantaged. Overall, IHT appears to reduce class overlap but at a cost to minority class recall, making it less suitable for models that rely on balanced boundary information.

# Conclusion

This study investigated the performance of various machine learning models on imbalanced data using different undersampling techniques, including Cluster Centroids, Random Undersampling, NearMiss variations, Tomek Links, Edited Nearest Neighbors (ENN), Repeated Edited Nearest Neighbors (RENN), All KNN, Condensed Nearest Neighbor (CNN), One-Sided Selection (OSS), Neighborhood Cleaning Rule (NCR), Instance Hardness Threshold, and without any sampling adjustments (baseline imbalanced data). The models evaluated include Logistic Regression, Support Vector Machine (SVM), Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, k-Nearest Neighbors (K-NN), Decision Tree, Naive Bayes, Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), and Extra Trees. 

Results indicated that sampling techniques substantially impacted model performance, particularly for minority class detection. Baseline results showed high accuracy across models due to their focus on the majority class, but they failed to capture minority instances effectively, as reflected by low precision and recall scores for the minority class. Cluster Centroids and Random Undersampling produced the best results among the sampling techniques, especially for simpler linear models and probabilistic models like Naive Bayes, which were more resilient to simplified data boundaries. Conversely, tree-based and ensemble models, such as Random Forest and Gradient Boosting, were sensitive to sampling approaches that heavily altered the dataset distribution. 

Overall, Cluster Centroids and Random Undersampling were the most effective techniques for improving minority class detection across models, while probabilistic models like Naive Bayes, LDA, and QDA showed a better balance in class performance under severe data reduction techniques. Ensemble models showed slight improvements in minority detection with sampling but struggled with more aggressive undersampling techniques.

# Discussion

The study highlights the importance of choosing an appropriate undersampling technique tailored to both the dataset and model. Cluster Centroids and Random Undersampling were effective in simplifying data structure, benefiting models such as Logistic Regression, SVM, and Naive Bayes by reducing class imbalance while retaining critical information. However, more complex models like tree-based and ensemble methods struggled under these methods, likely due to their reliance on granular decision boundaries that were obscured by oversimplification. These findings suggest that simpler models may adapt better to undersampling techniques that focus on creating centroids or random samples, while ensemble models may require techniques that retain data complexity.

The study also reveals the limitations of popular models like Random Forest and XGBoost in handling imbalanced data without sampling adjustments. Despite their robust performance in balanced datasets, these models often overfit to the majority class in imbalanced scenarios, as seen in their low recall scores for the minority class. The performance of probabilistic models, such as Naive Bayes and QDA, underscores their suitability in imbalanced contexts due to their relative insensitivity to boundary shifts caused by undersampling. 

Several undersampling techniques like Tomek Links and CNN preserved minority instances at the expense of reducing overall dataset size. Techniques like NCR, OSS, and All KNN performed well but still struggled with minority class precision. The need for more advanced techniques becomes clear as simple undersampling methods compromise data complexity, leading to lower performance in minority class detection. These results highlight that no single undersampling technique is universally optimal and that an effective undersampling strategy should consider the specific model’s data requirements and behavior.

# Future Work

Future work could explore advanced undersampling and hybrid techniques, such as SMOTE combined with ENN or RENN, which combine both oversampling of the minority class and cleaning of the majority class. This may provide better minority class representation without over-simplifying data boundaries. Additionally, the use of **ensemble-based resampling methods** that create multiple balanced datasets and aggregate predictions could improve minority class detection, especially for tree-based models.

Research could also examine model-specific tuning techniques for imbalanced datasets, such as cost-sensitive learning for ensemble models, where misclassification of minority instances is penalized more heavily. This may be particularly useful for complex models like Random Forest, Gradient Boosting, and CatBoost, which require more data complexity to perform optimally. 

Moreover, future studies could incorporate **feature selection methods** in conjunction with undersampling to further refine model performance by emphasizing the most predictive features and reducing the noise introduced by undersampling. Lastly, expanding the scope of testing to include more recent models, such as transformers for tabular data, might yield promising results in imbalanced classification scenarios, providing insights into novel approaches for addressing class imbalance in machine learning.

