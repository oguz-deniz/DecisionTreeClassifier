# DecisionTreeClassifier
This project showcases my implementation of a decision tree classifier from scratch, providing a fundamental and clear example of how this machine learning algorithm works.
Pure Python Implementation: This decision tree classifier is entirely implemented in Python, ensuring comprehensibility and adaptability.
Scikit-Learn Style API: I've designed the code to follow the intuitive API style of Scikit-Learn, making it user-friendly and compatible with other machine learning tools.
Binary and Multiclass Classification: This implementation supports both binary and multiclass classification tasks.
Performance Evaluation: The model's performance is assessed through metrics such as accuracy, precision, recall, and F1-score, calculated from the confusion matrices for the training and testing datasets.
Confusion Matrix Insights: In the confusion matrix, we observe that out of 30 instances, 29 are correctly classified on the diagonal, indicating only one misclassification.
Robust Training: The model achieves a high training accuracy (typically around 1.0) since it learns from the training dataset it has seen before. However, it maintains good generalization to unseen test data (around 0.96 accuracy).
ROC Curve Analysis: Examining the ROC curve reveals that the model's performance closely approaches the ideal. The model exhibits a high area under the curve (AUC) of approximately 0.97, indicating its strong discriminative power based on true positive and false positive rates.
