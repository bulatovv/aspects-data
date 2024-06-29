import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the data
q = pl.scan_csv('predicts.csv')

# Collect the relevant columns
data = q.collect()
references = data['sentiment'].to_list()
predictions = data['pred'].to_list()

# Calculate the confusion matrix
cm = confusion_matrix(references, predictions)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Sentiment Classes')
plt.show()

