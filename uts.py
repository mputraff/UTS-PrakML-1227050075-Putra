# %%
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree

# %%
# Load dataset citrus.csv
citrus = pd.read_csv("citrus.csv")  # sesuaikan path jika perlu
print(citrus.head())

# %%
# Visualisasi pairplot untuk melihat sebaran data
sns.pairplot(citrus, hue='name', palette='Set2')

# %%
# Split features dan label
X = citrus.drop('name', axis=1)
y = citrus['name']

# Split ke training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# %%
# Bangun model decision tree
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %%
# Evaluasi model
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# %%
# Visualisasi pohon keputusan
plt.figure(figsize=(20, 10))
tree.plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
plt.show()

# %%
# Prediksi data baru
citrus_test_data = {
    'diameter': 4.0,
    'weight': 90.0,
    'red': 160,
    'green': 80,
    'blue': 5
}

# Buat DataFrame dari data input
prediction_input_df = pd.DataFrame([citrus_test_data])
prediction = model.predict(prediction_input_df[X.columns])
print("Prediksi buah:", prediction[0])