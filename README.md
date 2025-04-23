# UTS-PrakML-1227050075-Putra
Pengumpulan dan Persiapan data
Dataset yang dipakai untuk pembuatan model Decision Tree berasal dari data klasifikasi apakah buah tersebut merupakan jeruk atau anggur. Detail lengkap dataset dapat diperoleh melalui link berikut ini (https://www.kaggle.com/datasets/joshmcadams/oranges-vs-grapefruit)

Untuk dapat melihat dataset tersebut anda dapat mencoba dengan menggunakan langkah-langkah berikut ini:

1. Buat sebuah project baru dan lakukan load library ini :
# %%
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

![image](https://github.com/user-attachments/assets/4da5e9eb-f000-489d-937b-a66f29914ed2)

Melakukan pembagian data (Training dan Testing)
1. Sebelum melakukan pembagian data, kita dapat menampilkan grafik dari data tersebut menggunakan perintah berikut ini :
# %%
# Visualisasi pairplot untuk melihat sebaran data
sns.pairplot(citrus, hue='name', palette='Set2')

![image](https://github.com/user-attachments/assets/64a740a9-2f2e-4e3d-a7d4-12868f641f94)

2. Saya melakukan pembagian data dengan tujuan menggunakan data training yang dipergunakan untuk membuat model. Pembagian data yang digunakan yaitu 70% data training dan 30% data testing. Adapun tahapannya adalah dengan menambahkan kode program berikut:
3. 
# %%
# Split features dan label
X = citrus.drop('name', axis=1)
y = citrus['name']

# Split ke training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
3. Anda dapat menambahkan perintah print(len(x_train)) untuk melihat jumlah dari data training yang digunakan.

Membuat Model Klasifikasi

# %%
# Bangun model decision tree
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

Membuat Report Hasil Klasifikasi
print(classification_report(y_test, y_pred))
Berikut hasil dari classification report yang dihasilkan:

![image](https://github.com/user-attachments/assets/cf4101ef-d56c-47c7-ac50-ce1f40484c72)

Melakukan Evaluasi
Selain itu untuk melakukan evaluasi terhadap model, kita dapat memperlihatkan confusion matrix yang dihasilkan dengan melakukan visualisasi dan tambahkan kode berikut ini.

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

![image](https://github.com/user-attachments/assets/5f073269-2a68-40d0-b461-92778618ef6a)

Menampilkan visualisasi Tree (Decision Tree)

Bagian ini untuk menunjukkan bagaimana tree/ pohon dalam melakukan klasifikasi, anda dapat menggunakan kode program ini.
# %%
# Visualisasi pohon keputusan
plt.figure(figsize=(20, 10))
tree.plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
plt.show()

![image](https://github.com/user-attachments/assets/894c9b01-646c-442e-93a5-e7baec748b07)


Prediksi dari data tersebut
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

![image](https://github.com/user-attachments/assets/491fa64b-4bc5-4895-b94d-98c47afe4aa3)



