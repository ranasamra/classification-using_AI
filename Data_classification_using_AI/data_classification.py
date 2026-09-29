
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import classification_report,confusion_matrix


iris =load_iris()
x = iris.data
y = iris.target

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train ,x_test, y_train, y_test =train_test_split(
    x_scaled, y ,test_size=0.20,random_state=42,shuffle=True
)

knn =KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)

error_rates = []
for k in range(1,20):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train,y_train)
    predictions = model.predict(x_test)
    error_rates.append(np.mean(predictions !=y_test))


plt.figure(figsize=(8,4))
plt.plot(range(1,20),error_rates,marker = 'o')
plt.title('error rate vs K value')
plt.xlabel('K value')
plt.ylabel('error rate')
plt.show()


print("confusion matrix :\n",confusion_matrix(y_test,y_pred))
print("\nclassification report(precision,Recall ,F1-f1-score):\n",classification_report(y_test,y_pred))