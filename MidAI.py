from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


File_path = 'D:/dataAI/'
File_name = 'car_data.csv'
df = pd.read_csv(File_path + File_name)

df.drop(columns=['User ID'], inplace = True)

encoders = []
for i in range(0, len(df.columns)-1):
    enc = LabelEncoder()
    df.iloc[:,i] = enc.fit_transform(df.iloc[:, i])
    encoders.append(enc)
    
x =df.iloc[:, 0:3]
y =df['Purchased']
x_train, x_test, y_train, y_test = train_test_split(x,y)

model = DecisionTreeClassifier(criterion='entropy')
model.fit(x,y)

x_pred = ['Male', 34, 147500]
for i in range(0, len(df.columns)-1):
    x_pred[i] = encoders[i].transform([x_pred[i]])
x_pred_adj =np.array(x_pred).reshape(-1,3)
y_pred = model.predict(x_pred_adj)

feature = x.columns.tolist()
Data_class = y.tolist()
plt.figure(figsize=(25,20))
_ = plot_tree(model,
              feature_names= feature,
              class_names = Data_class,
              label='all',
              impurity=True,
              precision=3,
              filled=True,
              rounded=True,
              fontsize=16)
plt.show()

Feature_imp = model.feature_importances_
feature_names = ['Gender','Age','AnnualSalary']
sns.set(rc = {'figure.figsize' : (11.7,8.7)})
sns.barplot(x = Feature_imp, y = feature_names)

print(Feature_imp)
print('Accuracy_train : ' + '{:.2f}'.format(model.score(x_train, y_train)))
print('Accuracy_test : ' + '{:.2f}'.format(model.score(x_test, y_test)))