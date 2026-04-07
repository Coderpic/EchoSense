import pickle
from src.dataset import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

x,y = load_dataset()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

# Random Forest Model
model = RandomForestClassifier()
model.fit(x_train,y_train)

# save model
pickle.dump(model,open("models/model.pkl","wb"))

print('Model trained and saved successfully')