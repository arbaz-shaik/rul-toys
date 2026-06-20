from sklearn.datasets import load_iris
import mlflow
import torch
import torch.nn as nn

dataset = load_iris()

X = dataset.data
y = dataset.target

#splitting the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#converting the data to torch tensors
x_test= torch.tensor(X_test, dtype=torch.float32)
y_test= torch.tensor(y_test, dtype=torch.long)

y_train = torch.tensor(y_train, dtype=torch.long)
x_train = torch.tensor(X_train, dtype=torch.float32)





mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("PyTorch_Iris")

for lr in [0.01, 0.001, 0.0001]:
    layer = nn.Linear(4, 3)
    model = nn.Sequential(layer)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    

    with mlflow.start_run():
        mlflow.log_param("lr", lr)
        for epoch in range(100):

            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            mlflow.log_metric("loss", loss.item(), step=epoch)

        #evaluting the model 
        with torch.no_grad():
            outputs = model(x_test)

            predicted = torch.argmax(outputs, dim=1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        #logging the accuracy to mlflow
        mlflow.log_metric("accuracy", accuracy)


