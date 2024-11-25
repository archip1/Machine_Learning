"""
TODO: Finish and submit your code for logistic regression, neural network, and hyperparameter search.

"""

# got more import libraries from the notebook
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.autograd import Variable
from torch.utils.data import random_split

# for error that wasnt downloading the dataset from stack overflow
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Set random seed for same results (TA suggested)
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Define the Logistic Regression model (reference from the MNIST multiple regression notebook)
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(28*28, 10) # input, classes

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
# (reference from the MNIST multiple regression notebook) - accurcy 92.1%
# FOR PART 1
def logistic_regression(device):
    # TODO: implement logistic regression here

    # Define Parameters
    n_epochs = 20
    batch_size_train = 200
    batch_size_test = 1000

    # 0.01 (91.95) score = 89, 
    # 0.02 (92.1) score = 91, 
    # 0.05 (92.22) score = 92.19, 
    # 0.1 (91.81) score = 88.1
    # 0.08 (91.97) score = 89.7
    # 0.001 (90.07) score = 70.69
    # 0.002 (90.84) score = 78.39
    learning_rate = 0.02

    # momentum = 0.5
    log_interval = 100

    # Load Dataset MNIST
    MNIST_training = torchvision.datasets.MNIST('./data', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

    MNIST_test_set = torchvision.datasets.MNIST('./data', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
    

    val_size = 12000 # Split train dataset (12000 validation - Use the last 12,000 samples of the train set as validation set)
    train_size = len(MNIST_training) - val_size

    # create a training and a validation set
    MNIST_training_set, MNIST_validation_set = random_split(MNIST_training, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(MNIST_training_set,batch_size=batch_size_train, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(MNIST_validation_set,batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(MNIST_test_set,batch_size=batch_size_test, shuffle=True)

    # Model and Optimizer
    multi_linear_model = LinearRegression().to(device)

    # optimizer = optim.Adam(multi_linear_model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(multi_linear_model.parameters(), lr=learning_rate, weight_decay=0.001)  # L2 regularization
    # optimizer = optim.SGD(multi_linear_model.parameters(), lr=0.01) # L1 regularization

    # Cross Entropy Loss
    criterion = nn.CrossEntropyLoss()

    # Training Function
    def myGD_train(epoch, data_loader, model, optimizer, criterion):
    # def myGD_train(epoch, data_loader, model, optimizer, criterion, l1_lambda=0.0): # L1 regularization
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))

    # Evaluation Function
    def eval(data_loader, model, dataset):
        loss = 0
        correct = 0
        with torch.no_grad(): # notice the use of no_grad
            for data, target in data_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                pred = output.data.max(1, keepdim=True)[1] # Max probability index (1)
                correct += pred.eq(target.data.view_as(pred)).sum()
                loss += criterion(output, target).item()
        loss /= len(data_loader.dataset)
        print(dataset+'set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))

    #  Evaluation on validation set
    eval(validation_loader, multi_linear_model, "Validation")

    # Training
    for epoch in range(1, n_epochs + 1):
        myGD_train(epoch, train_loader, multi_linear_model, optimizer, criterion)
        # myGD_train(epoch, train_loader, multi_linear_model, optimizer, criterion, l1_lambda=0.001) # L1 regularization
        eval(validation_loader, multi_linear_model, "Validation")
        
    # Evaluation on test set
    eval(test_loader,multi_linear_model,"Test")

    results = dict(
        model = multi_linear_model,
        test_accuracy = None
    )

    return results

# (reference from the MNIST_NN notebook) - accuracy 42.37%
# FOR PART 2
class FNN(nn.Module):
    def __init__(self, loss_type, num_classes):
        super(FNN, self).__init__()

        self.loss_type = loss_type
        self.num_classes = num_classes

        """add your code here"""
        # Define network layers
        self.fc1 = nn.Linear(32*32*3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):

        """add your code here"""
        # pass through layers
        x = x.view(x.size(0), -1) # flatten the input
        x = torch.tanh(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        output = x
        return output

    def get_loss(self, output, target):
        loss = None

        """add your code here"""
        # compute loss: ce becase FNN_main.py only has that type of loss
        if self.loss_type == "ce":
            loss = F.cross_entropy(output, target)
        else:
            raise ValueError("Wrong loss type")

        return loss

# implement logistic regression and FNN hyper-parameter tuning here - logistic = 91.36, FNN = 41.3
# FOR PART 3
def tune_hyper_parameter(target_metric, device):
    # Define Parameters values
    learning_rates = [0.01, 0.02, 0.05, 0.005]
    weight_decays = [0.001, 0.0001, 0.0005, 0.00001]
    # epoch_values = [10, 15, 20]
    # batch_sizes = [64, 128, 200]

    # Variables to store best parameters and metrics
    best_params = [
        {
            "logistic_regression":
            {
                "learning_rate": None,
                # "betas": None,
                "weight_decay": None
                # "epochs": None
            }
        },
        {
            "FNN":
            {
                "learning_rate": None,
                "betas": None,
                "weight_decay": None
                # "epochs": None
                # "batch_size": None
            }
        }
    ]

    best_metric = [
        {
            "logistic_regression":
            {
                "accuracy": 0.0
                # "loss": 0.0
            }
        },
        {
            "FNN":
            {
                "accuracy": 0.0
                # "loss": 0.0
            }
        }
    ]
    
    # train and evaluate the model copied from the logistic regression removed the loss metric calculations
    def evaluate_model(model, optimizer, train_loader, val_loader, criterion):
        for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # Validation
        correct = 0
        total = 0
        # loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.data.view_as(pred)).sum().item()
                total += target.size(0)
                # loss += criterion(output, target)

        accuracy = correct / total
        # accuracy = correct / len(val_loader.dataset)
        # avg_loss = loss / len(val_loader.dataset)

        # if target_metric == "accuracy":
        #     return accuracy
        # elif target_metric == "loss":
        #     return avg_loss

        return accuracy
    
    # Load Dataset MNIST
    MNIST_training = torchvision.datasets.MNIST('./data', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))]))    

    # Split train dataset
    val_size = 12000
    train_size = len(MNIST_training) - val_size

    # create a training and a validation set
    MNIST_training_set, MNIST_validation_set = random_split(MNIST_training, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(MNIST_training_set, batch_size=200, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(MNIST_validation_set, batch_size=200, shuffle=True)
    
    # Cross Entropy Loss
    criterion = nn.CrossEntropyLoss()

    # Grid search for logistic regression
    for lr in learning_rates:
        for weight_decay in weight_decays:
            # Logistic regression model with the adam optimizer
            logistic_model = LinearRegression().to(device)
            optimizer = optim.Adam(logistic_model.parameters(), lr=lr, weight_decay=weight_decay)

            # for epoch in range(5):
            #     evaluate_model(logistic_model, optimizer, train_loader, validation_loader, criterion)

            # Evaluate
            accuracy = evaluate_model(logistic_model, optimizer, train_loader, validation_loader, criterion)

            # Update if better accuracy
            if accuracy > best_metric[0]["logistic_regression"]["accuracy"]:
                best_metric[0]["logistic_regression"]["accuracy"] = accuracy
                best_params[0]["logistic_regression"]["learning_rate"] = lr
                # best_params[0]["logistic_regression"]["betas"] = beta
                best_params[0]["logistic_regression"]["weight_decay"] = weight_decay
                # best_params[0]["logistic_regression"]["epochs"] = epochs
                print(f"Best Accuracy: ", accuracy)

    print("Done with logistic regression")

    # Grid search for FNN
    learning_rates = [0.001, 0.01]
    weight_decays = [0.0001, 0.001]
    betas_values = [(0.9, 0.999), (0.85, 0.995)]

    # Load Dataset CIFAR
    CIFAR_training = torchvision.datasets.CIFAR10('.', train=True, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    # create a training and a validation set
    CIFAR_train_set, CIFAR_val_set = random_split(CIFAR_training, [40000, 10000])

    train_loader = torch.utils.data.DataLoader(CIFAR_train_set, batch_size=200, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(CIFAR_val_set, batch_size=200, shuffle= False)

    # Grid search for FNN
    for lr in learning_rates:
        for weight_decay in weight_decays:
            for beta in betas_values:
                # FNN model and adam optimizer
                fnn_model = FNN(loss_type="ce", num_classes=10).to(device)
                optimizer = optim.Adam(fnn_model.parameters(), lr=lr, weight_decay=weight_decay, betas=beta)

                for epoch in range(3):
                    evaluate_model(fnn_model, optimizer, train_loader, validation_loader, criterion)

                # Evaluate
                accuracy = evaluate_model(fnn_model, optimizer, train_loader, validation_loader, criterion)

                # Update if better accuracy
                if accuracy > best_metric[1]["FNN"]["accuracy"]:
                    best_metric[1]["FNN"]["accuracy"] = accuracy
                    best_params[1]["FNN"]["learning_rate"] = lr
                    best_params[1]["FNN"]["betas"] = beta
                    best_params[1]["FNN"]["weight_decay"] = weight_decay
                    # best_params[1]["FNN"]["batch_size"] = batch_size
                    # best_params[1]["FNN"]["epochs"] = epochs
                    print(f"Best Accuracy: ", accuracy)
    print("Done with FNN")
    return best_params, best_metric

