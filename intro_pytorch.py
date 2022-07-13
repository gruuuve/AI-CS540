import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from collections import Counter
# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    # transform to be used with the sets
    custom_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3801,))
        ])

    train_set = datasets.MNIST('./data', train=True, download=True, transform = custom_transform)    
    test_set = datasets.MNIST('./data', train=False, transform=custom_transform)

    if training == True: # use training set
        loader = torch.utils.data.DataLoader(train_set, batch_size=50)
    else:
        loader = torch.utils.data.DataLoader(test_set, batch_size=50, shuffle=False)
    
    return loader



def build_model():
    """
    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    # create the model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64,10),
        )

    return model


def train_model(model, train_loader, criterion, T):
    """
    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    # train mode
    model.train()

    # optimization algorithm
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(T):
        run_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad() # zero param gradients
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            run_loss += loss.item()
            
        # accuracy calcs
        valid = 0
        total = 0
        with torch.no_grad():
            for data in train_loader:
                inputs, labels = data
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                valid += (predicted == labels).sum().item()
        # end of epoch
        print("Train Epoch: {:d}   Accuracy: {:d}/{:d}({:.2f}%)   Loss: {:.4f}".format(epoch,
            valid, total, 100*valid/total, run_loss/total))

def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    correct = 0
    total = 0
    run_loss = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            run_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if show_loss == True:
        print("Average loss: {:.4f}\nAccuracy: {:.2f}%".format(run_loss/total, 100*correct/total))
    else:
        print("Accuracy: {:.2f}%".format(100*correct/total))

def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    model.eval()
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 
        'seven', 'eight', 'nine']
    #test_tensor = torch.cat(test_images, dim=0).reshape(test_images
    #print(test_tensor.shape)
    #images, labels = test_tensor

    prob_dic = {}
    # obtain logits from model and perform prediction
    logits = model(test_images[index])
    prob = F.softmax(logits, dim=1)
    prob.detach().numpy() # convert to numpy array
    for class_n, val in zip(class_names, prob[0]):
        prob_dic[val.item()] = class_n # place into dict
    counter = Counter(prob_dic)
    top_3 = counter.most_common(3)
    #print(prob_dic)
    print(top_3)
    

if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    model = build_model()
    train_model(model, train_loader, criterion, T = 5)

