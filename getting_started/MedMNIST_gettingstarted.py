'''
MedMNIST getting started
'''

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

'''
----------------------------------------------------------------------------------------------------------------------------------------------------
we first work on 2D images
----------------------------------------------------------------------------------------------------------------------------------------------------
'''

data_flag = 'pathmnist'
# data_flag = 'breastmnist'
download = True

NUM_EPOCHS = 3
BATCH_SIZE = 128
lr = 0.001

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])



'''
First, we read the MedMNIST data, preprocess them and encapsulate them into dataloader form.
'''

# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)

pil_dataset = DataClass(split='train', download=download)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)



print(train_dataset)
print("===================")
print(test_dataset)




# visualization
train_dataset.montage(length=1)
# montage
train_dataset.montage(length=20)



'''
Then, we define a simple model for illustration, 
object function and optimizer that we use to classify.
'''
# define a simple CNN model

class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = Net(in_channels=n_channels, num_classes=n_classes)
copymodel = model





# define loss function and optimizer
if task == "multi-label, binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()
    
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)





'''
Next, we can start to train and evaluate!
'''
# train

for epoch in range(NUM_EPOCHS):
    train_correct = 0
    train_total = 0
    test_correct = 0
    test_total = 0
    
    model.train()
    for inputs, targets in tqdm(train_loader):
        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(inputs)
        
        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()



# evaluation

def test(split):
    model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    
    data_loader = train_loader_at_eval if split == 'train' else test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()
        
        evaluator = Evaluator(data_flag, split)
        metrics = evaluator.evaluate(y_score)
    
        print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))

        
print('==> Evaluating ...')
test('train')
test('test')


    
'''-- welche parameter sind im trainierten model enthalten? --'''
for name, param in model.named_parameters():
    print('name: ', name)
    print(type(param))
    print('param.shape: ', param.shape)
    print('param.requires_grad: ', param.requires_grad)
    print('=====')

'''
https://towardsdatascience.com/everything-you-need-to-know-about-saving-weights-in-pytorch-572651f3f8de
Since all the params in the model have requires_grad = True, it means that all the parameters are learnable and will update on training the model. 
Had it been set to False for a any specific param, that parameter's weight would not update on training the model.
So, requires_grad is the flag that you might want to change when you want to train/freeze a specific set of layers of your model.
'''

help(model) # exit via q
isinstance(model.fc, nn.Module) # fc = fully connected layer

for name, child in model.named_children():
    print('name: ', name)
    print('isinstance({}, nn.Module): '.format(name), isinstance(child, nn.Module))
    print('=====')
# ‘layer1’, ‘layer2’, ..., and ‘fc’ are all the children of model and all of these are nn.Module class objects.


''' 
------------------------------------- WEIGHTS SPEICHERN -------------------------------------
um die weights zu speichern brauchen wir eine state_dict()-Struktur
--> state_dict() works on any nn.Module object and returns all it's immediate children(of class nn.Module).
'''
for key in model.fc.state_dict():       # fc = fully connected layer
    print('key: ', key)                 # model.fc.state_dict() or any nnModule.state_dict() is an ordered dictionary
    param = model.fc.state_dict()[key]  # So iterating over it gives us the keys of the dictionary which can be used to access the parameter tensor
    print('param.shape: ', param.shape) # the parameter tensor is not a nn.Module object, but a simple torch.Tensor with a shape and requires_grad attribute.
    print('param.requires_grad: ', param.requires_grad)
    print('param.shape, param.requires_grad: ', param.shape, param.requires_grad)
    print('isinstance(param, nn.Module) ', isinstance(param, nn.Module))
    print('isinstance(param, nn.Parameter) ', isinstance(param, nn.Parameter))
    print('isinstance(param, torch.Tensor): ', isinstance(param, torch.Tensor))
    print('=====')

'''
So it must be noted that when we save the state_dict() of a 
nn.Module object, the torch.Tensor objects are saved !
This is how we save the state_dict of the entire model.
'''
torch.save(model.state_dict(), 'weights_only.pth')
# This makes a ‘weights_only.pth’ file in the working directory and it holds, 
# in an ordered dictionary, the torch.Tensor objects of all the layers of the model.
'''
weights können für ein externes NN geladen werden:
    model_new = NeuralNet()
    model_new.load_state_dict(torch.load('weights_only.pth'))
'''




'''
------------------------------------- MODEL SPEICHERN -------------------------------------
By entire model, I mean the architecture of the model as well as it's weights.
'''
torch.save(model, 'entire_model.pth')
# This makes a ‘entire_model.pth’ file in the working directory and it contains the model architecture as well as the saved weights.

# loading via:
model_new = torch.load('entire_model.pth')





'''
----------------------------------------------------------------------------------------------------------------------------------------------------
on 3D data
----------------------------------------------------------------------------------------------------------------------------------------------------
'''
data_flag = 'organmnist3d'
download = True

info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

# load the data
train_dataset = DataClass(split='train',  download=download)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)




x, y = train_dataset[0]
print(x.shape, y.shape)



for x, y in train_loader:
    print(x.shape, y.shape)
    break



frames = train_dataset.montage(length=1, save_folder="tmp/")
frames[10]



frames = train_dataset.montage(length=20, save_folder="tmp/")
frames[10]