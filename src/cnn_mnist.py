import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307),(0.3081)) 

])

train_dataset = datasets.MNIST(
    root="./datam",
    train= True,
    download= True,
    transform= transform
)

train_loader = DataLoader(train_dataset,
                          shuffle = True,
                          batch_size = batch_size)

test_dataset = datasets.MNIST(
    root="./datam",
    train=False,
    download=True,
    transform=transform
)

test_loder = DataLoader(test_dataset,
                        shuffle = True,
                        batch_size = batch_size)

'''
CLASS torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
'''

'''
CLASS torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, 
dilation=1, return_indices=False, ceil_mode=False)
'''
class CnnNet(torch.nn.Module):
    def __init__(self):
        super(CnnNet,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=10,kernel_size=3)
        self.conv2 = torch.nn.Conv2d(in_channels=10,out_channels=20,kernel_size=3)
        self.conv3 = torch.nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3)
        self.pooling1 = torch.nn.MaxPool2d(kernel_size=2)
        self.pooling2 = torch.nn.MaxPool2d(kernel_size=2)
        self.pooling3 = torch.nn.MaxPool2d(kernel_size=2)
        self.linear1 = torch.nn.Linear(40,32) 
        self.linear2 = torch.nn.Linear(32, 10)

    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pooling1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pooling2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pooling3(x)
        x = x.view(x.size(0), -1)  
        x = self.linear1(x)
        x = self.linear2(x)
        return x 





model = CnnNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum= 0.5)


def train(epoch):
    total = 0
    running_loss = 0.0
    train_loss = 0.0 
    accuracy = 0 
    for batch_id, data in enumerate(train_loader,0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)

        _, predicted = torch.max(outputs.data, dim=1)
        accuracy += (predicted == target).sum().item()
        total += target.size(0)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_loss = running_loss
        if batch_id % 300 == 299:
            print('[%d, %5d] loss: %.3f' %(epoch+1, batch_id+1, running_loss / 300))
            running_loss = 0.0
    print('第 %d epoch的 Accuracy on train set: %d %%, Loss on train set: %f' % (epoch + 1, 100 * accuracy / total, train_loss))

    return 1.0 * accuracy / total, train_loss


def validation(epoch):
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for data in test_loder:
            images, target = data
            images, target = images.to(device), target.to(device)
            outputs = model(images)
            loss = criterion(outputs, target)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('第 %d epoch的 Accuracy on validation set: %d %%, Loss on validation set: %f' %(epoch+1,100*correct / total, val_loss))


    return 1.0 * correct / total, val_loss



def draw_in_one(list,epoch):

    x_axix = [x for x in range(1, epoch+1)]
    train_acc = list[0]
    train_loss = list[1]
    val_acc = list[2]
    val_loss = list[3]

    plt.title('Result Analysis')
    plt.plot(x_axix, train_acc, color='green', label='training accuracy')
    plt.plot(x_axix, train_loss, color='red', label='training loss')
    plt.plot(x_axix, val_acc, color='skyblue', label='val accuracy')
    plt.plot(x_axix, val_loss, color='blue', label='val loss')
    plt.legend() 
    plt.xlabel('epoch times')
    plt.ylabel('rate')
    plt.show()
if __name__ == '__main__':

    train_loss = []
    train_acc = []

    val_loss = []
    val_acc = []
    epoches = 10
    list = []
    for epoch in range(epoches):
        acc1, loss1 = train(epoch)

        train_loss.append(loss1)
        train_acc.append(acc1)

        acc2, loss2 = validation(epoch)

        val_loss.append(loss2)
        val_acc.append(acc2)
    torch.save(model.state_dict(), 'cnn_model.ckpt')
    list.append(train_acc)
    list.append(train_loss)
    list.append(val_acc)
    list.append(val_loss)
    draw_in_one(list, epoches)