import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

from model import Net

# データセットをダウンロード
mnist_data = MNIST('~/tmp/mnist', train=True,
                   download=True, transform=transforms.ToTensor())
data_loader = DataLoader(mnist_data, batch_size=4, shuffle=False)

data_iter = iter(data_loader)
images, labels = data_iter.next()

# matplotlibで1つ目のデータを可視化してみる
npimg = images[0].numpy()
npimg = npimg.reshape((28, 28))
plt.imshow(npimg, cmap='gray')
# plt.show()
print('Label:', labels[0])

# 訓練データとテストデータを用意
train_data = MNIST('~/tmp/mnist', train=True, download=True,
                   transform=transforms.ToTensor())
train_loader = DataLoader(mnist_data, batch_size=4, shuffle=True)
test_data = MNIST('~/tmp/mnist', train=False, download=True,
                  transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)


def get_results( test_loader, net ):
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    for data in test_loader:
        inputs, labels = data
        outputs = net(Variable(inputs))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        y_true += list( labels.numpy() )
        y_pred += list( predicted.numpy() )
    print('Accuracy {} / {} = {}'.format(correct.item(), total, correct.item()/total))
    #print('Confusion Matrix :')
    #print('{}'.format( confusion_matrix(y_true, y_pred) ))


for epoch in range(3):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        # Variableに変換
        inputs, labels = Variable(inputs), Variable(labels)
        # 勾配情報をリセット
        optimizer.zero_grad()
        # 順伝播
        outputs = net(inputs)
        # コスト関数を使ってロスを計算する
        loss = criterion(outputs, labels)
        # 逆伝播
        loss.backward()
        # パラメータの更新
        optimizer.step()
        running_loss += loss.data
        if (i+1) % 5000 == 0:
            print('%d %d loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0
    #get_results(test_loader, net)

print('Finished Training')

get_results(test_loader, net)
