import torch
import torch.nn as nn
from SparseTorchLayers.base import SparseLayer, DynamicSparseLayer, SETLayer, PruningLayer
from data import load_mnist


batch_log_interval = 10
use_cuda = True
use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

layer1 = SETLayer(784, 300, initial_density=0.6)
layer2 = SETLayer(300, 100, initial_density=0.6)
layer3 = SETLayer(100, 10, initial_density=0.8)

# layer1 = nn.Linear(784, 300)
# layer2 = nn.Linear(300, 100)
# layer3 = nn.Linear(100, 10)

# print(layer1.state_dict())
# print(layer2.state_dict())
# print(layer3.state_dict())

model = nn.Sequential(layer1, nn.ReLU(), layer2, nn.ReLU(), layer3, nn.Softmax(dim=1))

optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)
# optimizer = torch.optim.Adagrad(params=model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# fix_optimizer(optimizer)

model.to(device=device)
train_data, test_data, train_loader, test_loader = load_mnist({"batch_size": 64}, "data")


# layer1.update()
# layer2.update()
# layer3.update()

model.train()
for epoch in range(0, 1000):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()

        optimizer.step()

        # if self.conf.save_models == 'train' and self.epoch_ctr % self.conf.save_interval == 0:
        #     self.save(self.save_dir + "E" + str(self.epoch_ctr) + ".pth")

        if batch_idx % batch_log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_data),
                                100. * batch_idx / len(train_loader), loss.item()))

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(layer1.nelement(), layer2.nelement(), layer3.nelement())
    print(layer1.idx[:20])
    layer1.update()
    layer2.update()
    layer3.update()
    print(layer1.nelement(), layer2.nelement(), layer3.nelement())
    print(layer1.idx[:20])

    test_loss /= len(test_data)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_data),
        100. * correct / len(test_data)))

print(layer1.state_dict())


