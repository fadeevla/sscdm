from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from os import chdir
import sys 
from cd_optim import cd_optim
import random
#torch.set_default_dtype(torch.float16)

sys.path.insert(0, '../sscdm')
from sscdm_wrapper import SSCDM

import time
random.seed(0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class my_dataloader():
    def __init__(self, dataloader = None, sample_indices=[0]):
        self.sample_indices = sample_indices
        if dataloader == None:
            raise('dataloader must be an object of dataloader class')
        self.dataloader = dataloader
        self.dataset_len = len(self.dataloader.dataset)
    
    def get_sample_indices(self, sample_size, exclude_indices=[]):
        sample_size = min(sample_size, self.dataset_len - len(exclude_indices))
        return random.sample(population = set([i for i in range(self.dataset_len)])-set(exclude_indices), k = sample_size)
    def extend_sample_indices(self, new_sample_size):
        n = len(self.sample_indices)
        new_indices = self.get_sample_indices(sample_size=new_sample_size-n, exclude_indices=self.sample_indices)
        self.sample_indices = self.sample_indices + new_indices
        
    def get_mini_batch(self):
        l = self.get_sample_indices(sample_size=2)
        for i, idx in enumerate(self.sample_indices):
            single_input, single_label = self.dataloader.dataset[idx]
            if i ==0:
                labels=[single_label]
                inputs = single_input.unsqueeze(0)
            else:
                #stack inputs
                labels.extend([single_label])
                inputs = torch.cat([inputs, single_input.unsqueeze(0)])

        if isinstance(labels, list):
            labels = torch.Tensor(labels)
        
        return inputs, labels

def train(args, model, device, train_loader, optimizer, epoch):
    
    model.train()
    
    train_loader_new = my_dataloader(
        dataloader = train_loader, 
        sample_indices = [i for i in range(args.initial_batch_size)])
    data, target = train_loader_new.get_mini_batch()
    target = target.long()
    #batch_idx, (data, target) = [train_loader][0]
    #optimizer.get_mini_batch(dataset = train_loader.dataset, action=1)
    data, target = data.to(device), target.to(device)
    print(f"data.size={data.size()}")
    optimizer.zero_grad()
    output = model(data)
    assert (data==data).all(), f"data={data}"
    assert (output==output).all(), f"output={output}"
    loss = F.nll_loss(output, target)
    assert (loss==loss).all(), f"loss={loss}"
    #loss=F.mse_loss(output,target)
    loss.backward()
    assert (loss==loss).all(), f"backward loss={loss}"
    optimizer.test_sscdm()
    grad_norm = [layer.grad.norm() for layer in optimizer.param_groups[0]['params']]
    print(f'Before step_global:\n\tloss={loss}\ngrad_norm={grad_norm}')
    train_run_flag=True

    while train_run_flag:
        deltas_tensor = optimizer.step_global()
        for i,p in enumerate(optimizer.param_groups[0]['params']):
            optimizer.param_groups[0]['params'][i].data = p.add(deltas_tensor[i])
        

        output = model(data)
        prev_loss = loss
        loss = F.nll_loss(output, target)
        
        max_sample_size = train_loader_new.dataset_len * 1
        
        eps = float(abs((prev_loss - loss) / prev_loss))
        current_sample_size = len(train_loader_new.sample_indices)
        if False and eps < 0.01 and current_sample_size < max_sample_size:
            train_loader_new.extend_sample_indices(current_sample_size*2)
            output = model(data)
            prev_loss = loss
            loss = F.nll_loss(output, target)
        if eps < 0.0001 and current_sample_size >= max_sample_size:
            train_run_flag = False
        loss.backward()
        grad_norm = [layer.grad.norm() for layer in optimizer.param_groups[0]['params']]
        print(f'After step_global:\n\tloss={loss}\ngrad_norm={grad_norm}\nlen(sample_indices)={len(train_loader_new.sample_indices)}\neps={eps}\nlast_step={optimizer.last_step}')


    #if batch_idx % args.log_interval == 0:
    #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #        epoch, batch_idx * len(data), len(train_loader.dataset),
    #        100. * batch_idx / len(train_loader), loss.item()))

def train_old(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        print(f"data.size={data.size()}")
        optimizer.zero_grad()
        output = model(data)
        assert (data==data).all(), f"data={data}"
        assert (output==output).all(), f"output={output}"
        loss = F.nll_loss(output, target)
        assert (loss==loss).all(), f"loss={loss}"
        #loss=F.mse_loss(output,target)
        loss.backward()
        assert (loss==loss).all(), f"backward loss={loss}"
        
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--optimizer', type=str, default='SGD', metavar='N',
                        help='optimizer SGD, Adam (default: SGD)')
    parser.add_argument('--method', type=str, default='cd', metavar='N',
                        help='optimizer method, [\'gd\',\'adam\',\'sscdm\',\'cdwm\',\'cd\'] (default: cd)')
    parser.add_argument('--cd_max_steps', type=int, default=1, metavar='N',
                        help='max steps for SSCDM (default: 1)')                    
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument("--initial_batch-size", type=int, default=64, metavar='N',
                        help='initial input batch size for training (default: 64)')
                        
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    print(model)
    if args.optimizer=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer=='Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer=='SSCDM':
        optimizer = SSCDM(model.parameters(), lr=args.lr, cd_max_steps=args.cd_max_steps, method = args.method)
    print(f'optimizer {optimizer}')
    for epoch in range(1, args.epochs + 1):
        s=time.time()

        train(args, model, device, train_loader, optimizer, epoch)
        print(f'Epoch N{epoch} Elapsed training time {int(time.time()-s):d} seconds')
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()
"""
TODO: 
3.implement adam, rmsprpop, adagrad
1.learning rate dynamic, review relevance based on 3
2.dynamically and deterministrically extend training dataset

4.implement sscdm

"""