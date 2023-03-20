# SOURCE: https://github.com/lyeoni/pytorch-mnist-GAN/blob/master/pytorch-mnist-GAN.ipynb

# prerequisites
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from examples.example_models.GAN import Generator, Discriminator


def D_train(x, mnist_dim, bs, device, D, criterion, z_dim, G, D_optimizer):
    # =======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(bs, 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = Variable(torch.randn(bs, z_dim).to(device))
    x_fake, y_fake = G(z), Variable(torch.zeros(bs, 1).to(device))

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


def G_train(G ,bs, z_dim, device, D, criterion, G_optimizer):
    # =======================Train the generator=======================#
    G.zero_grad()

    z = Variable(torch.randn(bs, z_dim).to(device))
    y = Variable(torch.ones(bs, 1).to(device))

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bs = 100

    # MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    train_dataset = datasets.MNIST(root='example_data/mnist/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='example_data/mnist/', train=False, transform=transform, download=False)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

    # build network
    z_dim = 100
    mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)

    G = Generator(g_input_dim = z_dim, g_output_dim = mnist_dim).to(device)
    D = Discriminator(mnist_dim).to(device)

    # loss
    criterion = nn.BCELoss()

    # optimizer
    lr = 0.0002
    G_optimizer = optim.Adam(G.parameters(), lr=lr)
    D_optimizer = optim.Adam(D.parameters(), lr=lr)

    n_epoch = 5
    for epoch in range(1, n_epoch + 1):
        D_losses, G_losses = [], []
        for batch_idx, (x, _) in enumerate(train_loader):
            D_losses.append(D_train(x, mnist_dim, bs, device, D, criterion, z_dim, G, D_optimizer))
            G_losses.append(G_train(G, bs, z_dim, device, D, criterion, G_optimizer))

        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

    torch.save(G.state_dict(), 'example_weights/GAN_Generator_MNIST.pt')
    torch.save(D.state_dict(), 'example_weights/GAN_Discriminator_MNIST.pt')

    with torch.no_grad():
        test_z = Variable(torch.randn(bs, z_dim).to(device))
        generated = G(test_z)

        generated.view(generated.size(0), 1, 28, 28)

if __name__ == "__main__":
    main()