# SOURCE: https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from examples.example_models.AE import AE


def main():

    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    batch_size = 512
    epochs = 5
    learning_rate = 1e-3

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AE(input_shape=784).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # mean-squared error loss
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        loss = 0
        for batch_features, _ in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = batch_features.view(-1, 784).to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(batch_features)

            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))

    torch.save(model.state_dict(), 'example_weights/AE_MNIST.pt')

    test_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform, download=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=10, shuffle=False
    )

    test_examples = None

    with torch.no_grad():
        for batch_features in test_loader:
            batch_features = batch_features[0]
            test_examples = batch_features.view(-1, 784)
            reconstruction = model(test_examples.to(device))
            break

    with torch.no_grad():
        number = 10
        plt.figure(figsize=(20, 4))
        for index in range(number):
            # display original
            ax = plt.subplot(2, number, index + 1)
            plt.imshow(test_examples[index].numpy().reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, number, index + 1 + number)
            plt.imshow(reconstruction.cpu()[index].numpy().reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


if __name__ == "__main__":
    main()