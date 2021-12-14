
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import imageio
import numpy as np
import matplotlib

from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm

import time
import datetime
time1 = time.time()


# def GCDS_imp(X_train, Y_train, X_test):
#
#     # epochs = 10000
#     epochs = 1000
#     learning_rate = 0.0002
#
#     nz = 1  # latent vector size
#     k = 1  # number of steps to apply to the discriminator
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # p_g = 6
#
#
#     from sklearn.preprocessing import StandardScaler
#
#     n = X_train.shape[0]
#     p = X_train.shape[1]
#     p_g = p + 1
#     batch_size = n//8 * 2
#     # p = 6
#
#     # mu = [0] * p
#     # sigma_1 = np.diagflat([1.0] * p)
#     # X = np.random.multivariate_normal(mu, sigma_1, n)
#     #
#
#     ### Choose simulation model
#     # model_num = 1
#     # model_num = 2
#
#     # Y_label = 'M'+str(model_num)
#     #
#     # if Y_label == 'M1':
#     #     Y1 = X[:, 0] ** 2 + np.exp(X[:, 1] + X[:, 2] / 3) + np.sin(X[:, 3] + X[:, 4]) + X[:, 5]
#     #     Y1_u = X[:, 0] ** 2 + np.exp(X[:, 1] + X[:, 2] / 3) + np.sin(X[:, 3] + X[:, 4])
#     #     Y1_sd = np.ones(n).reshape(-1,1)
#     #     Y = np.reshape(Y1, (-1, 1))
#     #     Y_u = np.reshape(Y1_u, (-1, 1))
#     #
#     # elif Y_label == 'M2':
#     #     Y2 = X[:, 0] ** 2 + np.exp(X[:, 1] + X[:, 2] / 3) + X[:, 3] - X[:, 4] + (0.5 + X[:, 1] ** 2 / 2 + X[:, 4] ** 2 / 2) * X[:, 5]
#     #     Y2_u = X[:, 0] ** 2 + np.exp(X[:, 1] + X[:, 2] / 3) + X[:, 3] - X[:, 4]
#     #     Y2_sd = (0.5 + X[:, 1] ** 2 / 2 + X[:, 4] ** 2 / 2).reshape(-1,1)
#     #     Y = np.reshape(Y2, (-1, 1))
#     #     Y_u = np.reshape(Y2_u, (-1, 1))
#     #
#     # elif Y_label == 'M3':
#     #     u = np.random.random(n)
#     #     eps = np.random.normal(((u > 0.5)-0.5) * 2 * 2, 1, n)
#     #     exp_eps = np.exp(eps * 0.5)
#     #     Y3 = (5 + X[:, 0] ** 2 / 3 + X[:, 1] ** 2 + X[:, 2] ** 2 + X[:, 3] + X[:, 4]) * exp_eps
#     #     Y3_u = (5 + X[:, 0] ** 2 / 3 + X[:, 1] ** 2 + X[:, 2] ** 2 + X[:, 3] + X[:, 4]) * (exp_eps.mean())
#     #     Y3_sd = (5 + X[:, 0] ** 2 / 3 + X[:, 1] ** 2 + X[:, 2] ** 2 + X[:, 3] + X[:, 4]) * (exp_eps.std()).reshape(-1,1)
#     #     Y = np.reshape(Y3, (-1, 1))
#     #     Y_u = np.reshape(Y3_u, (-1, 1))
#     #
#     # elif Y_label == 'M4':
#     #     n4 = 1000
#     #     u = np.random.random(n * n4)
#     #     Y4 = np.random.normal(((u > 0.5)-0.5) * 2 * np.repeat(X[:, 0], n4), 0.25**2, n * n4).reshape(n, -1)
#     #     Y4_u = np.zeros(n)
#     #     Y4_sd = Y4.std(axis=1).reshape(-1,1)
#     #     Y = np.reshape(Y4, (-1, 1))
#     #     Y_u = np.reshape(Y4_u, (-1, 1))
#     #
#     Y_train = np.reshape(Y_train, (-1, 1))
#
#     train_data = X_train
#     train_data_label = Y_train
#
#     train_loader = DataLoader(list(zip(train_data, train_data_label)), batch_size=batch_size, shuffle=True, drop_last = True)
#
#     hidden_size_g = 50
#     hidden_size_d1 = 50
#     hidden_size_d2 = 25
#
#
#     # hidden_size_g = 50 * 2
#     # hidden_size_d1 = 50 * 2
#     # hidden_size_d2 = 25 * 2
#
#
#     class Generator(nn.Module):
#         def __init__(self, p_g):
#             super(Generator, self).__init__()
#             self.p_g = p_g
#             self.main = nn.Sequential(
#                 nn.Linear(self.p_g, hidden_size_g),
#                 nn.ReLU(),
#
#                 # nn.Linear(hidden_size_g, hidden_size_g),
#                 # nn.ReLU(),
#
#                 nn.Linear(hidden_size_g, nz),
#                 # nn.Sigmoid(),
#                 # nn.Tanh(),
#                 # nn.ReLU(),
#             )
#
#         def forward(self, eta_g, xc):
#             # return self.main(x).view(-1, p)
#             # print([xc,eta_g.unsqueeze(1)])
#             c = torch.cat((xc, eta_g.unsqueeze(1)), 1)
#             return self.main(c).view(-1, nz)
#
#
#     class Discriminator(nn.Module):
#         def __init__(self):
#             super(Discriminator, self).__init__()
#             self.n_input = p_g
#             self.main = nn.Sequential(
#                 nn.Linear(self.n_input, hidden_size_d1),
#                 nn.ReLU(),
#                 # nn.Dropout(0.3),
#
#                 # nn.Linear(1024, 512),
#                 # nn.LeakyReLU(0.2),
#                 # nn.Dropout(0.3),
#
#                 nn.Linear(hidden_size_d1, hidden_size_d2),
#                 nn.ReLU(),
#                 # nn.Dropout(0.3),
#
#                 nn.Linear(hidden_size_d2, 1),
#                 # nn.Sigmoid(),
#                 # nn.Tanh(),
#
#                 # YS
#                 # nn.ReLU()
#                 # nn.Sigmoid()
#             )
#
#         def forward(self, y, xc):
#             # y = y.float().view(-1, len(y))
#             c = torch.cat((xc, y), 1)
#             # return self.main(x)
#             return self.main(c)
#
#
#     generator = Generator(p_g).to(device)
#     discriminator = Discriminator().to(device)
#
#     print('##### GENERATOR #####')
#     print(generator)
#     print('######################')
#
#     print('\n##### DISCRIMINATOR #####')
#     print(discriminator)
#     print('######################')
#
#     # optimizers
#     optim_g = optim.Adam(generator.parameters(), lr=learning_rate)
#     optim_d = optim.Adam(discriminator.parameters(), lr=learning_rate)
#
#
#     losses_g = []  # to store generator loss after each epoch
#     losses_d = []  # to store discriminator loss after each epoch
#     mse_mean = []  # to store images generatd by the generator
#     mse_std = []  # to store images generatd by the generator
#
#
#     # to create real labels (1s)
#     def label_real(size):
#         data = torch.ones(size, 1)
#         return data.to(device)
#
#
#     # to create fake labels (0s)
#     def label_fake(size):
#         data = torch.zeros(size, 1)
#         return data.to(device)
#
#
#     # function to create the noise vector
#     def create_noise(sample_size, nz):
#         # return torch.randn(sample_size, nz).to(device)
#         return torch.tensor(np.random.normal(size=sample_size)).float().to(device)
#
#     # to save the images generated by the generator
#     def save_generator_image(image, path):
#         save_image(image, path)
#
#     def train_discriminator(optimizer, label_real, gen_fake, image_real, image_fake):
#
#         optimizer.zero_grad()
#         output_real = discriminator(label_real, image_real)
#         loss_real = torch.exp(output_real) - 1
#
#         output_fake = discriminator(gen_fake, image_fake)
#         loss_fake = output_fake
#
#         loss = - torch.mean(loss_fake - loss_real)
#
#         loss.backward()
#         optimizer.step()
#
#         return loss
#
#
#     # function to train the generator network
#     def train_generator(optimizer, data_fake, image_fake):
#         optimizer.zero_grad()
#
#         output = discriminator(data_fake, image_fake)
#         loss = torch.mean(output)
#
#         loss.backward()
#         optimizer.step()
#
#         return loss
#
#
#     generator.train()
#     discriminator.train()
#
#     for epoch in range(epochs):
#         loss_g = 0.0
#         loss_d = 0.0
#         for bi, data in tqdm(enumerate(train_loader), total=int(len(train_data) / train_loader.batch_size)):
#             image, label = data
#             image = image.to(device)
#             b_size = len(image) // 2  # b_size = B/2
#
#             # run the discriminator for k number of steps
#             for step in range(k):
#                 idx = np.random.choice(b_size * 2, b_size, replace=False)
#                 idx_not = [i for i in range(len(image)) if i not in idx]
#
#                 image_real = image[idx].float()
#                 image_fake = image[idx_not].float()
#                 label_real = label[idx].float()
#                 label_fake = label[idx_not].float()
#
#                 gen_fake = generator(create_noise(b_size, nz), image_fake).detach()
#
#                 # train the discriminator network
#                 loss_d += train_discriminator(optim_d, label_real, gen_fake, image_real, image_fake)
#
#             gen_fake = generator(create_noise(b_size, nz), image_fake)
#             # train the generator network
#             loss_g += train_generator(optim_g, gen_fake, image_fake)
#
#         # create the final fake image for the epoch
#         # generated_img = generator(noise).cpu().detach()
#         generated_img_tol = generator(create_noise(n, nz), torch.tensor(train_data).float()).cpu().detach().numpy()
#         n_rep = 10
#
#
#         for i in range(n_rep-1):
#             generated_img_i = generator(create_noise(n, nz), torch.tensor(train_data).float()).cpu().detach().numpy()
#             generated_img_tol = np.append(generated_img_tol, generated_img_i, axis=1)
#
#
#         generated_img = np.mean(generated_img_tol, axis=1).reshape(-1,1)
#         generated_img_sd = np.std(generated_img_tol, axis=1).reshape(-1,1)
#
#         mse_mean_temp = ((Y_train - generated_img) ** 2).mean()
#
#         # if Y_label == 'M1':
#         #     Y_sd = Y1_sd
#         # elif Y_label == 'M2':
#         #     Y_sd = Y2_sd
#         # elif Y_label == 'M3':
#         #     Y_sd = Y3_sd
#         # elif Y_label == 'M4':
#         #     Y_sd = Y4_sd
#
#         # mse_std_temp = ((Y_sd - generated_img_sd) ** 2).mean()
#
#         mse_mean.append(mse_mean_temp)
#         # mse_std.append(mse_std_temp)
#
#         epoch_loss_g = loss_g / (bi+1)  # total generator loss for the epoch
#         epoch_loss_d = loss_d / (bi+1)  # total discriminator loss for the epoch
#         losses_g.append(epoch_loss_g)
#         losses_d.append(epoch_loss_d)
#
#         # print(f"Epoch {epoch} of {epochs}")
#         # print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")
#
#         # if epoch % 100 == 99:
#         #     torch.save(generator.state_dict(), f'outputs/generator.pth_{Y_label}_{epoch // 100}')
#
#     print('DONE TRAINING')
#
#     # # plot and save the generator and discriminator loss
#     # plt.figure()
#     # plt.plot(losses_g, label='Generator loss')
#     # plt.plot(losses_d, label='Discriminator Loss')
#     # plt.legend()
#     # plt.savefig('outputs/loss.png')
#     # plt.show()
#     #
#     # plt.figure()
#     # plt.plot(mse_mean, label='MSE loss')
#     # plt.ylim([0, 3])
#     # plt.legend()
#     # # plt.savefig('outputs/loss.png')
#     # plt.show()
#
#     ## parenthese
#     n_test = X_test.shape[0]
#     # rep_10_mean = []
#     # rep_10_std = []
#     # for rep_10 in range(10):
#     #     mu = [0] * p
#     #     sigma_1 = np.diagflat([1.0] * p)
#     #     X = np.random.multivariate_normal(mu, sigma_1, n_test)
#     #
#     #
#     #
#     #     if Y_label == 'M1':
#     #         Y1 = X[:, 0] ** 2 + np.exp(X[:, 1] + X[:, 2] / 3) + np.sin(X[:, 3] + X[:, 4]) + X[:, 5]
#     #         Y1_u = X[:, 0] ** 2 + np.exp(X[:, 1] + X[:, 2] / 3) + np.sin(X[:, 3] + X[:, 4])
#     #         Y1_sd = np.ones(n_test).reshape(-1,1)
#     #         Y = np.reshape(Y1, (-1, 1))
#     #         Y_u = np.reshape(Y1_u, (-1, 1))
#     #
#     #     elif Y_label == 'M2':
#     #         Y2 = X[:, 0] ** 2 + np.exp(X[:, 1] + X[:, 2] / 3) + X[:, 3] - X[:, 4] + (0.5 + X[:, 1] ** 2 / 2 + X[:, 4] ** 2 / 2) * X[:, 5]
#     #         Y2_u = X[:, 0] ** 2 + np.exp(X[:, 1] + X[:, 2] / 3) + X[:, 3] - X[:, 4]
#     #         Y2_sd = (0.5 + X[:, 1] ** 2 / 2 + X[:, 4] ** 2 / 2).reshape(-1,1)
#     #         Y = np.reshape(Y2, (-1, 1))
#     #         Y_u = np.reshape(Y2_u, (-1, 1))
#     #
#     #     elif Y_label == 'M3':
#     #         u = np.random.random(n_test)
#     #         eps = np.random.normal(((u > 0.5)-0.5) * 2 * 2, 1, n_test)
#     #         exp_eps = np.exp(eps * 0.5)
#     #         Y3 = (5 + X[:, 0] ** 2 / 3 + X[:, 1] ** 2 + X[:, 2] ** 2 + X[:, 3] + X[:, 4]) * exp_eps
#     #         Y3_u = (5 + X[:, 0] ** 2 / 3 + X[:, 1] ** 2 + X[:, 2] ** 2 + X[:, 3] + X[:, 4]) * (exp_eps.mean())
#     #         Y3_sd = (5 + X[:, 0] ** 2 / 3 + X[:, 1] ** 2 + X[:, 2] ** 2 + X[:, 3] + X[:, 4]) * (exp_eps.std()).reshape(-1,1)
#     #         Y = np.reshape(Y3, (-1, 1))
#     #         Y_u = np.reshape(Y3_u, (-1, 1))
#     #
#     #     elif Y_label == 'M4':
#     #         n4 = 1000
#     #         u = np.random.random(n_test * n4)
#     #         Y4 = np.random.normal(((u > 0.5)-0.5) * 2 * np.repeat(X[:, 0], n4), 0.25**2, n_test * n4).reshape(n_test, -1)
#     #         Y4_u = np.zeros(n_test)
#     #         Y4_sd = Y4.std(axis=1).reshape(-1,1)
#     #         Y = np.reshape(Y4, (-1, 1))
#     #         Y_u = np.reshape(Y4_u, (-1, 1))
#     #
#     #
#
#     test_data = X_test
#     # generator.load_state_dict(torch.load(f'outputs/generator.pth_{Y_label}_{68}'))
#
#     generated_img_tol = generator(create_noise(n_test, nz), torch.tensor(test_data).float()).cpu().detach().numpy()
#     n_rep_l = 1000
#     for i in range(n_rep_l-1):
#         generated_img_i = generator(create_noise(n_test, nz), torch.tensor(test_data).float()).cpu().detach().numpy()
#         generated_img_tol = np.append(generated_img_tol, generated_img_i, axis=1)
#
#     generated_img = np.mean(generated_img_tol, axis=1).reshape(-1)
#     # generated_img_sd = np.std(generated_img_tol, axis=1).reshape(-1,1)
#     # mse_u = ((Y_u - generated_img) ** 2).mean()
#
#     # if Y_label == 'M1':
#     #     Y_sd = Y1_sd
#     # elif Y_label == 'M2':
#     #     Y_sd = Y2_sd
#     # elif Y_label == 'M3':
#     #     Y_sd = Y3_sd
#     # elif Y_label == 'M4':
#     #     Y_sd = Y4_sd
#     #
#     # mse_sd = ((Y_sd - generated_img_sd) ** 2).mean()
#     #
#     # rep_10_mean.append(mse_u)
#     # rep_10_std.append(mse_sd)
#
#         # print(mse_u)
#         # print(mse_sd)
#     # rep_10_std = np.sqrt(rep_10_std)
#     # print(np.mean(rep_10_mean))
#     # print(np.mean(rep_10_std))
#     #
#     # print(np.std(rep_10_mean))
#     # print(np.std(rep_10_std))
#
#     time2 = time.time()
#     print(str(datetime.timedelta(seconds=time2 - time1)))
#
#     return np.array(generated_img)
#
#     #
#     # for p in generator.named_parameters():
#     #     print(p)
#     #
#     # for p in discriminator.named_parameters():
#     #     print(p)
#





## simplified version
def GCDS_imp(X_train, Y_train, X_test):

    # epochs = 10000
    # epochs = 1000 * 2
    epochs = 10
    learning_rate = 0.0002

    nz = 1  # latent vector size
    k = 1  # number of steps to apply to the discriminator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # p_g = 6


    from sklearn.preprocessing import StandardScaler

    n = X_train.shape[0]
    p = X_train.shape[1]
    p_g = p + 1
    batch_size = n//8 * 2
    # p = 6

    Y_train = np.reshape(Y_train, (-1, 1))

    train_data = X_train
    train_data_label = Y_train

    train_loader = DataLoader(list(zip(train_data, train_data_label)), batch_size=batch_size, shuffle=True, drop_last = True)

    hidden_size_g = 50
    hidden_size_d1 = 50
    hidden_size_d2 = 25

    # alternative
    hidden_size_g = 50 * 3
    hidden_size_d1 = 50 * 3
    hidden_size_d2 = 25 * 2

    # ##### M4
    # class Generator(nn.Module):
    #     def __init__(self, p_g):
    #         super(Generator, self).__init__()
    #         self.p_g = p_g
    #         self.main = nn.Sequential(
    #             nn.Linear(self.p_g, 40),
    #             nn.ReLU(),
    #
    #             nn.Linear(40, 15),
    #             nn.ReLU(),
    #
    #             nn.Linear(15, nz),
    #             # nn.Sigmoid(),
    #             # nn.Tanh(),
    #             # nn.ReLU(),
    #         )
    #
    #     def forward(self, eta_g, xc):
    #         # return self.main(x).view(-1, p)
    #         # print([xc,eta_g.unsqueeze(1)])
    #         c = torch.cat((xc, eta_g.unsqueeze(1)), 1)
    #         return self.main(c).view(-1, nz)



    class Generator(nn.Module):
        def __init__(self, p_g):
            super(Generator, self).__init__()
            self.p_g = p_g
            self.main = nn.Sequential(
                nn.Linear(self.p_g, hidden_size_g),
                nn.ReLU(),

                # nn.Linear(hidden_size_g, hidden_size_g),
                # nn.ReLU(),

                nn.Linear(hidden_size_g, nz),
                # nn.Sigmoid(),
                # nn.Tanh(),
                # nn.ReLU(),
            )

        def forward(self, eta_g, xc):
            # return self.main(x).view(-1, p)
            # print([xc,eta_g.unsqueeze(1)])
            c = torch.cat((xc, eta_g.unsqueeze(1)), 1)
            return self.main(c).view(-1, nz)


    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.n_input = p_g
            self.main = nn.Sequential(
                nn.Linear(self.n_input, hidden_size_d1),
                nn.ReLU(),
                # nn.Dropout(0.3),

                # nn.Linear(1024, 512),
                # nn.LeakyReLU(0.2),
                # nn.Dropout(0.3),

                nn.Linear(hidden_size_d1, hidden_size_d2),
                nn.ReLU(),
                # nn.Dropout(0.3),

                nn.Linear(hidden_size_d2, 1),
                # nn.Sigmoid(),
                # nn.Tanh(),

                # YS
                # nn.ReLU()
                # nn.Sigmoid()
            )

        def forward(self, y, xc):
            # y = y.float().view(-1, len(y))
            c = torch.cat((xc, y), 1)
            # return self.main(x)
            return self.main(c)


    generator = Generator(p_g).to(device)
    discriminator = Discriminator().to(device)

    print('##### GENERATOR #####')
    print(generator)
    print('######################')

    print('\n##### DISCRIMINATOR #####')
    print(discriminator)
    print('######################')

    # optimizers
    optim_g = optim.Adam(generator.parameters(), lr=learning_rate)
    optim_d = optim.Adam(discriminator.parameters(), lr=learning_rate)


    losses_g = []  # to store generator loss after each epoch
    losses_d = []  # to store discriminator loss after each epoch
    mse_mean = []  # to store images generatd by the generator
    mse_std = []  # to store images generatd by the generator


    # to create real labels (1s)
    def label_real(size):
        data = torch.ones(size, 1)
        return data.to(device)


    # to create fake labels (0s)
    def label_fake(size):
        data = torch.zeros(size, 1)
        return data.to(device)


    # function to create the noise vector
    def create_noise(sample_size, nz):
        # return torch.randn(sample_size, nz).to(device)
        return torch.tensor(np.random.normal(size=sample_size)).float().to(device)

    # to save the images generated by the generator
    def save_generator_image(image, path):
        save_image(image, path)

    def train_discriminator(optimizer, label_real, gen_fake, image_real, image_fake):

        optimizer.zero_grad()
        output_real = discriminator(label_real, image_real)
        loss_real = torch.exp(output_real) - 1

        output_fake = discriminator(gen_fake, image_fake)
        loss_fake = output_fake

        loss = - torch.mean(loss_fake - loss_real)

        loss.backward()
        optimizer.step()

        return loss


    # function to train the generator network
    def train_generator(optimizer, data_fake, image_fake):
        optimizer.zero_grad()

        output = discriminator(data_fake, image_fake)
        loss = torch.mean(output)

        loss.backward()
        optimizer.step()

        return loss


    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        loss_g = 0.0
        loss_d = 0.0
        # for bi, data in tqdm(enumerate(train_loader), total=int(len(train_data) / train_loader.batch_size)):
        for bi, data in enumerate(train_loader):
            image, label = data
            image = image.to(device)
            b_size = len(image) // 2  # b_size = B/2

            # run the discriminator for k number of steps
            for step in range(k):
                idx = np.random.choice(b_size * 2, b_size, replace=False)
                idx_not = [i for i in range(len(image)) if i not in idx]

                image_real = image[idx].float()
                image_fake = image[idx_not].float()
                label_real = label[idx].float()
                label_fake = label[idx_not].float()

                gen_fake = generator(create_noise(b_size, nz), image_fake).detach()

                # train the discriminator network
                loss_d += train_discriminator(optim_d, label_real, gen_fake, image_real, image_fake)

            gen_fake = generator(create_noise(b_size, nz), image_fake)
            # train the generator network
            loss_g += train_generator(optim_g, gen_fake, image_fake)

        # create the final fake image for the epoch
        # generated_img = generator(noise).cpu().detach()

        # generated_img_tol = generator(create_noise(n, nz), torch.tensor(train_data).float()).cpu().detach().numpy()
        # n_rep = 10
        #
        #
        # for i in range(n_rep-1):
        #     generated_img_i = generator(create_noise(n, nz), torch.tensor(train_data).float()).cpu().detach().numpy()
        #     generated_img_tol = np.append(generated_img_tol, generated_img_i, axis=1)
        #
        #
        # generated_img = np.mean(generated_img_tol, axis=1).reshape(-1,1)
        # generated_img_sd = np.std(generated_img_tol, axis=1).reshape(-1,1)
        #
        # mse_mean_temp = ((Y_train - generated_img) ** 2).mean()
        #
        #
        #
        # mse_mean.append(mse_mean_temp)
        # # mse_std.append(mse_std_temp)
        #
        # epoch_loss_g = loss_g / (bi+1)  # total generator loss for the epoch
        # epoch_loss_d = loss_d / (bi+1)  # total discriminator loss for the epoch
        # losses_g.append(epoch_loss_g)
        # losses_d.append(epoch_loss_d)
        #
        # # print(f"Epoch {epoch} of {epochs}")
        # # print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")
        #
        # # if epoch % 100 == 99:
        # #     torch.save(generator.state_dict(), f'outputs/generator.pth_{Y_label}_{epoch // 100}')
        if epoch % 100 == 0:
            print(f"Epoch {epoch} of {epochs}")
            timel = time.time()
            print(str(datetime.timedelta(seconds=timel - time1)))


    print('DONE TRAINING')

    # # plot and save the generator and discriminator loss
    # plt.figure()
    # plt.plot(losses_g, label='Generator loss')
    # plt.plot(losses_d, label='Discriminator Loss')
    # plt.legend()
    # plt.savefig('outputs/loss.png')
    # plt.show()
    #
    # plt.figure()
    # plt.plot(mse_mean, label='MSE loss')
    # plt.ylim([0, 3])
    # plt.legend()
    # # plt.savefig('outputs/loss.png')
    # plt.show()

    ## parenthese
    n_test = X_test.shape[0]

    test_data = X_test
    # generator.load_state_dict(torch.load(f'outputs/generator.pth_{Y_label}_{68}'))

    generated_img_tol = generator(create_noise(n_test, nz), torch.tensor(test_data).float()).cpu().detach().numpy()
    n_rep_l = 1000
    for i in range(n_rep_l-1):
        generated_img_i = generator(create_noise(n_test, nz), torch.tensor(test_data).float()).cpu().detach().numpy()
        generated_img_tol = np.append(generated_img_tol, generated_img_i, axis=1)

    generated_img = np.mean(generated_img_tol, axis=1).reshape(-1)


    time2 = time.time()
    print('Time used: ' + str(datetime.timedelta(seconds=time2 - time1)))

    return np.array(generated_img)


