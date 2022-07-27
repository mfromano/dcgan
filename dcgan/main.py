from src import *
from src.models import G, D, weights_init

DEVICE = torch.device(
        "cuda:0" if (
                torch.cuda.is_available() and N_GPU > 0
        )
        else "cpu"
)

class CelebA(Dataset):
    def __init__(self, train_val_test: str):
        imgs = glob.glob(IMG_LOCATION)

netG = G(N_GPU).to(DEVICE)

netG.apply(weights_init)

netD = D(N_GPU).to(DEVICE)

netD.apply(weights_init)

def main():
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, N_FEAT_LATENT, 1, 1, device=DEVICE)
    real_label = 1
    fake_label = 0
    optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(Adam_beta, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(Adam_beta, 0.999))

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    for epoch in range(EPOCHS):
        for i, data in enumerate(DATALOADER):
            netD.zero_grad()
            real_cpu = data[0].to(DEVICE)
            b_size = real_cpu.size(0)
            label = torch.full(
                    (b_size,),
                    real_label,
                    dtype=torch.float,
                    device=DEVICE)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, N_FEAT_LATENT, 1, 1, device=DEVICE)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = D_x + D_G_z1
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()

            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print(
                    '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): '
                    '%.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, EPOCHS, i, len(DATALOADER),
                       errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            G_losses.append(errG.item())