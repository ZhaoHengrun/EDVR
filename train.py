import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import EDVR
from dataset import *
import visdom
import wandb
import numpy as np

torch.cuda.set_device(0)  # use the chosen gpu

none_or_wandb_or_visdom = 0  # 0 is none, 1 is wandb, 2 is visdom

if none_or_wandb_or_visdom == 1:
    wandb.init(project="EDVR")
elif none_or_wandb_or_visdom == 2:
    vis = visdom.Visdom(env='EDVR')

# Training settings
parser = argparse.ArgumentParser(description="PyTorch EDSR")
parser.add_argument("--dataset", default='datasets/', type=str, help="dataset path")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")  # default 16
parser.add_argument("--nEpochs", type=int, default=10000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=4e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=1000,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", default=True, action="store_true", help="use cuda?")
parser.add_argument("--resume", default='', type=str,
                    help="path to latest checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=16, help="number of threads for data loader to use")
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument("--weight-decay", "--wd", default=0, type=float, help="weight decay, Default: 0")

min_avr_loss = 99999999
save_flag = 0
epoch_avr_loss = 0
n_iter = 0


class CharbonnierLoss(torch.nn.Module):

    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        diff = torch.add(x, -y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    # print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = get_training_set()
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True,
                                      num_workers=opt.threads)

    print("===> Building model")
    model = EDVR()

    criterion = CharbonnierLoss()
    # criterion = nn.L1Loss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    if none_or_wandb_or_visdom == 1:
        wandb.watch(model)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = 0
            model.load_state_dict(checkpoint.state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # optimizer = optim.Adam(model.parameters(), lr=opt.lr,
    #                        weight_decay=opt.weight_decay, betas=(0.9, 0.999), eps=1e-08)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    # lr = opt.lr
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    print('lr{}  iter:'.format(lr, n_iter))
    return lr


def train(training_data_loader, optimizer, model, criterion, epoch):
    global min_avr_loss
    global save_flag
    global epoch_avr_loss
    global n_iter

    avr_loss = 0

    lr = adjust_learning_rate(optimizer, epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        n_iter = iteration
        input, target = batch[0], batch[1]  # input: b, t, c, h, w target: t, c, h, w

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        out = model(input)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avr_loss += loss.item()

        # if iteration % 100 == 0:
        print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader),
                                                            loss.item()))
    avr_loss = avr_loss / len(training_data_loader)

    if none_or_wandb_or_visdom == 1:
        wandb.log({"Test avr_loss": avr_loss})
    elif none_or_wandb_or_visdom == 2:
        vis.line(Y=np.array([avr_loss]), X=np.array([epoch]),
                 win='loss',
                 opts=dict(title='loss'),
                 update='append'
                 )

    epoch_avr_loss = avr_loss
    if epoch_avr_loss < min_avr_loss:
        min_avr_loss = epoch_avr_loss
        print('|||||||||||||||||||||min_epoch_loss is {:.10f}|||||||||||||||||||||'.format(min_avr_loss))
        save_flag = True
    else:
        save_flag = False
        print('epoch_avr_loss is {:.10f}'.format(epoch_avr_loss))


def save_checkpoint(model, epoch):
    global min_avr_loss
    global save_flag

    model_folder = "checkpoints/"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_out_path = model_folder + "model_epoch_{}.pth".format(epoch)
    # state = {"epoch": epoch, "model": model}
    if (epoch % 50) == 0:
        torch.save(model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    if save_flag is True:
        torch.save(model, '{}epoch_{}_min_batch_loss_{}.pth'.format(model_folder, epoch, min_avr_loss))
        print('min_loss model saved')


if __name__ == "__main__":
    main()
