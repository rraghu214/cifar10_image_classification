from __future__ import print_function
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchsummary import summary
from datetime import datetime
from ModelCIFAR10 import Net, get_optimizer_and_scheduler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.multiprocessing as mp

class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        # Albumentations works with numpy, so convert PIL → numpy
        img = np.array(img)
        augmented = self.transform(image=img)
        return augmented["image"]


# Transformations

train_transforms = AlbumentationsTransform(A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Affine(translate_percent={"x": 0.05, "y": 0.05},
         scale=(0.9, 1.1),
         rotate=(-10, 10),
         p=0.5),
    A.CoarseDropout(
        num_holes_range=(1, 1),
        hole_height_range=(16, 16),
        hole_width_range=(16, 16),
        fill = (125, 123, 114),
        fill_mask = None,
        p=0.5
),
    A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ToTensorV2(),
]))


test_transforms = AlbumentationsTransform(A.Compose([
    A.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)),
    ToTensorV2()
]))



train_losses = []
test_losses = []
train_acc = []
test_acc = []


def train(model, device, train_loader, optimizer, scheduler, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  running_loss = 0.0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    # train_losses.append(loss.item())

    # Backpropagation
    loss.backward()
    optimizer.step()

    scheduler.step()
    running_loss += loss.item()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(
        desc=f'Epoch {epoch} Loss={loss.item():.4f} '
             f'Batch_id={batch_idx} '
             f'Accuracy={100*correct/processed:0.2f}'
    )

    # train_losses.append(loss / len(train_loader))
    train_losses.append(loss.item()) 
    train_acc.append(100. * correct / processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    test_losses.append(test_loss)
    test_acc.append(accuracy)

    print(f"\nTest set: Average loss: {test_loss:.4f}, "
          f"Accuracy: {correct}/{len(test_loader.dataset)} "
          f"({accuracy:.2f}%)\n")

    return test_loss,accuracy





if __name__ == "__main__":

    mp.set_start_method('spawn', force=True)  # ✅ Windows-safe

    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transforms)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transforms)

    SEED = 1
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility


    if cuda:
        torch.cuda.manual_seed(SEED)
    else:
        torch.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args_train = dict(shuffle=True, batch_size=64, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args_train)

    dataloader_args_test = dict(shuffle=False, batch_size=1000, num_workers=2, pin_memory=True) if cuda else dict(shuffle=False, batch_size=64)
    # test dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args_test)


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    model = Net().to(device)
    summary(model, input_size=(3, 32, 32))

    model =  Net().to(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model,len(train_loader), EPOCHS=200)


    EPOCHS = 200
    target_acc = 85.0
    epoch = 0
    best_acc = 0.0
    # for epoch in range(EPOCHS):
    while True:
        epoch+=1
        print("EPOCH:", epoch)
        print(datetime.now().strftime("%d%m%Y-%H%M"))
        train(model, device, train_loader, optimizer, scheduler, epoch)
        print(datetime.now().strftime("%d%m%Y-%H%M"))
        val_loss,val_acc=test(model, device, test_loader)
        if val_acc > best_acc:
            best_acc = val_acc

        print(f"Validation Accuracy: {val_acc:.2f}% | Best Accuracy: {best_acc:.2f}%")
        print("-----------------------------------------------")
    
        if val_acc >= target_acc:
            print(f"Target reached! Accuracy = {val_acc:.2f}% at epoch {epoch}")
            break
    
        if epoch >= EPOCHS:  # safeguard stop
            print(f"Max epochs {EPOCHS} reached. Best Accuracy = {best_acc:.2f}%")
            break
        
        print("-----------------------------------------------")
    
    fig, axs = plt.subplots(2,2,figsize=(15,10))

    # axs[0, 0].plot(train_losses)
    axs[0, 0].plot([loss for loss in train_losses], label='Training Loss')
    axs[0, 0].set_title("Training Loss")
    axs[0, 0].legend()

    axs[1, 0].plot([acc for acc in train_acc], label="Train Accuracy")
    axs[1, 0].set_title("Training Accuracy")
    axs[1, 0].legend()
    
    axs[0, 1].plot([loss for loss in test_losses], label="Test Loss", color="orange")
    axs[0, 1].set_title("Test Loss")
    axs[0, 1].legend()

    axs[1, 1].plot([acc for acc in test_acc], label="Test Accuracy", color="green")
    axs[1, 1].set_title("Test Accuracy")
    axs[1, 1].legend()
        
    # Format timestamp as DDMMYYYY-HHMM
    
    timestamp = datetime.now().strftime("%d%m%Y-%H%M")

    # Save with timestamp in filename
    filename = f"training_results-{timestamp}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

    print(f"Plot saved as {filename}")
    # plt.show()