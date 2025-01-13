import argparse
import os
import torch
import yaml
from torch.utils.data import DataLoader
from torch import nn
from tensorboardX import SummaryWriter
from data_utils import genSpoof_list, Dataset_ASVspoof2019_train, Dataset_ASVspoof2021_eval
from model import RawNet
from startup_config import set_random_seed


def evaluate_accuracy(dev_loader, model, device):
    """Evaluate the accuracy of the model on the validation dataset."""
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in dev_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_out = model(batch_x)
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum().item()
            num_total += batch_y.size(0)
    return 100 * (num_correct / num_total)


def train_epoch(train_loader, model, optimizer, criterion, device):
    """Train the model for one epoch."""
    running_loss = 0.0
    num_correct = 0.0
    num_total = 0.0
    model.train()

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        batch_out = model(batch_x)
        loss = criterion(batch_out, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum().item()
        num_total += batch_y.size(0)

    epoch_loss = running_loss / num_total
    epoch_accuracy = 100 * (num_correct / num_total)
    return epoch_loss, epoch_accuracy


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    parser.add_argument('--database_path', type=str, default='your\\path\\to\\LA', help='Path to the LA database.')
    parser.add_argument('--protocols_path', type=str, default='your\\path\\to\\LA\\ASVspoof2019_LA_cm_protocols', help='Path to the protocol files.')
    parser.add_argument('--model_config_path', type=str, default='your\\path\\to\\LA\\model_config_RawNet.yaml', help='Path to the model configuration YAML file.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for optimizer.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--model_path', type=str, default=None, help='Path to load a pre-trained model.')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs.')
    args = parser.parse_args()

    # Make experiment reproducible
    set_random_seed(args.seed)

    # GPU or CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Load model configuration
    if not os.path.exists(args.model_config_path):
        raise FileNotFoundError(f"Model configuration file not found at {args.model_config_path}")
    with open(args.model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    # Initialize model
    model = RawNet(model_config['model'], device).to(device)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f'Model loaded from {args.model_path}')

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # Prepare training data
    train_audio_dir = os.path.join(args.database_path, 'ASVspoof2019_LA_train', 'flac')
    protocol_file_train = os.path.join(args.protocols_path, 'ASVspoof2019.LA.cm.train.trn.txt')
    if not os.path.exists(protocol_file_train):
        raise FileNotFoundError(f"Training protocol file not found at {protocol_file_train}")
    if not os.path.exists(train_audio_dir):
        raise FileNotFoundError(f"Training audio directory not found at {train_audio_dir}")

    d_label_trn, file_train = genSpoof_list(protocol_file_train, is_train=True, is_eval=False)
    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train, labels=d_label_trn, base_dir=train_audio_dir)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)

    # Prepare validation data
    val_audio_dir = os.path.join(args.database_path, 'ASVspoof2019_LA_dev', 'flac')
    protocol_file_dev = os.path.join(args.protocols_path, 'ASVspoof2019.LA.cm.dev.trl.txt')
    if not os.path.exists(protocol_file_dev):
        raise FileNotFoundError(f"Validation protocol file not found at {protocol_file_dev}")
    if not os.path.exists(val_audio_dir):
        raise FileNotFoundError(f"Validation audio directory not found at {val_audio_dir}")

    d_label_dev, file_dev = genSpoof_list(protocol_file_dev, is_train=False, is_eval=False)
    dev_set = Dataset_ASVspoof2019_train(list_IDs=file_dev, labels=d_label_dev, base_dir=val_audio_dir)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Training loop
    writer = SummaryWriter(args.log_dir)
    best_acc = 0.0
    for epoch in range(args.num_epochs):
        train_loss, train_acc = train_epoch(train_loader, model, optimizer, criterion, device)
        val_acc = evaluate_accuracy(dev_loader, model, device)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f'Epoch {epoch+1}/{args.num_epochs}, Loss: {train_loss:.4f}, '
              f'Train Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'your\\path\\to\\checkpoints\\best_model1.pth')
            print(f'Best model saved with accuracy: {best_acc:.2f}%')

    writer.close()
