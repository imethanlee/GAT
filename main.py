from data.cora import *
from model.gat import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--n_hid', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--n_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
adj, features, labels, idx_train, idx_val, idx_test = CoraDataset.load_data()
adj = adj.to(device)
features = features.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)
model = GAT().to(device)
criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


class EarlyStop:
    def __init__(self, patience: int, minimization: bool = True):
        self.cnt = 0
        self.patience = patience
        self.minimization = minimization
        if minimization:
            self.best_val = np.inf
        else:
            self.best_val = -np.inf

    def __is_better_than(self, input1, input2):
        if self.minimization:
            return input1 < input2
        else:
            return input1 > input2

    def check(self, val):
        if self.__is_better_than(val, self.best_val):
            self.best_val = val
            self.cnt = 0
        self.cnt += 1
        if self.cnt > self.patience:
            return True
        return False


def test(idx):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)
        test_mask_logits = logits[idx]
        predict_y = test_mask_logits.max(1)[1]
        accuracy = torch.eq(predict_y, labels[idx]).float().mean()
    return accuracy, test_mask_logits.cpu().numpy(), labels[idx].cpu().numpy()


def train():
    earlystop = EarlyStop(args.patience, False)
    loss_history = []
    val_acc_history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        y_pred = model(features, adj)[idx_train]
        loss = criterion(y_pred, labels[idx_train])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, _, _ = test(idx_train)
        val_acc, _, _ = test(idx_val)

        if earlystop.check(val_acc):
            break

        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
            epoch, loss.item(), train_acc.item(), val_acc.item()))
        print(earlystop.cnt)
    return loss_history, val_acc_history


train()
print(test(idx_test)[0])




