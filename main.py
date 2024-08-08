import os
import argparse
import torch
from models.GDFlow import GDFlow
import numpy as np
from sklearn.metrics import roc_auc_score
import random
from torch.nn.utils import clip_grad_value_
import copy

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, 
                    default='Data/input/', help='Location of datasets.')
parser.add_argument('--output_dir', type=str, 
                    default='./checkpoint')
parser.add_argument('--name',default='SMD', help='the name of dataset')

parser.add_argument('--graph', type=str, default='None')
parser.add_argument('--n_blocks', type=int, default=1, help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
parser.add_argument('--n_components', type=int, default=1, help='Number of Gaussian clusters for mixture of gaussians models.')
parser.add_argument('--hidden_size', type=int, default=32, help='Hidden layer size for MADE (and each MADE block in an MAF).')
parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
parser.add_argument('--input_size', type=int, default=1)
parser.add_argument('--batch_norm', type=bool, default=True)
parser.add_argument('--train_split', type=float, default=0.6)
parser.add_argument('--stride_size', type=int, default=10)
parser.add_argument('--quantile', type=float, default=0.05, help='Specifies the quantile value for Quantile loss during training [0,1]. If the value is outside the range of 0 and 1, Quantile loss will not be applied.')
parser.add_argument('--no_ncde', action='store_true', help='RNN instead of NCDE') # Default: False
parser.add_argument('--no_quantile', action='store_true', help='Ignore the Quantile function during training') # Default: False

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--window_size', type=int, default=60)
parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate.')
parser.add_argument('--epo', type=int, default=10)

args = parser.parse_known_args()[0]
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

pre_list = []
rec_list = []
f1_list = []
auc_list = []
prc_list = []

# seed = 15
for seed in range(15,16):
    args.seed = seed
    print(args)

    # seed fixation
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        
    model_name = f"{args.name}"


    # Data load (train, val, test all include anomalous data)
    print("Loading dataset:", args.name)
    from Dataset import load_SMD, load_MSL_SMAP

    if args.name.startswith('machine'): # SMD
        train_loader, val_loader, test_loader, n_sensor, entity_mean, entity_covar = load_SMD(args.name, \
                                                                    args.batch_size, args.window_size, args.stride_size, args.train_split)
    elif args.name in ['MSL', 'SMAP']:
        train_loader, val_loader, test_loader, n_sensor, entity_mean, entity_covar = load_MSL_SMAP(args.name, \
                                                                    args.batch_size, args.window_size, args.stride_size, args.train_split)

    # Define Model
    model = GDFlow(device,
                    args.n_blocks,
                    args.input_size,
                    args.hidden_size,
                    args.n_hidden,
                    args.window_size,
                    n_sensor,
                    dropout=0.0,
                    batch_norm=args.batch_norm,
                    entity_mean=entity_mean,
                    entity_covar=entity_covar,
                    q=args.quantile,
                    ab_ncde=args.no_ncde,
                    ab_qunatile=args.no_quantile)
    model = model.to(device)
        
    # Path to save learned model parameter file
    save_path = os.path.join(args.output_dir,args.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    epochs = args.epo
    lr = args.lr 
    patience = 10
    epochs_no_improve = 0
    train_best = float('inf')
    
    
    optimizer = torch.optim.AdamW([
        {'params':model.parameters(), 'weight_decay':args.weight_decay},
        ], lr=lr, weight_decay=0.0) # 1e-5

    # Set learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.0001)

    # Model Training Phase
    for epoch in range(epochs):
        loss_train = []

        # train
        model.train()
        
        for batch in train_loader:
            batch = tuple(b.to(device, dtype=torch.float) for b in batch[:-1])
            x, *train_coeffs = batch

            optimizer.zero_grad()
            loss = -model(x, train_coeffs) # Scalar value (Negative log-likelihood)
            total_loss = loss
            total_loss.backward()
            clip_grad_value_(model.parameters(), 1)
            optimizer.step()
            loss_train.append(loss.item())

        loss_train_avg = sum(loss_train)/len(loss_train)

        if loss_train_avg < train_best:
            print(f"[Epoch {epoch}/{epochs-1}] | Train loss {loss_train_avg:.4f} | [SAVE MODEL]")
            torch.save(model, f"{save_path}/{model_name}.pth")
            train_best = loss_train_avg
            epochs_no_improve = 0  # Reset early stopping counter
        else:
            print(f"[Epoch {epoch}/{epochs-1}] | Train loss {loss_train_avg:.4f}")
            epochs_no_improve += 1  # Increment early stopping counter
            
        scheduler.step() # Update scheduler
        print(f"Current learning rate: {scheduler.get_last_lr()}")  # Print the learning rate
        
        # Check early stopping condition
        if epochs_no_improve == patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Model Test Phase
    best_model = torch.load(f"{save_path}/{model_name}.pth")

    data_test = []
    loss_test = []

    with torch.no_grad():
        best_model.eval()
        
        for batch in test_loader:
            batch = tuple(b.to(device, dtype=torch.float) for b in batch[:-1])
            x, *test_coeffs = batch

            loss = -best_model.test(x, test_coeffs).cpu().numpy() # (b,)
            data_test.append(x.squeeze(-1).cpu().numpy())
            loss_test.append(loss)
                
    data_test = np.concatenate(data_test)
    pred_score = np.concatenate(loss_test) # numpy.ndarray (T_test,)

    gt_label = np.asarray(test_loader.dataset.label,dtype=int) # numpy.ndarray (T_test,)
    
    def PA(y, y_pred):
        anomaly_state = False
        y_pred_pa = copy.deepcopy(y_pred)
        for i in range(len(y)):
            if y[i] == 1 and y_pred_pa[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if y[j] == 0:
                        break
                    else:
                        if y_pred_pa[j] == 0:
                            y_pred_pa[j] = 1
                for j in range(i, len(y)):
                    if y[j] == 0:
                        break
                    else:
                        if y_pred_pa[j] == 0:
                            y_pred_pa[j] = 1
            elif y[i] == 0:
                anomaly_state = False
            if anomaly_state:
                y_pred_pa[i] = 1

        return y_pred_pa
    
    def benchmarks_AD(true_labels, pred_score, threshold):
        pred_label = (pred_score > threshold) + 0 # If the value is greater than the threshold, pred_label == 1 (Anomaly), otherwise, pred_label == 0 (Normal)

        pred_label_pa = PA(true_labels, pred_label)
        return pred_label_pa

    def get_evaluation(gt_label, pred_score, threshold):
        pred_label = benchmarks_AD(gt_label, pred_score, threshold)
        auroc = roc_auc_score(gt_label, pred_score)
        return auroc, pred_label

    best_result = [-1]  # auroc

    auroc, pred_label = get_evaluation(gt_label, pred_score, threshold=0.97)
    best_result = [auroc]

    print(f'\nAUROC: {best_result[0]:.4f}')

    auc_list.append(best_result[0])
    
# Final results
final_auroc = sum(auc_list) / len(auc_list)

# Printing results
print('Final AUCROC:', final_auroc)
