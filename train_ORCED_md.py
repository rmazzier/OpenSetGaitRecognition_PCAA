import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import os
import wandb
import pickle

import constants
from models import ORCED_MD
from datasets import MSRadarDatasetMD, SPLIT
from utils import save_model
from inference_ORCED_md import ORCED_inference


class DeterministicWarmup():
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1 / n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t  # 0->1
        return self.t
    
def train_orced_md(config, wandb_mode="online"):
    model_path = os.path.join("models", config["MODEL_NAME"])
    os.makedirs(model_path, exist_ok=True)

    config_path = os.path.join(model_path, "config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    beta = DeterministicWarmup(n=60)
    model = ORCED_MD(alpha=100, beta=0, gamma=1, xi=1, tau=0.07, num_class=len(config["TRAIN_CLASSES"])).to(constants.DEVICE)

    # Configure data loader
    radar_dataset_train = MSRadarDatasetMD(
        SPLIT.TRAIN,
        subsample_factor=config["SUBSAMPLE_FACTOR"], 
        imagenet_transform=False
    )

    dataloader_train = torch.utils.data.DataLoader(
        radar_dataset_train,
        batch_size=config["BATCH_SIZE"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    radar_dataset_valid = MSRadarDatasetMD(
        SPLIT.VALID,
        subsample_factor=config["SUBSAMPLE_FACTOR"], 
        imagenet_transform=False
    )

    dataloader_valid = torch.utils.data.DataLoader(
        radar_dataset_valid,
        batch_size=config["BATCH_SIZE"],
        drop_last=True,
        shuffle=False,
        num_workers=0,
    )

    # optimzer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=1e-3)
    decreasing_lr = [60]
    beta = DeterministicWarmup(n=60, t_max=1)  # Linear warm-up from 0 to 1 over 50 epoch

    # Init wandb
    wandb.login()
    run = wandb.init(
        project="Semisup_AAE",
        config=config,
        name=config["MODEL_NAME"],
        notes=config["NOTES"],
        reinit=True,
        mode=wandb_mode,
    )


    best_val_acc = 0
    patience=0
    max_patience=30
    # train
    for epoch in range(config["EPOCHS"]):
        model.train()
        print("Training... Epoch = %d" % epoch)
        correct_train = 0
        train_tot_losses = []
        train_ce_losses = []
        train_kl_losses = []
        train_rec_losses = []
        train_sc_losses = []
        model.beta = beta.__next__()
        print(f" ====== BETA INCREASED TO: {model.beta} ========")

        if epoch in decreasing_lr:
            optimizer.param_groups[0]['lr'] *= 0.1
            print("~~~learning rate:", optimizer.param_groups[0]['lr'])
            

        for batch_idx, (data, target) in enumerate(dataloader_train):
            target_en = torch.Tensor(target.shape[0], len(config["TRAIN_CLASSES"]))
            target_en.zero_()
            target_en.scatter_(1, target.unsqueeze(1), 1)  # one-hot encoding
            target_en = target_en.to(constants.DEVICE)

            data = data.to(constants.DEVICE)
            target = target.to(constants.DEVICE)
            data, target = Variable(data), Variable(target)

            loss, mu, output, x_re, L_ce, L_kl, L_rec, L_sc = model.loss(data, target, target_en)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_tot_losses.append(loss.item())
            train_ce_losses.append(L_ce.item() * model.alpha)
            train_kl_losses.append(L_kl.item() * model.beta)
            train_sc_losses.append(L_sc.item() * model.gamma)
            train_rec_losses.append(L_rec.item() * model.xi)

            outlabel = output.data.max(1)[1]  # get the index of the max log-probability
            correct_train += outlabel.eq(target.view_as(outlabel)).sum().item()
            
            if batch_idx % 50 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader_train.dataset)} ({100. * batch_idx * len(data) / len(dataloader_train.dataset):.0f}%)]")
        

        train_acc = float(100 * correct_train) / len(dataloader_train.dataset)
        train_tot_loss = np.mean(train_tot_losses)
        train_ce_loss = np.mean(train_ce_losses)
        train_kl_loss = np.mean(train_kl_losses)
        train_rec_loss = np.mean(train_rec_losses)
        train_sc_loss = np.mean(train_sc_losses)

        print('Train_Acc: {}/{} ({:.2f}%)'.format(correct_train, len(dataloader_train.dataset), train_acc))

        ## Validation phase
        model.eval()
        correct_val = 0

        val_tot_losses = []
        val_ce_losses = []
        val_kl_losses = []
        val_rec_losses = []
        val_sc_losses = []

        for data_val, target_val in dataloader_valid:
            target_val_en = torch.Tensor(target_val.shape[0], len(config["TRAIN_CLASSES"]))
            target_val_en.zero_()
            target_val_en.scatter_(1, target_val.view(-1, 1), 1)  # one-hot encoding
            target_val_en = target_val_en.to(constants.DEVICE)
            
            data_val = data_val.to(constants.DEVICE)
            target_val =  target_val.to(constants.DEVICE)
            with torch.no_grad():
                data_val, target_val = Variable(data_val), Variable(target_val)

            loss_val, mu_val, output_val, output_mu_val, L_ce_val, L_kl_val, L_rec_val, L_sc_val = model.loss(data_val, target_val, target_val_en)
            val_tot_losses.append(loss_val.item())
            val_ce_losses.append(L_ce_val.item() * model.alpha)
            val_kl_losses.append(L_kl_val.item() * model.beta)
            val_sc_losses.append(L_sc_val.item() * model.gamma)
            val_rec_losses.append(L_rec_val.item() * model.xi)

            vallabel = output_val.data.max(1)[1]  # get the index of the max log-probability
            correct_val += vallabel.eq(target_val.view_as(vallabel)).sum().item()

        val_acc = float(100 * correct_val) / len(dataloader_valid.dataset)
        val_tot_loss = np.mean(val_tot_losses)
        val_ce_loss = np.mean(val_ce_losses)
        val_kl_loss = np.mean(val_kl_losses)
        val_rec_loss = np.mean(val_rec_losses)
        val_sc_loss = np.mean(val_sc_losses)


        # print('====> Epoch: {} Val loss: {}/{} ({:.4f}={:.4f}+{:.4f}+{:.4f})'.format(epoch, total_val_loss, len(dataloader_valid.dataset), val_loss, val_rec, val_kl, val_ce))
        print('Val_Acc: {}/{} ({:.2f}%)'.format(correct_val, len(dataloader_valid.dataset), val_acc))

        wandb.log(
            {
                "Total Loss Train": train_tot_loss,
                "CE Loss Train": train_ce_loss,
                "KL Loss Train": train_kl_loss,
                "Reconstruction Loss Train": train_rec_loss,
                "Contrastive Loss Train": train_sc_loss,
                "Train Accuracy": train_acc,
                "Total Loss Valid": val_tot_loss,
                "CE Loss Valid": val_ce_loss,
                "KL Loss Valid": val_kl_loss,
                "Reconstruction Loss Valid": val_rec_loss,
                "Contrastive Loss Valid": val_sc_loss,
                "Valid Accuracy": val_acc,
            }
        )

        if val_acc > best_val_acc:
            patience=0
            best_val_acc = val_acc

            ## Save all model weights
            print("New best model found! Saving...")

            # Save model checkpoints
            encoder_path = os.path.join(
                model_path, f"{config['MODEL_NAME']}.pt"
            )
            save_model(model, encoder_path)

            ## Test load parameters 
            model2 = ORCED_MD(alpha=20, beta=0, gamma=1, xi=1, tau=0.07, num_class=len(config["TRAIN_CLASSES"])).to(constants.DEVICE)
            statedict = torch.load(os.path.join(model_path, f"{config['MODEL_NAME']}.pt"), weights_only=True)
            model2.load_state_dict(statedict)
            print("Model loaded successfully")
        else:
            patience+=1
        
        if patience >= max_patience:
            print("Max Patience Reached. Stopping Training.")
            break


if __name__ == "__main__":

    from utils import openness

    model_name_base = "ORCED_MD_TESTADAM"
    n_training_classes = [2, 4, 6, 8]
    # n_training_classes = [6]

    n_tests = 5
    # n_tests = 1

    all_classes = list(range(len(MSRadarDatasetMD.label_dict)))

    splits_seed = 0
    rng = np.random.default_rng(splits_seed)

    # Iterate on number of training classes
    for n_tr in n_training_classes:
        selected_classes_sets = []
        # Iterate the different tests over the same n_tr
        for i in range(n_tests):
            # This while loop is done to make sure we don't pick the same subset of classes two times
            while True:
                random_train_classes = rng.choice(len(MSRadarDatasetMD.label_dict), n_tr, replace=False)
                random_train_classes = list(sorted(random_train_classes))
                if random_train_classes not in selected_classes_sets:
                    selected_classes_sets.append(random_train_classes)
                    break

            model_name = f"{model_name_base}.{n_tr}.{i+1}"

            config = constants.CONFIG
            config["MODEL_NAME"] = model_name
            config["TRAIN_CLASSES"] = random_train_classes
            config["EPOCHS"] = 100
    
            config["Openness"] = openness(n_tr, len(MSRadarDatasetMD.label_dict))

            # FIXME Uncomment for real runs
            MSRadarDatasetMD.generate_splits(
                train_classes=config["TRAIN_CLASSES"], split_seed=0, safe_mode=False
            )

            train_orced_md(config, wandb_mode="online")
# 
            ORCED_inference([config["MODEL_NAME"]])

