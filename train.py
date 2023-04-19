import torch
import numpy as np
import pandas as pd
from model import AgeNet
from math import sqrt
import matplotlib.pyplot as plt
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dataset_prep import get_dl
from torch.nn import L1Loss
from time import time
from math import sqrt
from scipy.stats import pearsonr
import argparse
from loss import *
from copy import deepcopy

def gather_dist(dl, model, device='cpu'):
    pred_ages = []
    acc_ages = []
    for i,data in enumerate(dl):
        images, age = data['image'].to(device), data['label'].float().tolist()
        pred_age,_ = model(images)
        pred_age = pred_age.squeeze().tolist()
        acc_ages.extend(age)
        pred_ages.extend(pred_age)
    return acc_ages, pred_ages

def MAE_RMSE_DL(dl, model, device='cpu'):
    mae_tot = 0
    rmse_tot = 0
    num_samples = 0
    for i,data in enumerate(dl):
        images, acc_age = data['image'].to(device), data['label'].to(device).float()
        pred_age,_ = model(images)
        pred_age = pred_age.squeeze()
        mae_tot += abs(pred_age - acc_age).sum().item()
        rmse_tot += torch.pow(pred_age - acc_age,2).sum().item()
        num_samples += len(pred_age)
    return mae_tot/num_samples, sqrt(rmse_tot/num_samples)

def pearsonr_calc(acc_age, pred_age):
    return pearsonr(acc_age, pred_age)


def gather_error_dist(acc_age, pred_ages):
    acc_age_u = list(set(acc_age))
    acc_age = np.array(acc_age)
    pred_ages = np.array(pred_ages)
    l1_diff = abs(pred_ages - acc_age)
    label_rec = []
    mae_rec = []
    for age in acc_age_u:
        relevent = (acc_age == age)
        label_rec.append(age)
        mae = np.sum(relevent * l1_diff) / np.sum(relevent)
        mae_rec.append(mae)

    return label_rec, mae_rec

def train_loop(num_epochs, train_dl, val_dl, device, optimizer, loss_fnc_name, loss_fnc, model, tlRec, tMAERec, tRMSERec, vlRec, scheduler = None):
    best_model = None
    min_loss = np.inf
    for e in range(num_epochs):
        lsum = 0
        num_samples = 0
        
        t0 = time()

        for i,data in enumerate(train_dl):
            
            imgs, ages = data['image'].to(device), data['label'].to(device).float()
            optimizer.zero_grad()

            outputs, features = model(imgs)
            
            if loss_fnc_name == 'supCR':
                loss = loss_fnc(features, ages)
            else:
                loss = loss_fnc(outputs, ages.unsqueeze(1))
            
            loss.backward()

            optimizer.step()
            lsum += loss.item()
            num_samples += len(imgs)
            print(i, flush=True)
        
        t1 = time()
        print(f"epoch e={e}",flush=True)
        print(f"Training epoch took: {t1-t0:.2f} s",flush=True)
        model.eval()

        tlRec.append(lsum/num_samples)
        print("Train MAE and RMSE:",flush=True)
        mae, rmse = MAE_RMSE_DL(train_dl,model,device)
        print(mae, rmse, flush=True)
        tMAERec.append(mae)
        tRMSERec.append(rmse)

        lsum = 0
        num_samples = 0
        for i,data in enumerate(val_dl):
            images, ages = data['image'].to(device), data['label'].to(device).float()
            outputs, features = model(images)
            if loss_fnc_name == 'supCR':
                loss = loss_fnc(features, ages)
            else:
                loss = loss_fnc(outputs, ages.unsqueeze(1))
            lsum += loss.item()
            num_samples += len(images)
        
        latest_val_loss = lsum/num_samples
        vlRec.append(latest_val_loss)
        if latest_val_loss < min_loss:
            best_model = deepcopy(model)
            min_loss = latest_val_loss
            print(f"Improved val loss, current min at: {latest_val_loss}", flush=True)

        t2 = time()
        print(f"Eval epoch took: {t2-t1:.2f} s", flush=True)
        model.train()

        if scheduler:
            scheduler.step()
    
    return best_model

def train(out_path_root, 
          loss_fnc_name = 'l1', 
          show_plots = False, 
          model_base = "",
          num_epochs = 1000,
          num_epochs_reg = 1000,
          bs = 128,
          lr = 0.001,
          seed = 100,
          t = 2,
          optim="adam",
          nw = 0,
          img_size='og',
          save_best=True):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if ((img_size != 'og') and (img_size != '112')):
        print("image size invalid!")
        return

    train_dl, test_dl, val_dl, test_uniform_dl = get_dl(train_csv="imdb_train.csv",
                                                        val_csv="imdb_val.csv",
                                                        test_csv="imdb_test.csv",
                                                        test_uniform_csv="imdb_test_uniform.csv",
                                                        img_size=img_size,
                                                        batch_size=bs,
                                                        drop_last=True,
                                                        num_workers=nw)

    if loss_fnc_name == 'l1':
        loss_fnc = l1_loss
    elif loss_fnc_name == 'supCR':
        sc = SupCRLoss(t, device=device)
        loss_fnc = sc.supcr_v2_pt
    else:
        print("loss_fnc is incorrect!")
        return
    
    tlRec = []
    vlRec = []
    tMAERec = []
    tRMSERec = []
    # test_accs = []
    # test_preds = []
    # test_uniform_accs = []
    # test_pred_accs = []
    # train_accs = []
    # train_preds = []

    acc_pred_rec = dict([])

    model = AgeNet().float().to(device)
    if model_base:
        model.load_state_dict(torch.load(model_base))

    # optimizer = Adam(model.parameters(), lr=lr)
    if optim == "adam":
        optimizer = Adam(model.parameters(), lr=lr)
        scheduler = None
    elif optim == "SGDR":
        optimizer = SGD(model.parameters(), lr=lr)
        scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                T_0=50)
    else:
        print("Invalid optim option! " + optim)
        return


    model.train()

    if loss_fnc_name == 'supCR':
        model.lin_reg.requires_grad_(False)
        if optim=="adam":
            optimizer = Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=lr)
            scheduler = None
        elif optim == "SGDR":
            optimizer = SGD(model.parameters(), lr=lr)
            scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                T_0=50)
        temp : AgeNet = train_loop(num_epochs, train_dl, val_dl, device, optimizer, loss_fnc_name, loss_fnc, model, tlRec, tMAERec, tRMSERec, vlRec, scheduler)
        if save_best:
            model = temp
            model.train()
        model.lin_reg.requires_grad_(True)
        model.resnet.requires_grad_(False)
        if optim=="adam":
            optimizer = Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=lr)
            scheduler = None
        elif optim == "SGDR":
            optimizer = SGD(model.parameters(), lr=lr)
            scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                T_0=50)
        loss_fnc = l1_loss
        model = train_loop(num_epochs_reg, train_dl, val_dl, device, optimizer, 'l1', loss_fnc, model, tlRec, tMAERec, tRMSERec, vlRec, scheduler)
    else:
        train_loop(num_epochs, train_dl, val_dl, device, optimizer, loss_fnc_name, loss_fnc, model, tlRec, tMAERec, tRMSERec, vlRec, scheduler)

    model.eval()

    pearson_rec = dict([])
    
    for dl,mode in [(train_dl,'train'), (test_dl,'test'), (test_uniform_dl,'test_uniform')]:
        print(f"Getting dist for {mode}")
        a,b = gather_dist(dl,model,device)

        acc_pred_rec[mode] = dict([])
        acc_pred_rec[mode]['acc'] = a
        acc_pred_rec[mode]['pred'] = b

        plt.hist2d(a,b)
        plt.title(f"Pred vs actual HRs - {mode}")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predicted")
        plt.savefig(out_path_root + f"//2dhist_{mode}.png")
        if show_plots:
            plt.show()
        else:
            plt.close()

        plt.scatter(a,b, alpha=0.05)
        plt.title(f"Pred vs actual HRs - {mode}")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predicted")
        plt.savefig(out_path_root + f"//scatter_{mode}.png")
        if show_plots:
            plt.show()
        else:
            plt.close()

        l,e = gather_error_dist(a,b)
        plt.title(f"Distribution of errors")
        plt.xlabel("Label")
        plt.ylabel("Error")
        plt.bar(l,e,width=1.0)
        plt.ylim(((0,30)))
        plt.savefig(out_path_root + f"//error_dist_{mode}.png")
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        p = pearsonr_calc(a,b)
        pearson_rec[mode] = p

    # avgHR = avgHR/hr_cnt

    # print("Basline performance on test set:")
    # print("MAE using just average of train set on test set:", MAE_avg(avgHR,test_dl,device))
    # print("RMSE using just average of train set on test set:", RMSE_avg(avgHR,test_dl,device))
    if(loss_fnc_name == 'supCR'):
        plt.plot(tlRec[:num_epochs])
        plt.plot(vlRec[:num_epochs])
        plt.title("SupCR Loss")
        plt.legend(["Train","Val"])
        plt.savefig(out_path_root + f"//loss_supcr.png")
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        plt.plot(tlRec[num_epochs:])
        plt.plot(vlRec[num_epochs:])
        plt.title("L1 Loss")
        plt.legend(["Train","Val"])
        plt.savefig(out_path_root + f"//loss_l1.png")
        if show_plots:
            plt.show()
        else:
            plt.close()
    else:
        plt.plot(tlRec)
        plt.plot(vlRec)
        plt.title("Loss")
        plt.legend(["Train","Val"])
        plt.savefig(out_path_root + f"//loss.png")
        if show_plots:
            plt.show()
        else:
            plt.close()


    # plt.plot(tlRec)
    # plt.plot(vlRec)
    # plt.title("Loss")
    # plt.legend(["Train","Val"])
    # plt.savefig(out_path_root + f"//tloss.png")
    # if show_plots:
    #     plt.show()
    # else:
    #     plt.close()
    plt.plot(tMAERec)
    plt.title("MAE")
    plt.savefig(out_path_root + f"//MAE.png")
    if show_plots:
        plt.show()
    else:
        plt.close()
    plt.plot(tRMSERec)
    plt.title("RMSE")
    plt.legend(["Train"])
    plt.savefig(out_path_root + f"//RMSE.png")
    if show_plots:
        plt.show()
    else:
        plt.close()

    print("writing to out file")
    out_info_file = out_path_root + "//info.txt"

    final_train_MAE, final_train_RMSE = MAE_RMSE_DL(train_dl,model,device)
    final_test_MAE, final_test_RMSE = MAE_RMSE_DL(test_dl,model,device)
    final_val_MAE, final_val_RMSE = MAE_RMSE_DL(test_dl,model,device)
    final_test_uniform_MAE, final_test_uniform_RMSE = MAE_RMSE_DL(test_uniform_dl,model,device)

    with open(out_info_file,"w") as file:
        file.write(f"num_epochs: {num_epochs}\n")
        file.write(f"num_epochs_reg: {num_epochs_reg}\n")
        file.write(f"bs: {bs}\n")
        file.write(f"lr: {lr}\n")
        file.write(f"t: {t}\n")
        file.write(f"seed: {seed}\n")
        file.write(f"optim: {optim}\n")
        file.write(f"Save best: {save_best}\n")
        file.write(f"Final Train MAE: {final_train_MAE}\n")
        file.write(f"Final Train RMSE: {final_train_RMSE}\n")
        file.write(f"Final Test MAE: {final_test_MAE}\n")
        file.write(f"Final Test RMSE: {final_test_RMSE}\n")
        file.write(f"Final Val MAE: {final_val_MAE}\n")
        file.write(f"Final Val RMSE: {final_val_RMSE}\n")
        file.write(f"Final Test Uniform MAE: {final_test_uniform_MAE}\n")
        file.write(f"Final Test Uniform RMSE: {final_test_uniform_RMSE}\n")
        for mode in pearson_rec.keys():
            file.write(f"Final {mode} Pearson R: {pearson_rec[mode]}\n")
        file.write(f"Model base: {model_base}\n")
        file.write(f"Loss function: {loss_fnc_name}\n")
    # print(MAE_DL(test_dl,model,device))
    # print(RMSE_DL(test_dl,model,device))
    torch.save(model.state_dict(), out_path_root + "//model.pt")

    for mode in acc_pred_rec.keys():
        df = pd.DataFrame(acc_pred_rec[mode])
        df.to_csv(out_path_root + f"//{mode}_preds.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o",type=str,help="Output folder",default=r"./") # assumed to already exist
    parser.add_argument("-l",type=str,help="Loss function", choices=['l1','supCR'], default='l1')
    parser.add_argument("-e",type=int,help="Num epochs", default=1)
    parser.add_argument("-ner",type=int,help="Num epochs reg",default=1)
    parser.add_argument("-bs",type=int,help="Batch size",default=32)
    parser.add_argument("-lr",type=float,help="Learning rate",default=0.001)
    parser.add_argument("-s",type=int,help="Seed",default=100)
    parser.add_argument("-t",type=float,help="Temperature param",default=1.0)
    parser.add_argument("-optim",type=str, choices=['adam','SGDR'], default='adam')
    parser.add_argument("-nw",type=int,default=0)
    parser.add_argument("-img_size",type=str, choices=['og','112', '56'], default='og')
    parser.add_argument("-sb",action="store_true",help="Save best model based on train loss for SupCR")
    
    args = parser.parse_args()
    print(args, flush=True)
    train(out_path_root=args.o, 
        loss_fnc_name=args.l,
        num_epochs=args.e,
        num_epochs_reg=args.ner,
        bs=args.bs,
        lr=args.lr,
        seed=args.s,
        t=args.t,
        optim=args.optim,
        nw=args.nw,
        img_size=args.img_size,
        model_base="",
        save_best=args.sb)
