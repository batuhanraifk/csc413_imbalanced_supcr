import torch
from torch.nn import MSELoss

class SupCRLoss:
    def __init__(self, t = 1, device = 'cpu'):
        # self.dist = MSELoss()
        self.l2dist = MSELoss()
        self.t = t
        self.device = device

    def dist(self, l1, l2): # L1 distance
        return abs(l1-l2)

    def custom_dist(self, l1, l2):
        return torch.maximum(l1, l2) / torch.minimum(l1, l2)

    def sim(self, v1, v2): # Neg L2
        return -1 * torch.norm(v1 - v2)

    def sim_vec_pt(self, v1, m2):
        return -1 * torch.norm(m2 - v1, dim=1)

    def vectorized_supcr_pt(self, batch, labels, t = 1):
        N = len(batch)
        loss = 0
        sum1 = 0
        sum2 = 0
        sum3 = 0
        for i in range(N):
            # print(f"Anchor Index: {i} xy: {batch[i]}")
            sum2 = 0
            ones = torch.ones(N).to(self.device)
            ones[i] = 0
            for j in range(N):
                if j == i:
                    continue
                threshold = self.custom_dist(labels[i], labels[j])
                dist_vec = self.custom_dist(labels,labels[i])
                past_thresh = dist_vec >= threshold
                sims = torch.exp(self.sim_vec_pt(batch[i],batch) / t)
                sum3 = torch.sum(ones *past_thresh * sims)
                
                sum2 += torch.log(torch.exp(self.sim(batch[i], batch[j]) / t) / sum3)
            
            sum1 += ((1 / (N-1)) * sum2)
        loss = (-1/(N)) * sum1
        return loss

    def supcr_v2_pt(self, batch, labels, t = 1):
        N = len(batch)
        
        dists = torch.cdist(labels.unsqueeze(1), labels.unsqueeze(1), p=1)

        sims = torch.cdist(batch, batch) * (-1)
        sims = torch.exp(sims / self.t)
        
        loss = 0
        sum1 = 0
        sum2 = 0
        sum3 = 0

        for i in range(N):
            # print(f"Anchor Index: {i} xy: {batch[i]}")
            sum2 = 0
            for j in range(N):
                if j == i:
                    continue
                threshold = dists[i][j]
                past_thresh = dists[i] >= threshold
                past_thresh[i] = False

                sum3 = torch.sum(past_thresh * sims[i])
                if (sum3.item() == 0): # checking if denominator is 0, skipping if true
                    print("Sum3 is 0 - skipping",flush=True)
                    break
                if (sims[i][j] == 0): # checking if numerator is 0, skipping if true
                    print("sims[i][j] is 0 - skipping",flush=True)
                    break

                sum2 += torch.log(sims[i][j] / sum3)
            
            sum1 += ((1 / (N-1)) * sum2)
        loss = (-1/(N)) * sum1
            
        return loss

def custom_loss(pred, true):
    N = pred.shape[0]
    element_wise = torch.maximum(pred, true) / torch.minimum(pred, true)
    return torch.sum(element_wise) / N


def l1_loss(pred, true):
    N = pred.shape[0]
    return torch.sum(torch.abs(pred - true)) / N