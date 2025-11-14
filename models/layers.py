import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
class TensorNormalization(nn.Module):
    def __init__(self,mean, std):
        super(TensorNormalization, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std
    def forward(self,X):
        return normalizex(X,self.mean,self.std)
    def extra_repr(self) -> str:
        return 'mean=%s, std=%s'%(self.mean, self.std)

def normalizex(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    if mean.device != tensor.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    return tensor.sub(mean).div(std)

class MergeTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()

class ExpandTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        y_shape = [self.T, int(x_seq.shape[0]/self.T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

class RateBp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        mem = 0.
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem + x[t, ...]
            spike = ((mem - 1.) > 0).float()
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        out = torch.stack(spike_pot, dim=0)
        ctx.save_for_backward(out)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        T = out.shape[0]
        out = out.mean(0).unsqueeze(0)
        grad_input = grad_output * (out > 0).float()
        return grad_input

class LIFSpike(nn.Module):
    COLLECT: list[torch.Tensor] = []

    @classmethod
    def reset_collect(cls):
        cls.COLLECT.clear()

    @classmethod
    def take_collect(cls) -> list[torch.Tensor]:
        # return and clear (so next batch starts fresh)
        out, cls.COLLECT = cls.COLLECT, []
        return out
    def __init__(self, T, thresh=1.0, tau=1., gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.relu = nn.ReLU(inplace=True)
        self.ratebp = RateBp.apply
        self.mode = 'bptt'
        self.T = T
        #self.spike_calls_wide_cpu: list[torch.Tensor] = [] 
        print("Hello")
    def forward(self, x):
        #print("New call")
        #self.last_spikes_wide_cpu = None
        if self.mode == 'bptr' and self.T > 0:
            x = self.expand(x)
            x = self.ratebp(x)
            x = self.merge(x)
        elif self.T > 0:
            x = self.expand(x)
            mem = 0
            spike_pot = []
            for t in range(self.T):
                mem = mem * self.tau + x[t, ...]
                spike = self.act(mem - self.thresh, self.gama)
                mem = (1 - spike) * mem
                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)
            x_spikes=x.clone().detach()
            T, B = x_spikes.shape[0], x_spikes.shape[1]
            rest_shape = tuple(x_spikes.shape[2:])
            no_of_neurons=1
            for i in range(len(rest_shape)):
                no_of_neurons=no_of_neurons*rest_shape[i]#*rest_shape[1]*rest_shape[2]
            #print(no_of_neurons)
            wide = (
            x_spikes.reshape(T, B, no_of_neurons)# [T, B, N]
             .permute(1,0,2)
             .reshape(B, T * no_of_neurons)  # [B, T*N]
             .detach().cpu()
            )
            #print("*************************")
            #print(wide.shape)
            #print(T)
            #col_names = [f"t{t}_n{n}" for t in range(T) for n in range(no_of_neurons)]
            #df = pd.DataFrame(wide.numpy(), columns=col_names)

            # add sample IDs if provided
            #if sample_ids is None:
            #df.insert(0, "sample_id", range(B))
            #else:
                #df.insert(0, "sample_id", sample_ids)

            # save or append CSV
            #csv_path = "spikes_wide.csv"
            #header_needed = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
            #df.to_csv(csv_path, mode="a", index=False, header=header_needed)
            #print(B)
            #for t in range(T):
             #   for b in range(B):
              #      per_sample = x[t,b,:].detach().cpu().contiguous()  # [T, *rest]
                    #print(per_sample)
                # keep in RAM
                #self.spike_history.append(per_sample)
            #print(x[:, 64].detach().cpu().contiguous()) 
            #print(wide.shape)
            #self.spike_calls_wide_cpu.append(wide)
            #print(self.spike_calls_wide_cpu[0].shape)
            #print(len(self.spike_calls_wide_cpu))
            #print("*************************")
            LIFSpike.COLLECT.append(wide)
            #print(len(LIFSpike.COLLECT))
            #print(LIFSpike.COLLECT[len(LIFSpike.COLLECT)-1].shape)
            #print("****************")
            x = self.merge(x)
        else:
            x = self.relu(x)
        return x

def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(T, 1, 1, 1, 1)
    return x

class ConvexCombination(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.comb = nn.Parameter(torch.ones(n) / n)

    def forward(self, *args):
        assert(len(args) == self.n)
        out = 0.
        for i in range(self.n):
            out += args[i] * self.comb[i]
        return out
