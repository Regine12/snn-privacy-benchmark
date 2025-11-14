from cv2 import mean
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from models.layers import *
import random
import time
from models.layers import LIFSpike
import pyarrow as pa
import pyarrow.parquet as pq

def iter_lif_layers(model):
    #print("iter")
    """Yield (name, module) for every LIFSpike in the model. Handles DP/DDP."""
    root = model.module if hasattr(model, "module") else model
    seen = set()
    for name, m in root.named_modules():
        if isinstance(m, LIFSpike) and id(m) not in seen:
            seen.add(id(m))
            yield (name or f"lif_{len(seen)-1}"), m 
            return True

def _reset_spike_buffers(model):
    """Clear per-layer call lists BEFORE each model forward."""
    for _, m in iter_lif_layers(model):
        m.spike_calls_wide_cpu = []

def arsnn_reg(net, beta):
    l = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            weight = m.weight
            if isinstance(m, nn.Conv2d):
                weight = weight.view(weight.shape[0], -1)
            sum_1 = torch.sum(F.relu(0 - weight), dim=1)
            sum_2 = torch.sum(F.relu(weight), dim=1)
            l += (torch.max(sum_1) + torch.max(sum_2)) * beta
    return l

def train(model, device, train_loader, criterion, optimizer, T, atk, beta, parseval=False):
    running_loss = 0
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    for i, (images, labels) in enumerate((train_loader)):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)

        if atk is not None:
            atk.set_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            images = atk(images, labels)
        
        if T > 0:
            outputs = model(images).mean(0)
        else:
            outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        if parseval:
            orthogonal_retraction(model, beta)
            convex_constraint(model)
    
        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total

def val(model, test_loader, device, T, atk=None):
    out_dir = "spikes_parquet"           # put this near the top of val()
    os.makedirs(out_dir, exist_ok=True)
    correct = 0
    total = 0
    sample_offset = 0
    model.eval()
    spikes_csv="spikes_wide_correct_all_layers.csv"
    for batch_idx, (inputs, targets) in enumerate((test_loader)):
        inputs = inputs.to(device)
        #_reset_spike_buffers(model)
        LIFSpike.reset_collect()
        if atk is not None:
            atk.set_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            inputs = atk(inputs, targets.to(device))
            model.set_simulation_time(T)
        with torch.no_grad():
            if T > 0:
                outputs = model(inputs).mean(0)
            else:
                outputs = model(inputs)
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
    #final_acc = 100 * correct / total
    #mask = (predicted==targets)
        B = float(targets.size(0))
        blocks = LIFSpike.take_collect()
        print(len(blocks))
        if blocks:
            # build columns per block
            cols = []
            for k, blk in enumerate(blocks):
                TN = blk.shape[1]
                if TN % T != 0:
                    raise ValueError(f"Block {k}: TN={TN} not divisible by T={T}")
                N = TN // T
                cols.extend([f"blk{k}_t{t}_n{n}" for t in range(T) for n in range(N)])
            wide_all = torch.cat(blocks, dim=1)
            mask=(predicted==targets)
            if mask.any():
                idx = mask.nonzero(as_tuple=False).squeeze(1)
                wide_sel = wide_all.index_select(0, idx).numpy()
                ids = (torch.arange(B) + sample_offset).index_select(0, idx).numpy()
                y_true_sel = targets.index_select(0, idx).numpy()
                y_pred_sel = predicted.index_select(0, idx).numpy()

                #df = pd.DataFrame(wide_sel, columns=cols)
                #df.insert(0, "y_pred", y_pred_sel)
                #df.insert(0, "y_true", y_true_sel)
                #df.insert(0, "sample_id", ids)
                TN = wide_sel.shape[1]
                # FixedSizeListArray requires a flat values buffer
                values_flat = pa.array(wide_sel.reshape(-1), type=pa.uint8())
                spikes_col  = pa.FixedSizeListArray.from_arrays(values_flat, TN)

                table = pa.Table.from_arrays(
                    [
                        pa.array(ids,      type=pa.int64()),
                        pa.array(y_true_sel, type=pa.int32()),
                        pa.array(y_pred_sel, type=pa.int32()),
                        spikes_col
                    ],
                    names=["sample_id", "y_true", "y_pred", "spikes"],
                )

                pq.write_table(
                    table,
                    os.path.join(out_dir, f"batch={batch_idx:05d}.parquet"),
                    compression="zstd"  # or "snappy"
                )
                #header = not os.path.exists(spikes_csv) or os.path.getsize(spikes_csv) == 0
                #df.to_csv(spikes_csv, mode="a", index=False, header=header)
                #df.to_parquet(os.path.join(out_dir, f"batch={batch_idx:05d}.parquet"),index=False, engine="pyarrow")

        sample_offset += B

    final_acc = 100 * correct / total
    #print(df)
    return final_acc



def orthogonal_retraction(model, beta=0.002):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if isinstance(module, nn.Conv2d):
                    weight_ = module.weight.data
                    sz = weight_.shape
                    weight_ = weight_.reshape(sz[0],-1)
                    rows = list(range(module.weight.data.shape[0]))
                elif isinstance(module, nn.Linear):
                    if module.weight.data.shape[0] < 200: # set a sample threshold for row number
                        weight_ = module.weight.data
                        sz = weight_.shape
                        weight_ = weight_.reshape(sz[0], -1)
                        rows = list(range(module.weight.data.shape[0]))
                    else:
                        rand_rows = np.random.permutation(module.weight.data.shape[0])
                        rows = rand_rows[: int(module.weight.data.shape[0] * 0.3)]
                        weight_ = module.weight.data[rows,:]
                        sz = weight_.shape
                module.weight.data[rows,:] = ((1 + beta) * weight_ - beta * weight_.matmul(weight_.t()).matmul(weight_)).reshape(sz)


def convex_constraint(model):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, ConvexCombination):
                comb = module.comb.data
                alpha = torch.sort(comb, descending=True)[0]
                k = 1
                for j in range(1,module.n+1):
                    if (1 + j * alpha[j-1]) > torch.sum(alpha[:j]):
                        k = j
                    else:
                        break
                gamma = (torch.sum(alpha[:k]) - 1)/k
                module.comb.data -= gamma
                torch.relu_(module.comb.data)
