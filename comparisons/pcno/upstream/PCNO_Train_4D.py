import gc, glob, json, random, torch, csv, os, queue, threading, math
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from PCNO_Model_4D import EnhancedP_T_Net, denormalize, create_smooth_weight_map
from PCNO_Model_4D import physical_loss, compute_loss, weighted_mse, compute_enhanced_gradient_loss


class TrainConfig():
    @staticmethod
    def set_seed(seed=7):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def stats_to_device(stats, device):
        out = {}
        for k, v in stats.items():
            if isinstance(v, (list, tuple)):
                out[k] = torch.tensor(v, device=device, dtype=torch.float32)
            elif isinstance(v, (int, float)):
                out[k] = torch.tensor(v, device=device, dtype=torch.float32)
            else:
                try:
                    out[k] = torch.as_tensor(v, device=device, dtype=torch.float32)
                except Exception:
                    out[k] = v
        return out

    @staticmethod
    def async_prefetch(paths):
        q = queue.Queue(maxsize=2)

        def loader():
            for p in paths:
                chunk = torch.load(p, map_location="cpu", weights_only=False)
                q.put(chunk)
            q.put(None)

        threading.Thread(target=loader, daemon=True).start()
        return q

class ModelConfig():
    @staticmethod
    def lr_lambda(base_lr, warmup_start_lr, warmup_end_lr, lr_mid, lr_min, warmup_steps, plateau_steps, cosine1_steps, cosine2_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                lr_now = warmup_start_lr + (warmup_end_lr - warmup_start_lr) * (current_step / max(1, warmup_steps))
                return lr_now / base_lr

            elif current_step < warmup_steps + plateau_steps:
                return warmup_end_lr / base_lr

            elif current_step < warmup_steps + plateau_steps + cosine1_steps:
                progress = (current_step - warmup_steps - plateau_steps) / cosine1_steps
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                lr_now = lr_mid + (warmup_end_lr - lr_mid) * cosine_decay
                return lr_now / base_lr

            else:
                progress = (current_step - warmup_steps - plateau_steps - cosine1_steps) / max(1, cosine2_steps)
                progress = min(max(progress, 0.0), 1.0)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                lr_now = lr_min + (lr_mid - lr_min) * cosine_decay
                return lr_now / base_lr

        return lr_lambda

    @staticmethod
    def scale(base, secondary):
        ratio = base.item() / (secondary.item() + 1e-8)
        return max(0.1, min(5.0, ratio))

    @staticmethod
    def loss_weights(ep, mse, w, g):
        if ep < 5:
            return dict(w=(1.0, 0.0, 0.0), w_phys=0.0, w_task=0.0)
        if ep < 20:
            w_w = 0.9 * ModelConfig.scale(mse, w)
            return dict(w=(1.0, 0.0, w_w), w_phys=0.0, w_task=0.0)
        if ep < 35:
            w_g = 0.3 * ModelConfig.scale(mse, g)
            w_w = 0.9 * ModelConfig.scale(mse, w)
            return dict(w=(0.9, w_g, w_w), w_phys=0.0, w_task=0.0)
        if ep < 50:
            w_g = 0.5 * ModelConfig.scale(mse, g)
            w_w = 1.0 * ModelConfig.scale(mse, w)
            return dict(w=(0.9, w_g, w_w), w_phys=0.0, w_task=0.0)
        if ep < 65:
            w_g = 0.7 * ModelConfig.scale(mse, g)
            w_w = 1.0 * ModelConfig.scale(mse, w)
            return dict(w=(0.8, w_g, w_w), w_phys=0.1, w_task=0.0)
        if ep < 80:
            w_g = 0.9 * ModelConfig.scale(mse, g)
            w_w = 1.1 * ModelConfig.scale(mse, w)
            return dict(w=(0.8, w_g, w_w), w_phys=0.2, w_task=0.0)
        if ep < 95:
            w_g = 1.1 * ModelConfig.scale(mse, g)
            w_w = 1.1 * ModelConfig.scale(mse, w)
            return dict(w=(0.7, w_g, w_w), w_phys=0.4, w_task=0.1)
        if ep < 110:
            w_g = 1.3 * ModelConfig.scale(mse, g)
            w_w = 1.3 * ModelConfig.scale(mse, w)
            return dict(w=(0.7, w_g, w_w), w_phys=0.6, w_task=0.2)

        w_g = 1.5 * ModelConfig.scale(mse, g)
        w_w = 1.5 * ModelConfig.scale(mse, w)

        return dict(w=(0.6, w_g, w_w), w_phys=0.8, w_task=0.3)

    @staticmethod
    def loss_weights_finetune():
        w_mse = 0.8
        w_g   = 0.4
        w_w   = 0.6
        return dict(w=(w_mse, w_g, w_w), w_phys=0.5, w_task=0.5)

    @staticmethod
    def decode_inputs(input_i, g_i, s):
        Pini = denormalize(input_i[..., 0], s['Pini_mean'], s['Pini_std'])
        Tini = denormalize(input_i[..., 1], s['Tini_mean'], s['Tini_std'])
        q_inj = torch.where(input_i[..., 3] != 0, input_i[..., 3] * s["qinj_std"] + s["qinj_mean"], torch.zeros_like(input_i[..., 1]))
        T_inj = torch.where(input_i[..., 4] != 0, input_i[..., 4] * s["Tinj_std"] + s["Tinj_mean"], torch.zeros_like(input_i[..., 2]))
        pwf = torch.where(input_i[..., 5] != 0, input_i[..., 5] * s["ppro_std"] + s["ppro_mean"], torch.zeros_like(input_i[..., 3]))
        k = denormalize(input_i[..., 6], s['perm_mean'], s['perm_std'])
        phi = denormalize(input_i[..., 7], s['poro_mean'], s['poro_std'])

        d = denormalize(g_i[..., 0], s['global_mean'][0], s['global_std'][0])
        Cp_r = denormalize(g_i[..., 1], s['global_mean'][1], s['global_std'][1])
        lam_r = denormalize(g_i[..., 2], s['global_mean'][2], s['global_std'][2])
        dz = denormalize(g_i[..., 3], s['global_mean'][3], s['global_std'][3])

        return Pini, Tini, q_inj, T_inj, pwf, k, phi, d, Cp_r, lam_r, dz

def train(model, all_paths, optimizer, scheduler, epochs, device, t_mean, t_std, p_mean, p_std, stats_dev, batch_size, train_mode):

    print(f"\n------ Model Training Start ------\n")
    print('*'*100)
    print('Note: For functional testing, only a small subset of the data is provided here; therefore, the losses reported from this run are not representative of the actual training performance.')
    print('*'*100)

    for ep in range(1, epochs + 1):
        # k-fold
        random.shuffle(all_paths)
        train_paths = all_paths[:-1]
        val_paths = all_paths[-1:]

        # Train
        model.train()
        q = TrainConfig.async_prefetch(train_paths)

        total_loss = 0
        total_mse_loss = total_grad_loss = total_weighted_loss = 0
        total_phys_loss = total_task_loss = 0
        count = 0

        idx = 0
        while True:
            chunk = q.get()
            if chunk is None:
                break
            idx += 1

            x = chunk['spatial_params'].to(device)
            pres = chunk['pres'].to(device)
            temp = chunk['temp'].to(device)
            global_param = chunk['global_params'].to(device)
            Twh_all = chunk['Temp_wh']
            Hwh_all = chunk['Heat_wh']
            Pinj_all = chunk['P_inj']
            B = x.shape[0]

            for i in range(0, B, batch_size):
                input_i = x[i:i + batch_size]
                g_i = global_param[i:i + batch_size]
                Twh_t = Twh_all[i:i + batch_size]
                Hwh_t = Hwh_all[i:i + batch_size]
                Pinj_t = Pinj_all[i:i + batch_size]
                dis_inj = input_i[..., 8]
                dis_prod = input_i[..., 9]

                if train_mode == 'temp':
                    true_value = temp[i:i + batch_size]
                    clamp_min, clamp_max = -2.0, 4.0
                    task_type = 'temperature'

                if train_mode == 'pres':
                    true_value = pres[i:i + batch_size]
                    clamp_min, clamp_max = -3.0, 9.0
                    task_type = 'pressure'

                optimizer.zero_grad()
                pred_value = model(input_i, g_i)
                pred_value = torch.nan_to_num(pred_value, nan=0.0, posinf=1e4, neginf=-1e4)

                if train_mode == 'temp':
                    p_raw = pres[i:i + batch_size] * p_std + p_mean
                    pred_T_raw = pred_value * t_std + t_mean
                    T_true_raw = true_value * t_std + t_mean

                if train_mode == 'pres':
                    pred_p_raw = pred_value * p_std + p_mean
                    p_true_raw = true_value * p_std + p_mean
                    T_raw = temp[i:i + batch_size] * t_std + t_mean

                pred_value = torch.clamp(pred_value, clamp_min, clamp_max)
                mse_loss = F.mse_loss(pred_value, true_value)
                grad_loss = compute_enhanced_gradient_loss(pred_value, true_value)
                w_map = create_smooth_weight_map(pred_value, true_value, dis_inj, dis_prod, task=task_type)
                weighted_loss = weighted_mse(pred_value, true_value, w_map)

                Pini, Tini, q_inj, T_inj, pwf, k, phi, d, Cp_r, lam_r, dz = ModelConfig.decode_inputs(input_i, g_i, stats_dev)

                loss_func = physical_loss(T_i=Tini, p_i=Pini, Cp_r=Cp_r, lam_r=lam_r, dz=dz, q_inj=q_inj, T_inj=T_inj, pwf=pwf, k=k, phi=phi, depth=d)
                if train_mode == 'temp':
                    mse_mass, mse_energy, re_Twh, re_Ewh, re_Pinj = compute_loss(p_raw, pred_T_raw, loss_func, Twh_t, Hwh_t, Pinj_t)
                    loss_phys = mse_energy

                if train_mode == 'pres':
                    mse_mass, mse_energy, re_Twh, re_Ewh, re_Pinj = compute_loss(pred_p_raw, T_raw, loss_func, Twh_t, Hwh_t, Pinj_t)
                    loss_phys = mse_mass

                loss_task = re_Twh + re_Ewh + re_Pinj

                ws = ModelConfig.loss_weights(ep, mse_loss, weighted_loss, grad_loss)
                (w_mse, w_g, w_w) = ws["w"]
                w_phys = ws["w_phys"]
                w_task = ws['w_task']
                loss = w_mse * mse_loss + w_w * weighted_loss + w_g * grad_loss + w_phys * loss_phys + w_task * loss_task

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]

                total_loss += loss.item()
                total_mse_loss += mse_loss.item()
                total_grad_loss += grad_loss.item()
                total_weighted_loss += weighted_loss.item()
                total_phys_loss += loss_phys.item()
                total_task_loss += loss_task.item()
                count += batch_size

        del x, pres, temp, global_param, chunk
        torch.cuda.empty_cache()
        gc.collect()

        avg_total_loss = total_loss / count
        avg_mse_loss = total_mse_loss / count
        avg_grad_loss = total_grad_loss / count
        avg_weighted_loss = total_weighted_loss / count
        avg_phys_loss = total_phys_loss / count
        avg_task_loss = total_task_loss / count

        print(f'\n{ep:02d} - Total={avg_total_loss:.4f} | mse={avg_mse_loss:.4f} | g_mse={avg_grad_loss:.4f}'
              f' | w_mse={avg_weighted_loss:.4f} | Phys={avg_phys_loss:.4f} | Task={avg_task_loss:.4f}')

        ## Training Log
        # csv_path = f"/data/train_log_{train_mode}.csv"
        # os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        # with open(csv_path, "w", newline="", encoding="utf-8") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["epoch", "total", "mse", "g_mse", "w_mse", "phys", "task"])

        # if ep % 5 == 0:
        #     with open(csv_path, "a", newline="", encoding="utf-8") as f:
        #         writer = csv.writer(f)
        #         writer.writerow([ep, avg_total_loss, avg_mse_loss, avg_grad_loss, avg_weighted_loss, avg_phys_loss, avg_task_loss])

        # Validation
        if ep >= 30 and ep % 5 == 0:
            model.eval()
            val_total_loss = 0.0
            val_count = 0.0

            with (torch.no_grad()):
                for path in val_paths:
                    chunk = torch.load(path, map_location="cpu", weights_only=False)
                    x = chunk['spatial_params'].to(device)
                    pres = chunk['pres'].to(device)
                    temp = chunk['temp'].to(device)
                    global_param = chunk['global_params'].to(device)
                    Twh_all = chunk['Temp_wh']
                    Hwh_all = chunk['Heat_wh']
                    Pinj_all = chunk['P_inj']
                    B = x.shape[0]

                    for i in range(0, B, batch_size):
                        input_i = x[i:i + batch_size]
                        g_i = global_param[i:i + batch_size]
                        Twh_t = Twh_all[i:i + batch_size]
                        Hwh_t = Hwh_all[i:i + batch_size]
                        Pinj_t = Pinj_all[i:i + batch_size]
                        dis_inj = input_i[..., 8]
                        dis_prod = input_i[..., 9]

                        if train_mode == 'temp':
                            true_value = temp[i:i + batch_size]
                            clamp_min, clamp_max = -2.0, 4.0
                            task_type = 'temperature'
                        if train_mode == 'pres':
                            true_value = pres[i:i + batch_size]
                            clamp_min, clamp_max = -3.0, 9.0
                            task_type = 'pressure'

                        pred_value = model(input_i, g_i)
                        pred_value = torch.nan_to_num(pred_value, nan=0.0, posinf=1e4, neginf=-1e4)

                        if train_mode == 'temp':
                            p_raw = pres[i:i + batch_size] * p_std + p_mean
                            pred_T_raw = pred_value * t_std + t_mean
                            T_true_raw = true_value * t_std + t_mean

                        if train_mode == 'pres':
                            pred_p_raw = pred_value * p_std + p_mean
                            p_true_raw = true_value * p_std + p_mean
                            T_raw = temp[i:i + batch_size] * t_std + t_mean

                        pred_value = torch.clamp(pred_value, clamp_min, clamp_max)
                        mse_loss = F.mse_loss(pred_value, true_value)
                        grad_loss = compute_enhanced_gradient_loss(pred_value, true_value)
                        w_map = create_smooth_weight_map(pred_value, true_value, dis_inj, dis_prod, task=task_type)
                        weighted_loss = weighted_mse(pred_value, true_value, w_map)

                        Pini, Tini, q_inj, T_inj, pwf, k, phi, d, Cp_r, lam_r, dz = ModelConfig.decode_inputs(input_i, g_i, stats_dev)

                        loss_func = physical_loss(T_i=Tini, p_i=Pini, Cp_r=Cp_r, lam_r=lam_r, dz=dz, q_inj=q_inj,
                                                  T_inj=T_inj, pwf=pwf, k=k, phi=phi, depth=d)
                        if train_mode == 'temp':
                            mse_mass, mse_energy, re_Twh, re_Ewh, re_Pinj = compute_loss(p_raw, pred_T_raw, loss_func, Twh_t, Hwh_t, Pinj_t)
                            loss_phys = mse_energy

                        if train_mode == 'pres':
                            mse_mass, mse_energy, re_Twh, re_Ewh, re_Pinj = compute_loss(pred_p_raw, T_raw, loss_func, Twh_t, Hwh_t, Pinj_t)
                            loss_phys = mse_mass

                        loss_task = re_Twh + re_Ewh + re_Pinj

                        ws = ModelConfig.loss_weights(ep, mse_loss, weighted_loss, grad_loss)
                        (w_mse, w_g, w_w) = ws["w"]
                        w_phys = ws["w_phys"]
                        w_task = ws['w_task']
                        loss = w_mse * mse_loss + w_w * weighted_loss + w_g * grad_loss + w_phys * loss_phys + w_task * loss_task

                        val_total_loss += loss.item()
                        val_count += batch_size

                del x, pres, temp, global_param, chunk
                torch.cuda.empty_cache()
                gc.collect()

            avg_val_loss = val_total_loss / val_count
            print(f"{ep:02d} - Val Loss={avg_val_loss:.4f}")

            # SaveModel / Checkpoint
            if ep > 50 and ep % 5 == 0:
                os.makedirs("/data", exist_ok=True)
                path_ck = f"/data/MODEL_{train_mode}_ep{ep}_val{avg_val_loss:.4f}.pth"
                torch.save({
                    "epoch": ep,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, path_ck)
                print(f"## Checkpoint saved to {path_ck}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TrainConfig.set_seed(7)

    with open('/data/stats.json', "r") as f:
        stats = json.load(f)

    stats_dev = TrainConfig.stats_to_device(stats, device)
    t_mean, t_std = stats_dev["temp_mean"], stats_dev["temp_std"]
    p_mean, p_std = stats_dev["pres_mean"], stats_dev["pres_std"]

    all_chunks = sorted(glob.glob("/data/chunk_*.pt"))

    ## The model used for functional testing
    model = EnhancedP_T_Net(modes1=1, modes2=1, modes3=1, modes4=1, width_p=8, width_T=8, train_mode='pres').to(device)
    ## The model used for actual PCNO training
    # model = EnhancedP_T_Net(modes1=14, modes2=14, modes3=3, modes4=10, width_p=64, width_T=56, train_mode='pres').to device)

    torch.cuda.empty_cache()
    gc.collect()

    base_lr = 1e-3
    epochs = 250
    warmup_epochs = 15
    plateau_epochs = 5
    cosine1_epochs = 150
    cosine2_epochs = epochs - warmup_epochs - plateau_epochs - cosine1_epochs
    train_chunks_per_epoch = len(all_chunks) - 1
    tmp = torch.load(all_chunks[0], weights_only=False)
    samples_per_chunk = tmp["spatial_params"].shape[0]
    batch_size = 1

    steps_per_epoch = train_chunks_per_epoch * math.ceil(samples_per_chunk / batch_size)

    warmup_steps = warmup_epochs * steps_per_epoch
    plateau_steps = plateau_epochs * steps_per_epoch
    cosine1_steps = cosine1_epochs * steps_per_epoch
    cosine2_steps = cosine2_epochs * steps_per_epoch

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    lr_lambda = ModelConfig.lr_lambda(base_lr=1e-3, warmup_start_lr=9e-4, warmup_end_lr=1e-3, lr_mid=1e-5, lr_min=1e-6,
                                      warmup_steps=warmup_steps, plateau_steps=plateau_steps,
                                      cosine1_steps=cosine1_steps, cosine2_steps=cosine2_steps)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    ## Pres Model Training
    train(model, all_chunks, optimizer, scheduler, t_mean=t_mean, t_std=t_std, p_mean=p_mean, p_std=p_std,
          epochs=epochs, stats_dev=stats_dev, device=device, batch_size=batch_size, train_mode='pres')
    ## Temp Model Training
    # train(model, all_chunks, optimizer, scheduler, t_mean=t_mean, t_std=t_std, p_mean=p_mean, p_std=p_std,
    #       epochs=epochs, stats_dev=stats_dev, device=device, batch_size=batch_size, train_mode='temp')

if __name__ == "__main__":
    main()