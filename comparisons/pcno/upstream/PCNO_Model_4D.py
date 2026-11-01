import torch, math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from iapws import IAPWS97

torch.manual_seed(0)
np.random.seed(0)

### 4D_PCNO
class SpectralConv4d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, modes4):
        super(SpectralConv4d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4
        self.scale = (1 / (in_channels * out_channels))

        def cw():
            return self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat)

        self.weights = nn.ParameterList([nn.Parameter(cw()) for _ in range(8)])

    def compl_mul4d(self, input, weights):
        return torch.einsum("bixyzt,ioxyzt->boxyzt", input, weights)

    def forward(self, x):
        B = x.shape[0]
        x = x.contiguous().to(dtype=torch.float32)
        x_ft = torch.fft.rfftn(x, dim=[-4, -3, -2, -1])
        out_ft = torch.zeros(B, self.out_channels, x.size(-4), x.size(-3), x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)

        idx = lambda i: (slice(None), slice(None),
                         slice(None, self.modes1) if i & 1 == 0 else slice(-self.modes1, None),
                         slice(None, self.modes2) if i & 2 == 0 else slice(-self.modes2, None),
                         slice(None, self.modes3) if i & 4 == 0 else slice(-self.modes3, None),
                         slice(None, self.modes4))

        for i in range(8):
            out_ft[idx(i)] = self.compl_mul4d(x_ft[idx(i)], self.weights[i])

        x = torch.fft.irfftn(out_ft, s=x.shape[-4:])
        return x

class UNet3D_Lite(nn.Module):
    def __init__(self, in_channels, base_channels=32):
        super(UNet3D_Lite, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=base_channels),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=base_channels),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.enc2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, 3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=base_channels * 2),
            nn.ReLU(),
            nn.Conv3d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=base_channels * 2),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 1))  # 只pool X和Y

        self.bottom = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=base_channels * 4),
            nn.ReLU(),
            nn.Conv3d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=base_channels * 2),
            nn.ReLU()
        )
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels * 2, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.dec1 = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=base_channels * 2),
            nn.ReLU()
        )
        self.up2 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.dec2 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels, 3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=base_channels),
            nn.ReLU()
        )
        self.out_conv = nn.Conv3d(base_channels, in_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottom(self.pool2(e2))

        up1_crop = self.center_crop(self.up1(b), e2)
        d1 = self.dec1(torch.cat([e2, up1_crop], dim=1))

        up2_crop = self.center_crop(self.up2(d1), e1)
        d2 = self.dec2(torch.cat([e1, up2_crop], dim=1))

        out = self.out_conv(d2)
        return out

    def center_crop(self, source, target):
        _, _, h, w, d = target.shape
        src_h, src_w, src_d = source.shape[2:]
        crop_h = (src_h - h) // 2
        crop_w = (src_w - w) // 2
        crop_d = (src_d - d) // 2
        return source[:, :, crop_h:crop_h + h, crop_w:crop_w + w, crop_d:crop_d + d]

class Block4dWithUNet(nn.Module):
    def __init__(self, width, width2, modes1, modes2, modes3, modes4, out_dim):
        super(Block4dWithUNet, self).__init__()
        self.width = width
        self.width2 = width2
        self.out_dim = out_dim
        self.padding = 8
        self.Z_padding = 2

        self.conv_layers = nn.ModuleList([SpectralConv4d(width, width, modes1, modes2, modes3, modes4) for _ in range(4)])
        self.linear_layers = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(4)])
        self.norm_layers = nn.ModuleList([nn.GroupNorm(num_groups=8, num_channels=width) for _ in range(4)])

        self.unet = UNet3D_Lite(width, base_channels=width // 2)

        self.alpha = nn.Parameter(torch.ones(1) * 0.1)

        self.fc1 = nn.Linear(width, width2)
        self.fc2 = nn.Linear(width2, out_dim)

    def forward(self, x):
        B, C, X, Y, Z, T = x.shape

        x_unet = x.permute(0, 5, 1, 2, 3, 4).contiguous()
        x_unet = x_unet.view(B * T, C, X, Y, Z)
        unet_out = self.unet(x_unet)
        unet_out = unet_out.view(B, T, C, X, Y, Z).permute(0, 2, 3, 4, 5, 1).contiguous()

        for conv, lin, norm in zip(self.conv_layers, self.linear_layers, self.norm_layers):
            x1 = conv(x)
            x2 = lin(x.view(B, self.width, -1)).view(B, self.width, X, Y, Z, T)
            gated_unet_out = self.alpha * unet_out
            x = x1 + x2 + gated_unet_out
            x = norm(x)
            x = F.gelu(x)

        x = x[:, :, self.padding * 2:-self.padding * 2,
            self.padding * 2:-self.padding * 2,
            self.Z_padding:-self.Z_padding,
            self.padding:-self.padding]
        x = x.permute(0, 2, 3, 4, 5, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class FNO4dUNet(nn.Module):
    def __init__(self, modes1, modes2, modes3, modes4, width, in_dim):
        super(FNO4dUNet, self).__init__()
        self.padding = 8
        self.fc0 = nn.Linear(in_dim, width)
        self.conv = Block4dWithUNet(width, width * 4, modes1, modes2, modes3, modes4, out_dim=1)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 5, 1, 2, 3, 4)
        x = F.pad(x, [self.padding] * 2 + [self.padding * 2] * 2 + [self.padding * 2] * 2 + [self.padding] * 2)
        x = self.conv(x)
        return x.squeeze(-1)

class GlobalFeatureFusion(nn.Module):
    def __init__(self, global_dim, target_channels):
        super(GlobalFeatureFusion, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(global_dim, 64),
            nn.ReLU(),
            nn.Linear(64, target_channels)
        )

    def forward(self, global_params, shape_to_expand):
        B, C, X, Y, Z, T = shape_to_expand
        fused = self.mlp(global_params)  # [B, C]
        return fused.view(B, C, 1, 1, 1, 1).expand(B, C, X, Y, Z, T)

class FNO4dUNet_WithGlobalFusion(nn.Module):
    def __init__(self, modes1, modes2, modes3, modes4, width, in_dim, global_dim):
        super(FNO4dUNet_WithGlobalFusion, self).__init__()
        self.fno_core = FNO4dUNet(modes1, modes2, modes3, modes4, width, in_dim)
        self.global_fusion = GlobalFeatureFusion(global_dim, width)
        self.fusion_conv1d = nn.Conv1d(2 * width, width, kernel_size=1)
        self.norm_fusion = nn.GroupNorm(num_groups=8, num_channels=width)

    def forward(self, x, global_params):
        x = self.fno_core.fc0(x)
        x = x.permute(0, 5, 1, 2, 3, 4).contiguous()
        x = F.pad(x, [8, 8, 2, 2, 16, 16, 16, 16])
        global_feat = self.global_fusion(global_params, x.shape)
        x = torch.cat([x, global_feat], dim=1)

        B, C2, Xp, Yp, Zp, Tp = x.shape
        L = Xp * Yp * Zp * Tp
        x = x.reshape(B, C2, L)
        x = self.fusion_conv1d(x)
        x = self.norm_fusion(x)
        x = x.reshape(B, -1, Xp, Yp, Zp, Tp).contiguous()
        x = self.fno_core.conv(x)
        return x.squeeze(-1)

class EnhancedP_T_Net(nn.Module):
    def __init__(self, modes1, modes2, modes3, modes4, width_p, width_T, in_dim=14, global_dim=4, train_mode='pres'):
        super(EnhancedP_T_Net, self).__init__()

        assert train_mode in ['pres', 'temp'], \
            "train_mode must be either 'pres' or 'temp'"

        self.train_mode = train_mode
        self.pressure_net = None
        self.temperature_net = None

        if train_mode == 'pres':
            self.pressure_net = FNO4dUNet_WithGlobalFusion(modes1, modes2, modes3, modes4, width_p, in_dim, global_dim)
        elif train_mode == 'temp':
            self.temperature_net = FNO4dUNet_WithGlobalFusion(modes1, modes2, modes3, modes4, width_T, in_dim, global_dim)

    def forward(self, x, global_params):
        if self.train_mode == 'pres':
            return self.pressure_net(x, global_params)
        elif self.train_mode == 'temp':
            return self.temperature_net(x, global_params)


### Data_Preprocessing
def denormalize(tensor, mean, std):
    return tensor * std + mean


### Data_driven Loss
def compute_enhanced_gradient_loss(pred, target, eps=1e-4, max_rel=80.0):
    def _relative_grad_loss(pred_grad, target_grad, eps=1e-4, max_rel=None):
        diff2 = (pred_grad - target_grad) ** 2
        den = target_grad ** 2 + eps
        rel = diff2 / den

        if max_rel is not None:
            rel = torch.clamp(rel, 0.0, max_rel)

        rel = torch.log1p(rel)
        w = torch.abs(target_grad)
        w_mean = w.mean(dim=(1, 2, 3, 4), keepdim=True) + eps
        w_norm = w / w_mean
        loss = (w_norm * rel).mean()
        return loss

    pred_grad_x   = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    target_grad_x = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
    rel_x_loss = _relative_grad_loss(pred_grad_x, target_grad_x, eps=eps, max_rel=max_rel)

    pred_grad_y   = pred[:, 1:, :, :, :] - pred[:, :-1, :, :, :]
    target_grad_y = target[:, 1:, :, :, :] - target[:, :-1, :, :, :]
    rel_y_loss = _relative_grad_loss(pred_grad_y, target_grad_y, eps=eps, max_rel=max_rel)

    pred_grad_z   = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
    target_grad_z = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
    rel_z_loss = _relative_grad_loss(pred_grad_z, target_grad_z, eps=eps, max_rel=max_rel)

    pred_grad_t   = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
    target_grad_t = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
    rel_t_loss = _relative_grad_loss(pred_grad_t, target_grad_t, eps=eps, max_rel=max_rel)

    first_order_loss = (rel_x_loss + rel_y_loss + 0.3 * rel_z_loss + rel_t_loss)

    total_loss =  first_order_loss
    total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=1e4, neginf=0.0)
    return total_loss

def create_smooth_weight_map(pred, target, dist_inj, dist_prod, sigma=3.0, task='pressure', alpha_base=1.0, epsilon=1e-4):
    if dist_prod is None or task != 'pressure':
        dist = dist_inj
    else:
        dist = torch.minimum(dist_inj, dist_prod)

    dist = torch.sqrt(dist ** 2 + 1e-6)
    dist_weight = torch.exp(- (dist ** 2) / (2 * sigma ** 2))
    dist_weight = 0.2 + 0.8 * dist_weight

    rel_err = F.smooth_l1_loss(pred, target, reduction='none')
    mean_rel_err = rel_err.mean(dim=(1, 2, 3, 4), keepdim=True) + epsilon
    rel_norm = rel_err / mean_rel_err

    rel_norm = torch.log1p(rel_norm)
    alpha = alpha_base
    weight = dist_weight * (1.0 + alpha * rel_norm)

    if task == 'temperature':
        inj_mask = (dist_inj < 0.5).float()
        weight = weight * (1.0 + 1.5 * inj_mask)

    if task == 'pressure' and dist_prod is not None:
        inj_mask = (dist_inj < 0.5).float()
        prod_mask = (dist_prod < 0.5).float()
        well_mask = (inj_mask + prod_mask).clamp(0, 1)
        weight = weight * (1.0 + 1.5 * well_mask)

    w_mean = weight.mean(dim=(1, 2, 3, 4), keepdim=True).detach()
    weight = weight / (w_mean + 1e-6)

    weight = torch.nan_to_num(weight, nan=0.0, posinf=1.0, neginf=0.0)

    return weight

def weighted_mse(pred, target, weight):
    return torch.mean(weight * (pred - target) ** 2)


### PDE_Loss
class PDE_F():
    @staticmethod
    def den(p, T):
        den_0 = 1000.11  # kg/m3
        a = 7e-7  # kPa-1
        b = 8.8e-4  # C-1
        pr = 100  # kPa
        Tr = 25  # C
        return den_0 * torch.exp(a * (p - pr) - b * (T - Tr))

    @staticmethod
    def vis(T):
        cond1 = T < 40
        cond2 = (T >= 40) & (T < 100)
        cond3 = T >= 100
        val1 = 1.787 * torch.exp(T * (-0.033 + 0.0001962 * T))
        val2 = 1e-3 * (1 + 0.015512 * (T - 20)) ** -1.572
        val3 = 0.02414 * 10 ** (247.8 / (T + 133.15))
        result = torch.where(cond1, val1, torch.where(cond2, val2, val3))
        return result * 0.001

    @staticmethod
    def ent(T):
        A = -203.6060
        B = 1523.29
        C = -3196.413
        D = 2474.455
        E = 3.855326
        F = -256.5478
        H_const = -285.8304
        t = (T + 273.15) / 1000
        H_val = A * t + B * t ** 2 / 2 + C * t ** 3 / 3 + D * t ** 4 / 4 - E / t + F - H_const
        H = (H_val * 1000) / 0.01802
        return H

    @staticmethod
    def harmonic_mean_x(k, eps=1e-30):
        k_l = k[:, :-2,  :, :, :]
        k_c = k[:,  1:-1, :, :, :]
        k_r = k[:,  2:,  :, :, :]

        mean_k_l = 2.0 * k_c * k_l / (k_c + k_l + eps)
        mean_k_r = 2.0 * k_c * k_r / (k_c + k_r + eps)

        mean_k_l = mean_k_l[:, :, 1:-1, 1:-1, :]
        mean_k_r = mean_k_r[:, :, 1:-1, 1:-1, :]
        return mean_k_l, mean_k_r

    @staticmethod
    def harmonic_mean_y(k, eps=1e-30):
        k_d = k[:, :, :-2, :, :]
        k_c = k[:, :, 1:-1, :, :]
        k_u = k[:, :, 2:,  :, :]

        mean_k_d = 2.0 * k_c * k_d / (k_c + k_d + eps)
        mean_k_u = 2.0 * k_c * k_u / (k_c + k_u + eps)

        mean_k_d = mean_k_d[:, 1:-1, :, 1:-1, :]
        mean_k_u = mean_k_u[:, 1:-1, :, 1:-1, :]
        return mean_k_d, mean_k_u

    @staticmethod
    def harmonic_mean_z(k, eps=1e-30):
        k_b = k[:, :, :, :-2, :]
        k_c = k[:, :, :, 1:-1, :]
        k_f = k[:, :, :, 2:,  :]

        mean_k_b = 2.0 * k_c * k_b / (k_c + k_b + eps)
        mean_k_f = 2.0 * k_c * k_f / (k_c + k_f + eps)

        mean_k_b = mean_k_b[:, 1:-1, 1:-1, :, :]
        mean_k_f = mean_k_f[:, 1:-1, 1:-1, :, :]
        return mean_k_b, mean_k_f

    @staticmethod
    def harmonic_mean_v(k):
        k_l = k[..., 0:-2, :]
        k_c = k[..., 1:-1, :]
        k_u = k[..., 2:, :]

        mean_k_low = 2 * k_c * k_l / (k_c + k_l)
        mean_k_up = 2 * k_c * k_u / (k_c + k_u)
        mean_k_low = mean_k_low[..., 1:-1, :, :]
        mean_k_up = mean_k_up[..., 1:-1, :, :]
        return mean_k_low, mean_k_up

    @staticmethod
    def upstream_x(x, p):
        p_l = p[:, :-2,  :, :, :]
        p_c = p[:,  1:-1, :, :, :]
        p_r = p[:,  2:,  :, :, :]

        mat_r = p_r - p_c
        pd_r  = (mat_r >= 0).float()
        x_r   = x[:, 1:-1, :, :, :] * (1.0 - pd_r) + x[:, 2:, :, :, :] * pd_r
        x_r   = x_r[:, :, 1:-1, 1:-1, :]

        mat_l = p_c - p_l
        pd_l  = (mat_l >= 0).float()
        x_l   = x[:, :-2,  :, :, :] * (1.0 - pd_l) + x[:, 1:-1, :, :, :] * pd_l
        x_l   = x_l[:, :, 1:-1, 1:-1, :]
        return x_r, x_l

    @staticmethod
    def upstream_y(x, p):
        p_d = p[:, :, 2:,  :, :]
        p_c = p[:, :, 1:-1, :, :]
        p_u = p[:, :, :-2, :, :]

        mat_u = p_u - p_c
        pd_u  = (mat_u >= 0).float()
        x_u   = x[:, :, 1:-1, :, :] * (1.0 - pd_u) + x[:, :, :-2, :, :] * pd_u
        x_u   = x_u[:, 1:-1, :, 1:-1, :]

        mat_d = p_d - p_c
        pd_d  = (mat_d >= 0).float()
        x_d   = x[:, :, 1:-1, :, :] * (1.0 - pd_d) + x[:, :, 2:,  :, :] * pd_d
        x_d   = x_d[:, 1:-1, :, 1:-1, :]
        return x_u, x_d

    @staticmethod
    def upstream_z(x, p):
        p_b = p[:, :, :, :-2, :]
        p_c = p[:, :, :, 1:-1, :]
        p_f = p[:, :, :, 2:,  :]

        mat_f = p_f - p_c
        pd_f  = (mat_f >= 0).float()
        x_f   = x[:, :, :, 1:-1, :] * (1.0 - pd_f) + x[:, :, :, 2:, :] * pd_f
        x_f   = x_f[:, 1:-1, 1:-1, :, :]

        mat_b = p_b - p_c
        pd_b  = (mat_b >= 0).float()
        x_b   = x[:, :, :, 1:-1, :] * (1.0 - pd_b) + x[:, :, :, :-2, :] * pd_b
        x_b   = x_b[:, 1:-1, 1:-1, :, :]
        return x_f, x_b

class WellboreModel:
    def __init__(self, flow_rate, R_TC, R_cp, tubing_id=0.18, tubing_od=0.2, angle=90, R_rho=2500, U=10, rough=1e-5):
        self.w = flow_rate                 # w (kg/s)
        self.d = tubing_id                 # d (m)
        self.r_to = tubing_od / 2.0        # r_to (m)
        self.theta = np.radians(angle)     # θ
        self.k_e = R_TC                    # k_e, formation conductivity
        self.rho_e = R_rho                 # ρ_e
        self.cp_e = R_cp                   # c_e
        self.g = 9.80665                   # g (m/s^2)
        self.U_to = U                      # U_to (W/m2-K)
        self.eps = rough                   # ε (m)
        self.pump_depth = 1500.0
        self.pump_dp = 10.0

    def friction(self, rho_m, mu_m, v_m):
        Re = rho_m * abs(v_m) * self.d / mu_m

        if (not np.isfinite(Re)) or (Re <= 0.0):
            return 0.0

        Re = float(Re)
        f = None

        Lambda = ((self.eps / self.d) ** 1.1098) / 2.8257 + 7.149 / (Re ** 0.8981)
        if np.isfinite(Lambda) and (Lambda > 0.0):
            term = (self.eps / self.d) / 3.7065 - 5.0452 / Re * np.log10(Lambda)
            if np.isfinite(term) and (term > 0.0):
                logt = np.log10(term)
                if np.isfinite(logt):
                    f_candidate = 1.0 / (4.0 * (logt ** 2))
                    if np.isfinite(f_candidate) and (f_candidate >= 0.0):
                        f = float(f_candidate)

        if f is None:
            f_blasius = 0.3164 / (Re ** 0.25)
            if np.isfinite(f_blasius) and (f_blasius >= 0.0):
                f = float(f_blasius)
            else:
                f = 0.0
        return f

    @staticmethod
    def ramey(t_D):
        if t_D <= 0.0:
            return 0.0
        inner = np.log(np.exp(-0.2) * t_D) + 1.5 - 0.3719 * np.exp(-t_D)
        return inner * np.sqrt(t_D)

    def select_flow_pattern(self, v_sg, v_sL, rho_g, rho_L, mu_g, mu_L, sigma):
        vm = v_sg + v_sL
        if vm <= 0.0 or rho_L <= rho_g:
            return "single_phase", 1.0, 0.0, 0.0

        g = self.g
        d = self.d

        v_inf_b = 1.53 * (g * sigma * (rho_L - rho_g) / (rho_L ** 2)) ** 0.25
        v_inf_T = 0.35 * np.sqrt(g * d * (rho_L - rho_g) / rho_L)
        v_gb = 0.429 * v_sL + 0.357 * v_inf_b

        if v_sg > v_gb and (v_sg - v_gb) > 1e-8:
            expo = np.exp(-0.1 * v_gb / (v_sg - v_gb))
            v_inf_avg = v_inf_b * (1.0 - expo) + v_inf_T * expo
        else:
            v_inf_avg = v_inf_b

        v_ann = 3.1 * (g * sigma * (rho_L - rho_g) / (rho_L ** 2)) ** 0.25

        C0_trial = 1.2
        v_inf_trial = v_inf_b if v_inf_T <= v_inf_b else v_inf_avg
        fg_trial = v_sg / (C0_trial * vm + v_inf_trial)
        fg_trial = float(np.clip(fg_trial, 0.0, 0.999))

        rho_m_trial = fg_trial * rho_g + (1.0 - fg_trial) * rho_L
        x_mass = fg_trial
        mu_m_trial = mu_g * x_mass + mu_L * (1.0 - x_mass)

        f_m = self.friction(rho_m_trial, mu_m_trial, vm)

        LHS = (2.0 * vm ** 1.2) * (f_m / (2.0 * d)) ** 0.4 * (rho_L / sigma) ** 0.6 * ((0.4 * sigma) / (g * (rho_L - rho_g))) ** 0.5
        RHS = 0.725 + 4.15 * np.sqrt(v_sg / vm)

        # Regime Detection
        flow_regime = None

        if fg_trial > 0.85 and v_sg > v_ann:
            flow_regime = "annular"
        else:
            if fg_trial < 0.25:
                flow_regime = "bubbly"
            if v_sg > v_gb:
                flow_regime = "slug"
                if v_sg > 1.08 * v_sL:
                    flow_regime = "churn"
            if LHS > RHS:
                if fg_trial < 0.58:
                    flow_regime = "dispersed_bubbly"
                if v_sg > 1.08 * v_sL:
                    flow_regime = "churn"

        if flow_regime is None:
            flow_regime = "bubbly"

        # Calculate C0, v_inf, fg
        if flow_regime == "bubbly":
            C0 = 1.2
            v_inf = v_inf_b
        elif flow_regime == "slug":
            C0 = 1.2
            v_inf = v_inf_avg
        elif flow_regime == "churn":
            C0 = 1.15
            v_inf = v_inf_avg
        elif flow_regime == "dispersed_bubbly":
            C0 = 1.15
            v_inf = v_inf_b
        elif flow_regime == "annular":
            C0 = 1.0
            v_inf = 0.0
        else:  # 退化
            C0 = 1.0
            v_inf = 0.0

        fg = v_sg / (C0 * vm + v_inf)
        fg = float(np.clip(fg, 0.0, 0.999))

        return flow_regime, C0, v_inf, fg

    def solve_step(self, P_MPa, h_kJkg, T_ei_C, t_days, r_wb_m=None):
        P_MPa = float(P_MPa)
        h_kJkg = float(h_kJkg)
        T_ei_C = float(T_ei_C)

        if not np.isfinite(P_MPa):
            P_MPa = 10.0
        P_MPa = float(np.clip(P_MPa, 0.1, 80.0))

        if not np.isfinite(h_kJkg):
            h_kJkg = 1000.0
        h_kJkg = float(np.clip(h_kJkg, 100.0, 4000.0))

        if not np.isfinite(T_ei_C):
            T_ei_C = 100.0
        T_ei_C = float(np.clip(T_ei_C, 0.0, 350.0))
        T_K = T_ei_C + 273.15
        T_K = float(np.clip(T_K, 273.15, 1073.15))

        try:
            fluid = IAPWS97(P=P_MPa, h=h_kJkg)
        except NotImplementedError:
            try:
                fluid = IAPWS97(P=P_MPa, T=T_K)
                h_kJkg = fluid.h
            except NotImplementedError:
                fluid = IAPWS97(P=P_MPa, x=0)
                h_kJkg = fluid.h

        T_f_C = fluid.T - 273.15

        # Phase Detection
        if P_MPa < 22.064 and np.isfinite(P_MPa):
            sat_L = IAPWS97(P=P_MPa, x=0)
            sat_g = IAPWS97(P=P_MPa, x=1)
            h_L, h_g = sat_L.h, sat_g.h

            if h_kJkg <= h_L:
                x = 0.0
                rho_L = rho_g = fluid.rho
                mu_L = mu_g = fluid.mu
                sigma = sat_L.sigma
                cp_mix_kJ = fluid.cp
            elif h_kJkg >= h_g:
                x = 1.0
                rho_L = rho_g = fluid.rho
                mu_L = mu_g = fluid.mu
                sigma = sat_L.sigma
                cp_mix_kJ = fluid.cp
            else:
                x = (h_kJkg - h_L) / (h_g - h_L)
                x = float(np.clip(x, 0.0, 1.0))
                rho_L, rho_g = sat_L.rho, sat_g.rho
                mu_L, mu_g = sat_L.mu, sat_g.mu
                sigma = sat_L.sigma

                cp_L_kJ = sat_L.cp
                cp_g_kJ = sat_g.cp
                cp_mix_kJ = x * cp_g_kJ + (1.0 - x) * cp_L_kJ
        else:
            x = 0.0
            rho_L = rho_g = fluid.rho
            mu_L = mu_g = fluid.mu
            sigma = 0.05
            cp_mix_kJ = fluid.cp

        # Velocity
        A = np.pi * self.d ** 2 / 4.0
        if x <= 0.0:
            q_L = self.w / rho_L
            v_sL = q_L / A
            v_sg = 0.0
        elif x >= 1.0:
            q_g = self.w / rho_g
            v_sg = q_g / A
            v_sL = 0.0
        else:
            q_g = self.w * x / rho_g
            q_L = self.w * (1.0 - x) / rho_L
            v_sg = q_g / A
            v_sL = q_L / A

        v_m = v_sg + v_sL

        # Drift flux
        if x <= 0.0:
            f_g = 0.0
        elif x >= 1.0:
            f_g = 1.0
        else:
            flow_regime, C0, v_inf, f_g = self.select_flow_pattern(
                v_sg=v_sg,
                v_sL=v_sL,
                rho_g=rho_g, rho_L=rho_L,
                mu_g=mu_g, mu_L=mu_L,
                sigma=sigma
            )

        # Mixed Properties
        rho_m = rho_g * f_g + rho_L * (1.0 - f_g)
        mu_m = mu_g * x + mu_L * (1.0 - x)

        # Momentum transport
        f_fric = self.friction(rho_m, mu_m, v_m)
        grad_fric_Pa_per_m = f_fric * rho_m * v_m ** 2 / (2.0 * self.d)
        grad_grav_Pa_per_m = rho_m * self.g * np.sin(self.theta)
        dp_dz_sf_MPa = - (grad_grav_Pa_per_m + grad_fric_Pa_per_m) / 1e6

        # Energy transport
        if r_wb_m is None:
            r_wb_m = self.r_to

        t_sec = t_days * 24.0 * 3600.0
        t_D = self.k_e * t_sec / (self.rho_e * self.cp_e * r_wb_m ** 2)
        T_D = self.ramey(t_D)

        cp_f = cp_mix_kJ * 1000.0
        L_R = (2.0 * np.pi / (self.w * cp_f)) * \
              (self.r_to * self.U_to * self.k_e) / (self.k_e + self.r_to * self.U_to * T_D)

        Q_wb_per_m = L_R * self.w * cp_f * (T_f_C - T_ei_C)  # W/m

        grav_term_kJkg_per_m = - self.g * np.sin(self.theta) / 1000.0
        heat_term_kJkg_per_m = (Q_wb_per_m / self.w) / 1000.0
        dh_dz_no_acc = grav_term_kJkg_per_m - heat_term_kJkg_per_m

        return dp_dz_sf_MPa, dh_dz_no_acc, v_m, rho_m, x, f_g, T_f_C, Q_wb_per_m

    def simulate(self, depth, P_btm, Tf_btm, T_grad, T_surf=15, steps=40, t_days=0.0):
        P_btm = float(P_btm)
        Tf_btm = float(Tf_btm)

        if not np.isfinite(P_btm):
            P_btm = 10.0
        P_btm = float(np.clip(P_btm, 0.1, 80.0))

        if not np.isfinite(Tf_btm):
            Tf_btm = 100.0
        Tf_btm = float(np.clip(Tf_btm, 0.0, 350.0))
        Tf_btm_K = Tf_btm + 273.15
        Tf_btm_K = float(np.clip(Tf_btm_K, 273.15, 1073.15))

        try:
            fluid_btm = IAPWS97(P=P_btm, T=Tf_btm_K)
        except NotImplementedError:
            P_btm = float(np.clip(P_btm, 0.5, 50.0))
            Tf_btm_K = float(np.clip(Tf_btm_K, 300.0, 800.0))
            fluid_btm = IAPWS97(P=P_btm, T=Tf_btm_K)

        dz = depth / steps
        P_curr = P_btm
        h_curr = fluid_btm.h
        v_m_prev = None

        rho_ref = fluid_btm.rho
        P_surf_MPa = 0.1
        P_hydro_MPa = P_surf_MPa + rho_ref * self.g * depth / 1e6
        use_pump = P_btm < P_hydro_MPa
        pump_applied = False

        z_list = []
        P_list = []
        T_list = []
        h_list = []
        Q_per_m_list = []

        for i in range(steps + 1):
            z_curr = depth - i * dz
            T_ei = T_surf + T_grad * z_curr

            dp_dz_sf_MPa, dh_dz_no_acc, v_m, rho_m, x, f_g, T_f_C, Q_loss = self.solve_step(P_curr, h_curr, T_ei, t_days)

            z_list.append(z_curr)
            P_list.append(P_curr)
            T_list.append(T_f_C)
            h_list.append(h_curr)
            Q_per_m_list.append(Q_loss)

            if i < steps:
                if v_m_prev is None:
                    dv_dz = 0.0
                else:
                    dv_dz = (v_m - v_m_prev) / dz

                acc_term_MPa_per_m = rho_m * v_m * dv_dz / 1e6
                dp_dz = dp_dz_sf_MPa - acc_term_MPa_per_m
                dh_dz = dh_dz_no_acc - (v_m * dv_dz) / 1000.0

                P_curr += dp_dz * dz
                h_curr += dh_dz * dz
                v_m_prev = v_m

                if use_pump and (not pump_applied):
                    z_next = z_curr - dz
                    if (z_curr >= self.pump_depth) and (z_next <= self.pump_depth):
                        P_curr += self.pump_dp
                        pump_applied = True

                if P_curr < 1:
                    P_curr = 1

        T_wellhead = T_list[-1]
        P_wellhead = P_list[-1]
        h_wellhead = h_curr

        Q_loss_total = sum(Q_per_m_list) * dz

        return T_wellhead, P_wellhead, Q_loss_total, h_wellhead

def wellbore_model_run(BHT, BHP, depth, T_grad, m_dot, R_TC, R_cp):
    well = WellboreModel(flow_rate=m_dot, R_TC=R_TC, R_cp=R_cp)
    T_wh, P_wh, Q_loss_total, h_wh = well.simulate(depth=depth, P_btm=BHP, Tf_btm=BHT, T_grad=T_grad)
    return T_wh, P_wh, Q_loss_total, h_wh

def extract_wellData(field, mask):
    B, X, Y, Z, T = field.shape
    device = field.device
    dtype = field.dtype

    if mask.dim() == 5:
        mask = mask.any(dim=-1)
    elif mask.dim() == 4:
        mask = mask.bool()

    result = []

    for b in range(B):
        field_b = field[b]
        mask_b  = mask[b]
        active_xy = mask_b.any(dim=2)
        coords_xy = torch.nonzero(active_xy, as_tuple=False)

        well_ts_list = []

        for coord_xy in coords_xy:
            x, y = coord_xy.tolist()

            mask_z = mask_b[x, y, :]
            if not mask_z.any():
                continue

            ts_blocks = field_b[x, y, mask_z, :]
            ts = ts_blocks.mean(dim=0)
            well_ts_list.append(ts)

        if well_ts_list:
            well_ts_tensor = torch.stack(well_ts_list, dim=0)
        else:
            well_ts_tensor = torch.zeros(0, T, device=device, dtype=dtype)

        result.append(well_ts_tensor)

    return result

def process_wellResult(dz_list, grad_T_list, BHT_list, BHP_list, BHQ_list, TC_list, Cp_list):
    Twh_list = []
    Q_loss_list = []
    Pwh_list = []
    hwh_list = []

    for sample_idx in range(len(BHT_list)):
        bht_tensor = BHT_list[sample_idx]
        bhq_tensor = BHQ_list[sample_idx]
        bhp_tensor = BHP_list[sample_idx]

        grad_T = float(grad_T_list[sample_idx])
        depth = float(dz_list[sample_idx])

        TC_tensor = TC_list[sample_idx]
        Cp_tensor = Cp_list[sample_idx]
        R_TC = float(TC_tensor.mean().item())
        R_cp = float(Cp_tensor.mean().item())

        N_i, Tn = bht_tensor.shape
        T_outs = torch.zeros(N_i, Tn, dtype=bht_tensor.dtype, device=bht_tensor.device)
        Q_losses = torch.zeros(N_i, Tn, dtype=bht_tensor.dtype, device=bht_tensor.device)
        Pwh = torch.zeros(N_i, Tn, dtype=bht_tensor.dtype, device=bht_tensor.device)
        hwh = torch.zeros(N_i, Tn, dtype=bht_tensor.dtype, device=bht_tensor.device)

        for well_idx in range(N_i):
            for t in range(Tn):
                T0 = float(bht_tensor[well_idx, t].item())
                m_dot = float(bhq_tensor[well_idx, t].item())
                P_btm = float(bhp_tensor[well_idx, t].item())

                T_wh, P_wh, Q_loss_total, h_wh = wellbore_model_run(
                    BHT=T0, BHP=P_btm, depth=depth,
                    T_grad=grad_T, m_dot=m_dot,
                    R_TC=R_TC, R_cp=R_cp
                )

                T_outs[well_idx, t] = T_wh
                Q_losses[well_idx, t] = Q_loss_total
                Pwh[well_idx, t] = P_wh
                hwh[well_idx, t] = h_wh

        Twh_list.append(T_outs)
        Q_loss_list.append(Q_losses)
        Pwh_list.append(Pwh)
        hwh_list.append(hwh)

    return Twh_list, Q_loss_list, Pwh_list, hwh_list

class physical_loss(nn.Module):
    def __init__(self, T_i, p_i, Cp_r, lam_r, dz, q_inj, pwf, T_inj, k, phi, depth, dt=24*3600*365, dx=10.0, dy=10.0):
        super().__init__()

        self.dx, self.dy, self.dt = float(dx), float(dy), dt
        self.T_i, self.p_i = T_i, p_i
        self.Cp_r, self.lam_r = Cp_r, lam_r
        self.pwf, self.k, self.d = pwf, k, depth
        self.dz = dz / 5.0
        self.q_inj, self.T_inj = q_inj, T_inj
        self.k_unit_conversion = 9.86923e-16

        self.cc = 0.249
        self.rw = 0.0762
        self.re = self.cc * math.sqrt((self.dx ** 2 + self.dy ** 2) / math.pi)
        self.log_re_rw = math.log(self.re / self.rw)

        self.block_v = dx * dy * self.dz
        self.fluid_v = self.block_v * phi
        self.rock_v = self.block_v - self.fluid_v
        self.lam_fluid = 5.6e4 / (24 * 3600)
        self.lam_rock = lam_r / (24 * 3600)
        self.lam_eff = self.lam_fluid * ((self.lam_rock / self.lam_fluid) ** 0.6354)

    def phy_loss(self, p_pred, T_pred):
        device = p_pred.device
        S_c = (slice(None), slice(1, -1), slice(1, -1), slice(1, -1), slice(1, None))

        def pad_3D(x, mode='replicate'):
            x = x.permute(0, 4, 3, 2, 1)
            x = F.pad(x, (1, 1, 1, 1, 1, 1), mode=mode)
            x = x.permute(0, 4, 3, 2, 1)
            return x

        p_pred = pad_3D(p_pred)
        T_pred = pad_3D(T_pred)
        k_pad = pad_3D(self.k)
        pwf_pad = pad_3D(self.pwf)
        q_inj_pad = pad_3D(self.q_inj.to(device))
        T_inj_pad = pad_3D(self.T_inj.to(device))

        p_Pa = p_pred * 1e6
        p_kPa = p_Pa / 1000.0
        rho = torch.clamp(PDE_F.den(p_kPa, T_pred), 500.0, 2000.0)
        mu = torch.clamp(PDE_F.vis(T_pred), 1e-4, 1e1)
        h = PDE_F.ent(T_pred)
        h = torch.nan_to_num(h, nan=0.0, posinf=1e7, neginf=-1e7).clamp(-1e6, 1e6)
        k = (k_pad * self.k_unit_conversion).to(device=device)

        T_c, p_c, rho_c, mu_c, h_c, k_c = T_pred[S_c], p_Pa[S_c], rho[S_c], mu[S_c], h[S_c], k[S_c]
        p_t, rho_t, mu_t, h_t, k_t = p_Pa[..., 1:], rho[..., 1:], mu[..., 1:], h[..., 1:], k[..., 1:]
        dz_b = self.dz.to(device=k_c.device, dtype=k_c.dtype).view(-1, 1, 1, 1, 1)

        pwf_local = (pwf_pad.to(device))[S_c] * (-1.0) * 1e3
        pwf_mask = (pwf_local != 0).to(p_c.dtype)

        q_inj = q_inj_pad[S_c] * 1001.7 / 24 / 3600
        H_inj = PDE_F.ent(T_inj_pad)[S_c]
        energy_inj = q_inj * H_inj

        I = (2.0 * math.pi) * k_c * dz_b / self.log_re_rw

        inj_mask = (q_inj != 0).to(p_c.dtype)
        P_inj = q_inj / (I * (1.0 / mu_c)) / rho_c + p_c

        p_block = p_c * pwf_mask
        q_out = I * (1.0 / mu_c) * (pwf_local - p_block) * rho_c
        q_out = q_out * -1

        energy_out = q_out * h_c
        heat_pro = q_out * (h_c - H_inj)

        fluid_v = self.fluid_v.to(device=device, dtype=k_c.dtype)[..., 1:]
        rock_v  = self.rock_v.to(device=device, dtype=k_c.dtype)[..., 1:]

        ### PDE Variance
        rho_next, rho_last = rho[:, 1:-1, 1:-1, 1:-1, 1:], rho[:, 1:-1, 1:-1, 1:-1, :-1]
        T_next, T_last = T_pred[:, 1:-1, 1:-1, 1:-1, 1:], T_pred[:, 1:-1, 1:-1, 1:-1, :-1]
        p_next, p_last = p_Pa[:, 1:-1, 1:-1, 1:-1, 1:], p_Pa[:, 1:-1, 1:-1, 1:-1, :-1]
        h_next, h_last = h[:, 1:-1, 1:-1, 1:-1, 1:], h[:, 1:-1, 1:-1, 1:-1, :-1]

        p_r, p_l = p_Pa[:, 2:,   1:-1, 1:-1, 1:], p_Pa[:, 0:-2, 1:-1, 1:-1, 1:]
        p_u, p_d = p_Pa[:, 1:-1, 0:-2, 1:-1, 1:], p_Pa[:, 1:-1, 2:,   1:-1, 1:]
        p_b, p_f = p_Pa[:, 1:-1, 1:-1, 0:-2, 1:], p_Pa[:, 1:-1, 1:-1, 2:,   1:]

        T_r, T_l = T_pred[:, 2:,   1:-1, 1:-1, 1:], T_pred[:, 0:-2, 1:-1, 1:-1, 1:]
        T_u, T_d = T_pred[:, 1:-1, 0:-2, 1:-1, 1:], T_pred[:, 1:-1, 2:,   1:-1, 1:]
        T_b, T_f = T_pred[:, 1:-1, 1:-1, 0:-2, 1:], T_pred[:, 1:-1, 1:-1, 2:,   1:]

        Ax, Ay, Az = self.dy * dz_b, self.dx * dz_b, self.dx * self.dy

        rho_r, rho_l = PDE_F.upstream_x(rho_t, p_t)
        rho_u, rho_d = PDE_F.upstream_y(rho_t, p_t)
        rho_f, rho_b = PDE_F.upstream_z(rho_t, p_t)

        mu_r, mu_l = PDE_F.upstream_x(mu_t, p_t)
        mu_u, mu_d = PDE_F.upstream_y(mu_t, p_t)
        mu_f, mu_b = PDE_F.upstream_z(mu_t, p_t)

        h_r, h_l = PDE_F.upstream_x(h_t, p_t)
        h_u, h_d = PDE_F.upstream_y(h_t, p_t)
        h_f, h_b = PDE_F.upstream_z(h_t, p_t)

        k_l, k_r = PDE_F.harmonic_mean_x(k_t)
        k_d, k_u = PDE_F.harmonic_mean_y(k_t)
        k_b, k_f = PDE_F.harmonic_mean_z(k_t)

        # Mass Balance
        tx_r = (rho_r / mu_r) * k_r * (Ax / self.dx)
        tx_l = (rho_l / mu_l) * k_l * (Ax / self.dx)
        ty_u = (rho_u / mu_u) * k_u * (Ay / self.dy)
        ty_d = (rho_d / mu_d) * k_d * (Ay / self.dy)
        tz_b = (rho_b / mu_b) * k_b * (Az / self.dz)
        tz_f = (rho_f / mu_f) * k_f * (Az / self.dz)

        mass_flux = (tx_r * (p_r - p_c) + tx_l * (p_l - p_c) + ty_u * (p_u - p_c) + ty_d * (p_d - p_c) +
                     tz_f * (p_f - p_c) + tz_b * (p_b - p_c))
        mass_acc = (rho_next - rho_last) * fluid_v / self.dt
        mass_residual = mass_acc - mass_flux - (q_inj + q_out)
        mse_mass = torch.mean(torch.abs(mass_residual)) * 5e-1

        # Energy Balance
        energy_acc = ((fluid_v * rho_next * (h_next - p_next / rho_next) + rock_v * (self.Cp_r * T_next)) -
                      (fluid_v * rho_last * (h_last - p_last / rho_last) + rock_v * (self.Cp_r * T_last))) / self.dt

        lam_r = (rho_r / mu_r) * h_r * k_r * (Ax / self.dx)
        lam_l = (rho_l / mu_l) * h_l * k_l * (Ax / self.dx)
        lam_u = (rho_u / mu_u) * h_u * k_u * (Ay / self.dy)
        lam_d = (rho_d / mu_d) * h_d * k_d * (Ay / self.dy)
        lam_b = (rho_b / mu_b) * h_b * k_b * (Az / self.dz)
        lam_f = (rho_f / mu_f) * h_f * k_f * (Az / self.dz)

        energy_conv = (lam_r * (p_r - p_c) + lam_l * (p_l - p_c) + lam_u * (p_u - p_c) + lam_d * (p_d - p_c) +
                       lam_f * (p_f - p_c) + lam_b * (p_b - p_c))

        energy_cond = self.lam_eff * ((T_r - T_c + T_l - T_c) * (Ax / self.dx) + (T_u - T_c + T_d - T_c) * (Ay / self.dy)
                                      + (T_f - T_c + T_b - T_c) * (Az / self.dz))

        energy_residual = energy_acc - energy_conv - energy_cond - (energy_inj - energy_out)
        mse_energy = torch.mean(torch.abs(energy_residual)) * 1e-8

        ### Wellbore
        B = p_pred.shape[0]

        T_out_map = T_c
        q_out_map = q_out
        P_out_map = p_c
        heat_map = heat_pro
        Pinj_map = P_inj

        BHT_list = extract_wellData(T_out_map, pwf_mask)
        BHQ_list = extract_wellData(q_out_map, pwf_mask)
        BHP_list = extract_wellData(P_out_map / 1e6, pwf_mask)
        BHH_list = extract_wellData(heat_map, pwf_mask)

        dz_list = [float(self.d)] * B
        grad_T_list = [float(self.T_i.max().item() / self.d)] * B
        TC_list = self.lam_r
        Cp_list = self.Cp_r

        Temp_wh, Q_loss, Pres_wh, Enth_wh = process_wellResult(
            dz_list=dz_list,
            grad_T_list=grad_T_list,
            BHT_list=BHT_list,
            BHP_list=BHP_list,
            BHQ_list=BHQ_list,
            TC_list=TC_list,
            Cp_list=Cp_list
        )

        Heat_wh = [a - b for a, b in zip(BHH_list, Q_loss)]
        P_inj = extract_wellData(Pinj_map, inj_mask)

        return mse_mass, mse_energy, Temp_wh, Heat_wh, P_inj, Enth_wh, q_out

def compute_loss(p_out, T_out, loss_func, Temp_wh_t, Heat_wh_t, P_inj_t):
    mse_mass, mse_energy, Temp_wh, Heat_wh, P_inj, Enth_wh, q_out = loss_func.phy_loss(p_out, T_out)

    def re_task(a, b, eps=1e-8):
        if torch.is_tensor(a) and torch.is_tensor(b):
            rel = torch.abs(a - b) / (torch.abs(b) + eps)
            return rel.mean()

        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) == 0 or len(b) == 0:
                dev = mse_mass.device if hasattr(mse_mass, "device") else "cpu"
                return torch.tensor(0.0, device=dev)
            dev = a[0].device if torch.is_tensor(a[0]) else (mse_mass.device if hasattr(mse_mass, "device") else "cpu")
            total_error = torch.tensor(0.0, device=dev)
            total_count = 0

            for x, y in zip(a, b):
                if not (torch.is_tensor(x) and torch.is_tensor(y)):
                    raise TypeError("Elements in list must be torch.Tensor")
                rel = torch.abs(x - y) / (torch.abs(y) + eps)
                total_error = total_error + rel.sum()
                total_count += rel.numel()

            if total_count == 0:
                return torch.tensor(0.0, device=dev)

            return total_error / float(total_count)

        raise TypeError(
            f"re_task is not Tensor or list, but {type(a)} and {type(b)}"
        )

    re_Twh = re_task(Temp_wh, Temp_wh_t) * 0.1
    re_Ewh = re_task(Heat_wh, Heat_wh_t) * 0.1
    re_Pinj = re_task(P_inj, P_inj_t) * 0.1

    return mse_mass, mse_energy, re_Twh, re_Ewh, re_Pinj
