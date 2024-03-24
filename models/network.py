import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, module_name='sr3', **kwargs):
        super(Network, self).__init__(**kwargs)
        if module_name == 'sr3':
            from .sr3_modules.unet import UNet
        elif module_name == 'guided_diffusion':
            from .guided_diffusion_modules.unet import UNet
        elif module_name == 'DiT':
            from .DiT.models import DiT_models

        self.denoise_fn = DiT_models['DiT-B/4'](
            input_size=64,
            num_classes=3
        )

        # self.denoise_fn = UNet(**unet)
        self.beta_schedule = beta_schedule
        self.model_mean_type = 'dualx'

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def q_posterior_mean_dualx(self, x_0, x_t, t):
        mean = (extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        return mean

    def q_posterior_mean_dualx_implicit(self, x_0, x_t, t, noise):
        if x_0 is None:
            # predict mean using x_t and noise
            betas = extract(self.betas, t, x_t.shape)
            alphas = 1 - betas
            alphas_cumprod = extract(self.alphas_cumprod, t, x_t.shape)
            sqrt_one_minus_alphas_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

            posterior_ddim_coef = torch.sqrt(alphas - alphas_cumprod) - sqrt_one_minus_alphas_cumprod
            mean = (x_t + posterior_ddim_coef * noise) / torch.sqrt(alphas)
        else:
            # predict mean using x_0 and noise
            sqrt_alphas_cumprod_prev = extract(self.sqrt_alphas_cumprod_prev, t, x_0.shape)
            sqrt_one_minus_alphas_cumprod_prev = extract(self.sqrt_one_minus_alphas_cumprod_prev, t, x_0.shape)

            mean = sqrt_alphas_cumprod_prev * x_0 + sqrt_one_minus_alphas_cumprod_prev * noise

        return mean

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)

        class_labels = [3]
        y = torch.tensor(class_labels, device=y_t.device)
        
        y_0_hat = self.predict_start_from_noise(
                y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), t, y)[:, 3:6])

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8):
        b, *_ = y_cond.shape

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps//sample_num)
        
        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=y_cond)
            if mask is not None:
                y_t = y_0*(1.-mask) + mask*y_t
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
        return y_t, ret_arr

    def predict_dualx(self, model_output, x_t, t, mode=None, is_implicit=False, return_pred_x0=False, fn=None):

        model_output_xstart, model_output_eps, model_output_thr = model_output[:, :3], model_output[:, 3:6], model_output[:, 6:7]
        pred_x0_xstart = model_output_xstart
        pred_x0_eps = self.predict_start_from_noise(x_t, t, model_output_eps)

        s = model_output_thr.sigmoid()
        pred_x0_thr = s * pred_x0_xstart.detach() + (1-s) * (pred_x0_eps).detach()

        if not is_implicit:
            model_output_thr_mean = self.q_posterior_mean_dualx(x_0=pred_x0_thr, x_t=x_t, t=t)
        else:
            pred_noise_thr = self.predict_noise_from_start(pred_x0_thr, x_t, t)
            model_output_thr_mean = self.q_posterior_mean_dualx_implicit(x_0=pred_x0_thr, x_t=x_t, t=t, noise=pred_noise_thr)

        if mode == 'mean':
            if not return_pred_x0:
                return model_output_thr_mean
            else:
                return model_output_thr_mean, pred_x0_thr
        else:
            assert mode is None
            pred = torch.cat((model_output_xstart, model_output_eps, model_output_thr_mean), dim=1)
            return pred

    def forward(self, y_0, y_cond=None, mask=None, noise=None):
        # sampling from p(gammas)
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)
        class_labels = [3]

        y = torch.tensor(class_labels, device=y_0.device)

        if mask is not None:
            # noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas)
            # loss = self.loss_fn(mask*noise, mask*noise_hat)

            model_output = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), t, y)
            mean = self.q_posterior_mean_dualx(x_0=y_0, x_t=y_noisy, t=t)
            target = {
                'xprev': lambda: mean,
                'xstart': lambda: y_0,
                'eps': lambda: noise,
                'dualx': lambda: torch.cat((y_0, noise, mean), dim=1),
            }[self.model_mean_type]()
            if self.model_mean_type != 'dualx':
                assert model_output.shape[1] == 3
                pred = model_output
            else:
                assert model_output.shape[1] == 7
                pred = self.predict_dualx(model_output=model_output, x_t=y_noisy, t=t)

            losses = torch.mean((target - pred).view(y_0.shape[0], -1) ** 2, dim=1)

            loss = losses.mean()

        else:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
            loss = self.loss_fn(noise, noise_hat)

        return loss


# gaussian diffusion trainer class
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape=(1,1,1,1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


