import torch


def _mean_flat(x):
    return x.mean(dim=tuple(range(1, x.ndim)))


def _expand_time(t, x):
    return t.view(t.shape[0], *((1,) * (x.ndim - 1)))


def _logit_normal(shape, mean, std):
    if std <= 0:
        raise ValueError("Logit-normal standard deviation must be positive")
    uniform = torch.rand(shape, dtype=torch.float32)
    standard_normal = torch.distributions.Normal(
        torch.tensor(0.0), torch.tensor(1.0)
    )
    return torch.sigmoid(mean + std * standard_normal.icdf(uniform))


class Transport:
    def __init__(self, time_dist_type, time_dist_shift):
        if time_dist_shift < 1:
            raise ValueError("time_dist_shift must be at least 1")
        distribution = time_dist_type.split("_")
        if len(distribution) != 3 or distribution[0] != "logit-normal":
            raise ValueError(
                "Only time_dist_type='logit-normal_<mean>_<std>' is supported"
            )
        self.time_mean = float(distribution[1])
        self.time_std = float(distribution[2])
        self.time_dist_shift = float(time_dist_shift)

    def _sample_time(self, x):
        t = _logit_normal(
            (x.shape[0],),
            self.time_mean,
            self.time_std,
        ).to(x)
        return self.time_dist_shift * t / (
            1 + (self.time_dist_shift - 1) * t
        )

    def training_losses(self, model, x1, model_kwargs=None):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        x0 = torch.randn_like(x1)
        t = self._sample_time(x1)
        expanded_t = _expand_time(t, x1)
        xt = (1 - expanded_t) * x1 + expanded_t * x0
        target = x0 - x1
        prediction = model(xt, t, **model_kwargs)
        if prediction.shape != x1.shape:
            raise RuntimeError(
                f"Model output shape {tuple(prediction.shape)} does not match "
                f"latent shape {tuple(x1.shape)}"
            )
        return {
            "pred": prediction,
            "loss": _mean_flat((prediction - target).square()),
        }


class Sampler:
    def __init__(self, transport):
        self.transport = transport

    def sample_ode(self, *, num_steps):
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2")
        schedule = 1 - torch.linspace(0, 1, num_steps)
        shift = self.transport.time_dist_shift
        schedule = shift * schedule / (1 + (shift - 1) * schedule)

        @torch.no_grad()
        def sample(x, model, **model_kwargs):
            samples = [x]
            timesteps = schedule.to(device=x.device, dtype=x.dtype)
            for current, following in zip(timesteps[:-1], timesteps[1:]):
                t = torch.ones(
                    x.shape[0], device=x.device, dtype=x.dtype
                ) * current
                velocity = model(x, t, **model_kwargs)
                if velocity.shape != x.shape:
                    raise RuntimeError(
                        f"Model output shape {tuple(velocity.shape)} does not "
                        f"match sample shape {tuple(x.shape)}"
                    )
                x = x + (following - current) * velocity
                samples.append(x)
            return samples

        return sample
