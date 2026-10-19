from .transport import Sampler, Transport


def create_transport(
    time_dist_type="logit-normal_0_1",
    time_dist_shift=1.0,
):
    return Transport(
        time_dist_type=time_dist_type,
        time_dist_shift=time_dist_shift,
    )


__all__ = ["Sampler", "Transport", "create_transport"]
