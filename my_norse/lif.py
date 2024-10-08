"""This code is taken from the Norse library (https://github.com/norse/norse) due to problems when importing it as a
whole."""
from typing import NamedTuple, Optional, Tuple
import torch
import torch.jit


@torch.jit.script
def heaviside(data):
    return torch.gt(data, torch.as_tensor(0.0)).to(data.dtype)  # pragma: no cover


class SuperSpike(torch.autograd.Function):
    @staticmethod
    @torch.jit.ignore
    def forward(ctx, input_tensor: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.alpha = alpha
        return heaviside(input_tensor)

    @staticmethod
    @torch.jit.ignore
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        grad = grad_input / (alpha * torch.abs(inp) + 1.0).pow(
            2
        )  # section 3.3.2 (beta -> alpha)
        return grad, None


@torch.jit.ignore
def super_fn(x: torch.Tensor, alpha: float = 100.0) -> torch.Tensor:
    return SuperSpike.apply(x, alpha)


def threshold(x: torch.Tensor, method: str, alpha: float) -> torch.Tensor:
    if method == "heaviside":
        return heaviside(x)
    elif method == "super":
        return super_fn(x, torch.as_tensor(alpha))
    else:
        raise ValueError(
            f"Attempted to apply threshold function {method}, but no such "
            + "function exist. We currently support heaviside, super, "
            + "tanh, triangle, circ, and heavi_erfc."
        )


class LIFParameters:
    tau_mem_inv: torch.Tensor = torch.as_tensor(1 / 134.)
    v_leak: torch.Tensor = torch.as_tensor(0.0)
    v_th: torch.Tensor = torch.as_tensor(1.0)
    v_reset: torch.Tensor = torch.as_tensor(0.0)
    method: str = "super"
    alpha: float = torch.as_tensor(100.0)


class LIFState(NamedTuple):
    z: torch.Tensor
    v: torch.Tensor


class LIFFeedForwardState(NamedTuple):
    v: torch.Tensor


class LIFParametersJIT(NamedTuple):
    tau_mem_inv: torch.Tensor
    v_leak: torch.Tensor
    v_th: torch.Tensor
    v_reset: torch.Tensor
    method: str
    alpha: torch.Tensor


@torch.jit.script
def _lif_step_jit(
        input_spikes: torch.Tensor,
        state: LIFState,
        input_weights: torch.Tensor,
        recurrent_weights: torch.Tensor,
        p: LIFParametersJIT,
        dt: float = 1.0,
) -> Tuple[torch.Tensor, LIFState]:  # pragma: no cover
    # compute current jumps
    i_in = torch.nn.functional.linear(input_spikes, input_weights) \
             + torch.nn.functional.linear(state.z, recurrent_weights)

    # compute voltage updates
    dv = dt * p.tau_mem_inv * (p.v_leak - state.v)
    v_decayed = state.v + dv
    # print(p.tau_mem_inv)

    # clip values smaller than -1
    low_pass = torch.gt(v_decayed, -torch.ones_like(v_decayed))
    low_pass = 1. * low_pass
    v_decayed = v_decayed - (torch.ones_like(v_decayed) - low_pass) * (v_decayed + torch.ones_like(v_decayed))

    # add input
    v_decayed = v_decayed + i_in

    # compute new spikes
    z_new = threshold(v_decayed - p.v_th, p.method, p.alpha)
    # compute reset
    v_new = (1 - z_new.detach()) * v_decayed + z_new.detach() * p.v_reset

    return z_new, LIFState(z_new, v_new)


def lif_step(
        input_spikes: torch.Tensor,
        state: LIFState,
        input_weights: torch.Tensor,
        recurrent_weights: torch.Tensor,
        p: LIFParameters = LIFParameters(),
        dt: float = 1.0,
) -> Tuple[torch.Tensor, LIFState]:
    return _lif_step_jit(
        input_spikes, state, input_weights, recurrent_weights, LIFParametersJIT(*p), dt
    )


@torch.jit.script
def _lif_feed_forward_step_jit(
        input_tensor: torch.Tensor,
        state: LIFFeedForwardState,
        p: LIFParametersJIT,
        dt: float = 1.0,
) -> Tuple[torch.Tensor, LIFFeedForwardState]:  # pragma: no cover

    i_in = input_tensor

    # compute voltage updates
    dv = dt * p.tau_mem_inv * (p.v_leak - state.v)
    v_decayed = state.v + dv
    # print(p.tau_mem_inv)

    # clip values smaller than -1
    low_pass = torch.gt(v_decayed, -torch.ones_like(v_decayed))
    low_pass = 1. * low_pass
    v_decayed = v_decayed - (torch.ones_like(v_decayed) - low_pass) * (v_decayed + torch.ones_like(v_decayed))

    # add input
    v_decayed = v_decayed + i_in

    # compute new spikes
    z_new = threshold(v_decayed - p.v_th, p.method, p.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset

    return z_new, LIFFeedForwardState(v=v_new)


def lif_feed_forward_step(
        input_spikes: torch.Tensor,
        state: LIFFeedForwardState,
        p: LIFParameters = LIFParameters(),
        dt: float = 1.0,
) -> Tuple[torch.Tensor, LIFFeedForwardState]:

    jit_params = LIFParametersJIT(
        tau_mem_inv=p.tau_mem_inv,
        v_leak=p.v_leak,
        v_th=p.v_th,
        v_reset=p.v_reset,
        method=p.method,
        alpha=torch.as_tensor(p.alpha),
    )

    z, state = _lif_feed_forward_step_jit(input_spikes, state, jit_params, dt)
    return z, state
