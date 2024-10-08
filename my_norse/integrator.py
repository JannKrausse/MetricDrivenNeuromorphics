"""This code is taken from the Norse library (https://github.com/norse/norse) due to problems when importing it as a
whole."""
import torch
import torch.jit

from typing import NamedTuple, Tuple


class IntegratorState(NamedTuple):
    v: torch.Tensor


class IntegratorParameters(NamedTuple):
    v_leak: torch.Tensor = torch.as_tensor(0.0)


def integrator_step(
    input_spikes: torch.Tensor,
    state: IntegratorState,
    input_weights: torch.Tensor,
    p: IntegratorParameters = IntegratorParameters(),
    dt: float = 1.0,
) -> Tuple[torch.Tensor, IntegratorState]:

    # compute current jumps
    i_in = torch.nn.functional.linear(input_spikes, input_weights)

    # compute voltage updates
    dv = 0
    v_new = state.v + dv
    v_new = v_new + i_in

    return v_new, IntegratorState(v_new)


# @torch.jit.script
def integrator_feed_forward_step(
    input_tensor: torch.Tensor,
    state: IntegratorState,
    p: IntegratorParameters = IntegratorParameters(),
    dt: float = 1.0,
) -> Tuple[torch.Tensor, IntegratorState]:

    # compute voltage updates
    dv = 0
    v_new = state.v + dv
    v_new = v_new + input_tensor

    return v_new, IntegratorState(v_new)
