"""This code is taken from the Norse library (https://github.com/norse/norse) due to problems when importing it as a
whole."""
import torch

from my_norse.lif import (
    lif_feed_forward_step,
    LIFFeedForwardState
)
from my_norse.integrator import (
    integrator_feed_forward_step,
    IntegratorState,
)

from typing import Any, Callable, List, Optional, Tuple

FeedforwardActivation = Callable[
    # Input        State         Parameters       dt
    [torch.Tensor, torch.Tensor, torch.nn.Module, float],
    Tuple[torch.Tensor, torch.Tensor],
]

RecurrentActivation = Callable[
    # Input        State         Input weights Rec. weights  Parameters       dt
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.nn.Module, float],
    Tuple[torch.Tensor, torch.Tensor],
]


def _merge_states(states: List[Any]):
    """
    Merges states recursively by using :method:`torch.stack` on individual state variables to
    produce a single output tuple, with an extra outer dimension.

    Arguments:
        states (List[Tuple]): The input list of states to merge

    Return:
        A single state of the same type as the first state in the input list of states, but with
        its members replaced with a stacked version of the members from the input states.
    """
    state_dict = states[0]._asdict()
    cls = states[0].__class__
    keys = list(state_dict.keys())
    tuples = [isinstance(s, tuple) for s in state_dict.values()]
    output_dict = {}
    for key, nested in zip(keys, tuples):
        if nested:
            nested_list = [getattr(s, key) for s in states]
            output_dict[key] = _merge_states(nested_list)
        else:
            values = [getattr(s, key) for s in states]
            output_dict[key] = torch.stack(values)
    return cls(**output_dict)


class SNN(torch.nn.Module):
    def __init__(
        self,
        activation: FeedforwardActivation,
        state_fallback: Callable[[torch.Tensor], torch.Tensor],
        p: Any,
        dt: float = 1.0,
        activation_sparse: Optional[FeedforwardActivation] = None,
        record_states: bool = False,
    ):
        super().__init__()
        self.activation = activation
        self.activation_sparse = activation_sparse
        self.state_fallback = state_fallback
        self.p = p
        self.dt = dt
        self.record_states = record_states

    def extra_repr(self) -> str:
        return f"p={self.p}, dt={self.dt}"

    def forward(self, input_tensor: torch.Tensor, state: Optional[Any] = None):
        state = state if state is not None else self.state_fallback(input_tensor)

        T = input_tensor.shape[0]
        outputs = []
        states = []

        activation = (
            self.activation_sparse
            if self.activation_sparse is not None and input_tensor.is_sparse
            else self.activation
        )

        for ts in range(T):
            out, state = activation(
                input_tensor[ts],
                state,
                self.p,
                self.dt,
            )
            outputs.append(out)
            if self.record_states:
                states.append(state)

        return torch.stack(outputs), state if not self.record_states else _merge_states(
            states
        )


class LIFParameters:
    tau_mem_inv: torch.Tensor = torch.as_tensor(1 / 100)
    v_leak: torch.Tensor = torch.as_tensor(0.0)
    v_th: torch.Tensor = torch.as_tensor(1.0)
    v_reset: torch.Tensor = torch.as_tensor(0.0)
    method: str = "heaviside"
    alpha: float = torch.as_tensor(100.0)


class IntegratorParameters:
    v_leak: torch.Tensor = torch.as_tensor(0.0)


class Integrator(SNN):
    def __init__(self, p: IntegratorParameters = IntegratorParameters(), **kwargs):
        super().__init__(
            activation=integrator_feed_forward_step,
            state_fallback=self.initial_state,
            p=p,
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> IntegratorState:
        state = IntegratorState(
            v=torch.full(
                input_tensor.shape[1:],  # Assume first dimension is time
                self.p.v_leak.detach(),
                device=input_tensor.device,
                dtype=torch.float32,
            ),
        )
        state.v.requires_grad = True
        return state


class LIF(SNN):
    def __init__(self, p: LIFParameters, **kwargs):
        super().__init__(
            activation=lif_feed_forward_step,
            activation_sparse=None,
            state_fallback=self.initial_state,
            p=p,
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIFFeedForwardState:
        state = LIFFeedForwardState(
            v=torch.full(
                input_tensor.shape[1:],  # Assume first dimension is time
                torch.as_tensor(self.p.v_leak).detach(),
                device=input_tensor.device,
                dtype=torch.float32,
            ),
        )
        state.v.requires_grad = True
        return state
