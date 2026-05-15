from dataclasses import dataclass
from typing import Tuple, List, Callable, Any

import torch as th



class TensorDebugger:

    @dataclass
    class Condition:
        inf: bool = True
        nan: bool = True
        rng: Tuple[float | None, float | None] = (None, None)
    

    def __init__(
            self,
            inf: bool = True,
            nan: bool = True,
            rng: Tuple[float | None, float | None] = (None, None),
        ):
        self.cond = self.Condition(inf, nan, rng)
        self._gateways = {}
    

    @staticmethod
    def _to_safe(x: th.Tensor, cond: Condition) -> th.Tensor:
        x_safe = th.ones_like(x, dtype=th.bool)

        if cond.inf:
            x_safe.logical_and_(x.isinf().logical_not_())
        if cond.nan:
            x_safe.logical_and_(x.isnan().logical_not_())
        
        if cond.rng[0] is not None:
            x_safe.logical_and_(cond.rng[0] <= x)
        if cond.rng[1] is not None:
            x_safe.logical_and_(x <= cond.rng[1])
        
        return x_safe
    

    @staticmethod
    def _is_safe(x_safe: th.Tensor) -> bool:
        return x_safe.all().item()
    

    @staticmethod
    def _autofill(x: th.Tensor, x_safe: th.Tensor, val):
        x[x_safe.logical_not()] = val
    

    def to_safe(self, x: th.Tensor) -> th.Tensor:
        return self._to_safe(x, self.cond)
    

    def is_safe(self, x: th.Tensor) -> bool:
        return self._is_safe(self._to_safe(x, self.cond))
    

    def autofill(self, x: th.Tensor, val):
        self._autofill(x, self._to_safe(x, self.cond), val)
    

    def register_gateway(
            self,
            id: str | int,
            inf: bool = None,
            nan: bool = None,
            rng: Tuple[float | None, float | None] = None,
            val = None,
            callbacks: List[Callable[[th.Tensor], Any]] = None,
        ):
        if callbacks is None:
            callbacks = []
        
        # check whether key already exists
        if id in self._gateways:
            raise KeyError(f'ID {id} is already used.')
        
        self._gateways[id] = {
            'cond': self.Condition(
                self.cond.inf if inf is None else inf,
                self.cond.nan if nan is None else nan,
                self.cond.rng if rng is None else rng,),
            'val': val,
            'callbacks': callbacks,
        }


    def run_gateway(
            self,
            id: str | int,
            x: th.Tensor | List[th.Tensor],
        ):
        if isinstance(x, th.Tensor):
            x = [x]
        
        gateway = self._gateways[id]
        cond = gateway['cond']
        val = gateway['val']
        callbacks = gateway['callbacks']

        for xi in x:
            xi_safe = self._to_safe(xi, cond)
            xi_is_safe = self._is_safe(xi_safe)

            if not xi_is_safe:
                for callback in callbacks:
                    callback(xi)
            
            if val is not None:
                self._autofill(xi, xi_safe, val)
