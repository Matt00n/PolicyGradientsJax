
import copy
import functools
from typing import Any, Dict, Optional, Sequence, Union

from flax import struct
import jax
from jax import numpy as jp
from jax import vmap
from jax.tree_util import tree_map



@struct.dataclass
class Base:
    """Base functionality extending all brax types.

    These methods allow for brax types to be operated like arrays/matrices.
    """

    def __add__(self, o: Any) -> Any:
        return tree_map(lambda x, y: x + y, self, o)

    def __sub__(self, o: Any) -> Any:
        return tree_map(lambda x, y: x - y, self, o)

    def __mul__(self, o: Any) -> Any:
        return tree_map(lambda x: x * o, self)

    def __neg__(self) -> Any:
        return tree_map(lambda x: -x, self)

    def __truediv__(self, o: Any) -> Any:
        return tree_map(lambda x: x / o, self)

    def reshape(self, shape: Sequence[int]) -> Any:
        return tree_map(lambda x: x.reshape(shape), self)

    def select(self, o: Any, cond: jp.ndarray) -> Any:
        return tree_map(lambda x, y: (x.T * cond + y.T * (1 - cond)).T, self, o)

    def slice(self, beg: int, end: int) -> Any:
        return tree_map(lambda x: x[beg:end], self)

    def take(self, i, axis=0) -> Any:
        return tree_map(lambda x: jp.take(x, i, axis=axis, mode='wrap'), self)

    def concatenate(self, *others: Any, axis: int = 0) -> Any:
        return tree_map(lambda *x: jp.concatenate(x, axis=axis), self, *others)

    def index_set(
        self, idx: Union[jp.ndarray, Sequence[jp.ndarray]], o: Any
    ) -> Any:
        return tree_map(lambda x, y: x.at[idx].set(y), self, o)

    def index_sum(
        self, idx: Union[jp.ndarray, Sequence[jp.ndarray]], o: Any
    ) -> Any:
        return tree_map(lambda x, y: x.at[idx].add(y), self, o)

    def vmap(self, in_axes=0, out_axes=0):
        """Returns an object that vmaps each follow-on instance method call."""

        # TODO: i think this is kinda handy, but maybe too clever?

        outer_self = self

        class VmapField:
            """Returns instance method calls as vmapped."""

            def __init__(self, in_axes, out_axes):
                self.in_axes = [in_axes]
                self.out_axes = [out_axes]

            def vmap(self, in_axes=0, out_axes=0):
                self.in_axes.append(in_axes)
                self.out_axes.append(out_axes)
                return self

            def __getattr__(self, attr):
                fun = getattr(outer_self.__class__, attr)
                # load the stack from the bottom up
                vmap_order = reversed(list(zip(self.in_axes, self.out_axes)))
                for in_axes, out_axes in vmap_order:
                    fun = vmap(fun, in_axes=in_axes, out_axes=out_axes)
                fun = functools.partial(fun, outer_self)
                return fun

        return VmapField(in_axes, out_axes)

    def tree_replace(
        self, params: Dict[str, Optional[jax.typing.ArrayLike]]
    ) -> 'Base':
        """Creates a new object with parameters set.

        Args:
        params: a dictionary of key value pairs to replace

        Returns:
        data clas with new values

        Example:
        If a system has 3 links, the following code replaces the mass
        of each link in the System:
        >>> sys = sys.tree_replace(
        >>>     {'link.inertia.mass', jp.array([1.0, 1.2, 1.3])})
        """
        new = self
        for k, v in params.items():
            new = _tree_replace(new, k.split('.'), v)
        return new

    @property
    def T(self):  # pylint:disable=invalid-name
        return tree_map(lambda x: x.T, self)
    

def _tree_replace(
    base: Base,
    attr: Sequence[str],
    val: Optional[jax.typing.ArrayLike],
) -> Base:
    """Sets attributes in a struct.dataclass with values."""
    if not attr:
        return base

    # special case for List attribute
    if len(attr) > 1 and isinstance(getattr(base, attr[0]), list):
        lst = copy.deepcopy(getattr(base, attr[0]))

        for i, g in enumerate(lst):
            if not hasattr(g, attr[1]):
                continue
            v = val if not hasattr(val, '__iter__') else val[i]
            lst[i] = _tree_replace(g, attr[1:], v)

        return base.replace(**{attr[0]: lst})

    if len(attr) == 1:
        return base.replace(**{attr[0]: val})

    return base.replace(
        **{attr[0]: _tree_replace(getattr(base, attr[0]), attr[1:], val)}
    )


@struct.dataclass
class State(Base):
    """Environment state for training and inference."""

    obs: jp.ndarray
    reward: jp.ndarray
    done: jp.ndarray
    metrics: Dict[str, jp.ndarray] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)