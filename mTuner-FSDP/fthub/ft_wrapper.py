import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, ActivationWrapper
from torch.utils.checkpoint import checkpoint as torch_utils_checkpoint
from torch.distributed.utils import _pack_kwargs, _unpack_kwargs
from functools import partial

def ft_checkpoint_wrapper(
    module: torch.nn.Module,
    checkpoint_impl: CheckpointImpl = CheckpointImpl.REENTRANT,
    checkpoint_fn=None,
    checkpoint_config=None,
    **checkpoint_fn_kwargs,
) -> torch.nn.Module:

    assert checkpoint_impl == CheckpointImpl.NO_REENTRANT
    return FTCheckpointWrapper(
        module,
        checkpoint_impl,
        checkpoint_fn,
        checkpoint_config,
        **checkpoint_fn_kwargs,
    )

class FTCheckpointWrapper(ActivationWrapper):

    def __init__(
        self,
        mod: torch.nn.Module,
        checkpoint_impl: CheckpointImpl = CheckpointImpl.REENTRANT,
        checkpoint_fn=None,
        checkpoint_config=None,
        **checkpoint_fn_kwargs,
    ):
        super().__init__(mod)
        self.group_id = mod.group_id
        self.num_groups = mod.num_groups
        self.checkpoint_impl = checkpoint_impl
        self.checkpoint_config = checkpoint_config
        if checkpoint_fn is None:
            # use torch.utils.checkpoint
            self.checkpoint_fn = partial(
                torch_utils_checkpoint,
                use_reentrant=(self.checkpoint_impl == CheckpointImpl.REENTRANT),
                **checkpoint_fn_kwargs,
            )
        else:
            # Construct user-specified checkpoint function.
            self.checkpoint_fn = partial(
                checkpoint_fn,
                **checkpoint_fn_kwargs,
            )

    def forward_impl(self, *args, **kwargs):
        # Support keyword arguments for reentrant checkpoint. Note that this
        # only works if user has specified self.checkpoint_impl and is not
        # using their own custom checkpoint_fn.
        if self.checkpoint_impl == CheckpointImpl.REENTRANT and kwargs != {}:
            # Pack the args and kwargs
            flat_args, kwarg_keys = _pack_kwargs(*args, **kwargs)

            # Function that only takes (packed) args, but can unpack them
            # into the original args and kwargs for the checkpointed
            # function, and runs that function.
            def my_function(*inputs):
                # unpack back into args and kwargs
                unpacked_args, unpacked_kwargs = _unpack_kwargs(inputs, kwarg_keys)
                # run original module
                return self._checkpoint_wrapped_module(
                    *unpacked_args, **unpacked_kwargs
                )

            # Pass the function that only takes packed args into reentrant
            # checkpoint API.
            return self.checkpoint_fn(  # type: ignore[misc]
                my_function,
                *flat_args,
            )
        else:
            return self.checkpoint_fn(  # type: ignore[misc]
                self._checkpoint_wrapped_module, *args, **kwargs
            )

    def forward(self, *args, **kwargs):
        if self.num_groups - self.group_id <= self.checkpoint_config.non_offload_num:
            return self.forward_impl(*args, **kwargs)
        else:
            with torch.autograd.graph.save_on_cpu():
                return self.forward_impl(*args, **kwargs)