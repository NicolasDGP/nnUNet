from typing import Union, Tuple, List

from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerResSE(nnUNetTrainer):
    """nnU-Net v2 trainer variant that swaps the network class to a Residual+SE UNet.

    The macro-topology (stages/strides/features/skip wiring) is still dictated by the plans via the
    architecture kwargs. Only the network class name passed to `get_network_from_plans` is replaced.

    For this to work, the target network class must be importable and accept the same constructor
    kwargs as the original plans-defined architecture (typically PlainConvUNet for 3d_fullres).
    """

    # Fully-qualified import path used by pydoc.locate in get_network_from_plans
    RESSE_NETWORK_CLASS_NAME: str = "nnunetv2.custom_nets.unet_resse.UNetResSE3D"

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        # We ignore the plans-provided class name but keep all kwargs to preserve the plan-derived topology.
        return nnUNetTrainer.build_network_architecture(
            nnUNetTrainerResSE.RESSE_NETWORK_CLASS_NAME,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )