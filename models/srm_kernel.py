import torch
import numpy as np

def setup_srm_weights(input_channels: int = 3) -> torch.Tensor:
    """Creates the SRM kernels for noise analysis."""
    # note: values taken from Zhou et al., "Learning Rich Features for Image Manipulation Detection", CVPR2018
    srm_kernel = torch.from_numpy(np.array([
        [  # srm 1/2 horiz
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., 1., -2., 1., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
        ], [  # srm 1/4
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., -1., 2., -1., 0.],  # noqa: E241,E201
            [0., 2., -4., 2., 0.],  # noqa: E241,E201
            [0., -1., 2., -1., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
        ], [  # srm 1/12
            [-1., 2., -2., 2., -1.],  # noqa: E241,E201
            [2., -6., 8., -6., 2.],  # noqa: E241,E201
            [-2., 8., -12., 8., -2.],  # noqa: E241,E201
            [2., -6., 8., -6., 2.],  # noqa: E241,E201
            [-1., 2., -2., 2., -1.],  # noqa: E241,E201
        ]
    ])).float()
    srm_kernel[0] /= 2
    srm_kernel[1] /= 4
    srm_kernel[2] /= 12
    return srm_kernel.view(3, 1, 5, 5).repeat(1, input_channels, 1, 1)


def setup_srm_layer(input_channels: int = 3) -> torch.nn.Module:
    """Creates a SRM convolution layer for noise analysis."""
    weights = setup_srm_weights(input_channels)
    srm_conv = torch.nn.Conv2d(input_channels, out_channels=3, kernel_size=5, stride=1, padding=2, bias=False, )
    with torch.no_grad():
        srm_conv.weight = torch.nn.Parameter(weights, requires_grad=False)
    return srm_conv