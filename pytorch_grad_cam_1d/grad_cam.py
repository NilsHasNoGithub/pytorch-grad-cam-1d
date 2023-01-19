import numpy as np
from pytorch_grad_cam_1d.base_cam import BaseCAM


class GradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False, normalize_cam_image=True,
                 reshape_transform=None, **base_cam_kwargs):
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform,
            normalize_cam_image=normalize_cam_image, **base_cam_kwargs)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.mean(grads, axis=2)
