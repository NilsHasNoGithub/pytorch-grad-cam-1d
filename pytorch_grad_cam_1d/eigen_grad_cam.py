from pytorch_grad_cam_1d.base_cam import BaseCAM
from pytorch_grad_cam_1d.utils.svd_on_activations import get_projection

# Like Eigen CAM: https://arxiv.org/abs/2008.00299
# But multiply the activations x gradients


class EigenGradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        raise NotImplementedError("Not adapted for 1d")
        super(EigenGradCAM, self).__init__(model, target_layers, use_cuda,
                                           reshape_transform)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_projection(grads * activations)
