import torch
from pytorch_grad_cam_1d.base_cam import BaseCAM


class ScoreCAM(BaseCAM):
    def __init__(
            self,
            model,
            target_layers,
            use_cuda=False,
            reshape_transform=None,
            normalize_cam_image:bool=True, **base_cam_kwargs):
        super(ScoreCAM, self).__init__(model,
                                       target_layers,
                                       use_cuda,
                                       reshape_transform=reshape_transform,
                                       uses_gradients=False,
                                       normalize_cam_image=normalize_cam_image, **base_cam_kwargs)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        targets,
                        activations,
                        grads):
        with torch.no_grad():
            upsample = torch.nn.Upsample(
                size=input_tensor.shape[-1:], mode="linear")
            activation_tensor = torch.from_numpy(activations)
            if self.cuda:
                activation_tensor = activation_tensor.cuda()

            upsampled = upsample(activation_tensor)

            maxs = upsampled.view(upsampled.size(0), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0), -1).min(dim=-1)[0]

            maxs, mins = maxs[:, None], mins[:, None]
            upsampled = (upsampled - mins) / (maxs - mins)

            input_tensors = input_tensor[:, None,
                                         :] * upsampled[:, :, None, :]

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16

            scores = []
            for target, tensor in zip(targets, input_tensors):
                for i in range(0, tensor.size(0), BATCH_SIZE):
                    batch = tensor[i: i + BATCH_SIZE, :]
                    outputs = [target(o).cpu().item()
                               for o in self.model(batch)]
                    scores.extend(outputs)
            scores = torch.Tensor(scores)
            scores = scores.view(activations.shape[0], activations.shape[1])
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return weights
