import numpy as np
import torch
from PIL import Image
from retinatrust.data.transforms import get_eval_transforms, get_train_transforms

def test_train_transforms_shape_and_type():
    img_array = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    image = Image.fromarray(img_array)

    transform = get_train_transforms(input_size=(224, 224))
    tensor = transform(image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32

def test_eval_transforms_deterministic():
    img_array = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    image = Image.fromarray(img_array)

    transform = get_eval_transforms(input_size=(224, 224))
    tensor1 = transform(image)
    tensor2 = transform(image)

    assert torch.allclose(tensor1, tensor2)
    assert tensor1.shape == (3, 224, 224)
