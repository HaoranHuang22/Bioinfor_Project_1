from coordinate_diffusion_model import *

def test_Unet_input_output_dim():
    x = torch.randn(128, 1, 20, 3)
    t = torch.randint(0, 1000, (128,))
    model = UNet()
    out = model(x, t)
    assert x.shape == out.shape, 'Input tensor and output tensor of network should be same.'
