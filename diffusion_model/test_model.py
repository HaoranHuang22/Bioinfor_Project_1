from coordinate_diffusion_model import *
from structure_diffusion_model import rigidFrom3Points
from pytorch3d.transforms import quaternion_multiply, quaternion_to_matrix
import roma
from scipy.spatial import geometric_slerp
import numpy as np

def test_Unet_input_output_dim():
    x = torch.randn(128, 1, 20, 3)
    t = torch.randint(0, 1000, (128,))
    model = UNet()
    out = model(x, t)
    assert x.shape == out.shape, 'Input tensor and output tensor of network should be same.'

def test_roma_slerp_and_scipy_Slerp():
    q0 = [1, 0, 0, 0]
    q1 = [0, 0, 0, 1]

    q0_scipy = np.array(q0)
    q1_scipy = np.array(q1)
    q0_roma = torch.Tensor(q0)
    q1_roma = torch.Tensor(q1)

    t_vals_scipy = np.linspace(0, 1, 4)
    t_vals_roma = torch.linspace(0, 1, 4)

    scipy_slerp = geometric_slerp(q0_scipy, q1_scipy, t_vals_scipy)
    roma_slerp = roma.utils.unitquat_slerp(q0_roma, q1_roma, t_vals_roma)
    print(scipy_slerp)
    print(roma_slerp)

    scipy_slerp = torch.from_numpy(scipy_slerp)


    assert torch.linalg.norm(scipy_slerp - roma_slerp) <= 1e-7

def test_roma_slerp_dim():
    q0, q1 = roma.random_unitquat(size=(128,20)), roma.random_unitquat(size=(128, 20))
    steps = torch.linspace(0, 1.0, 128)
    t_length = len(steps)
    
    q_interpolated = roma.utils.unitquat_slerp(q0, q1, steps) #(t_size, batch_size, 20, 4)
    result = torch.empty(128, 20, 4)

    for t in range(t_length):
        batch = q_interpolated[t, t % 128]
        result[t] = batch
    
    print(q_interpolated[1, 1, :, :])
    print(result[1])
    assert torch.linalg.norm(q_interpolated[0, 0, :, :] - result[0]) == 0

def test_quaternion_to_matrix_dim():
    quaternions = roma.random_unitquat(size=(128,20))
    rotations = quaternion_to_matrix(quaternions)
    assert rotations.shape == torch.Size([128, 20, 3, 3])

def test_rigidFrom3Points_orthogonal():
    x1 = torch.randn(128, 20, 3)
    x2 = torch.randn(128, 20, 3)
    x3 = torch.randn(128, 20, 3)

    R, t = rigidFrom3Points(x1, x2, x3)

    e1 = R[:, :, :, 0]
    e2 = R[:, :, :, 1]
    e3 = R[:, :, :, 2]

    inner_product_e1_e2 = torch.einsum('bij,bij->bi', e1, e2)
    inner_product_e1_e3 = torch.einsum('bij,bij->bi', e1, e3)
    inner_product_e2_e3 = torch.einsum('bij,bij->bi', e2, e3)

    assert torch.sum(inner_product_e1_e2) < 1e-5
    assert torch.sum(inner_product_e1_e3) < 1e-5
    assert torch.sum(inner_product_e2_e3) < 1e-5