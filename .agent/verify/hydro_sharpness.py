import torch
import torch.nn as nn
from autopath.models.hydro import Hydro, PerceptualLoss, GradientLoss

def test_loss_scaling():
    print("Testing Loss Scaling and GradientLoss...")
    lpips = PerceptualLoss()
    grad_loss = GradientLoss()
    
    # Dummy images [0, 255]
    x = torch.rand(1, 3, 256, 256) * 255.0
    y = x + torch.randn_like(x) * 5.0 # Add some noise
    
    l_lpips = lpips(x, y)
    l_grad = grad_loss(x, y)
    
    print(f"LPIPS loss: {l_lpips.item():.6f}")
    print(f"Gradient loss: {l_grad.item():.6f}")
    
    assert l_lpips.item() > 0, "LPIPS loss should be positive"
    assert l_grad.item() > 0, "Gradient loss should be positive"
    print("Perceptual and Gradient losses initialized and computed successfully.")

def test_model_arch():
    print("\nTesting Model Architecture (CNN with PixelShuffle and Residual)...")
    # Test CNN with PixelShuffle and Residual
    cfg = Hydro.CONFIG(
        model='cnn',
        cnn_use_pixel_shuffle=True,
        cnn_use_residual=True,
        cnn_use_spatial_attention_gates=True,
        cnn_initial_size=8,
        image_size=64, # Small size for quick test
        cnn_hidden_channels=64,
        cnn_channel_multipliers=(4, 2, 1) # 8 -> 16 -> 32 -> 64
    )
    
    model = Hydro(spec=dict(**cfg.__dict__))
    cnn_model = model.model()
    
    latent = torch.randn(1, 1536)
    output = cnn_model(latent)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (1, 3, 64, 64), f"Expected (1, 3, 64, 64), got {output.shape}"
    assert output.max() <= 255.0 and output.min() >= 0.0, "Output range should be [0, 255]"
    
    print("CNN with PixelShuffle and Residual forward pass successful.")

if __name__ == "__main__":
    try:
        test_loss_scaling()
        test_model_arch()
        print("\nAll verifications passed!")
    except Exception as e:
        print(f"\nVerification failed: {e}")
        import traceback
        traceback.print_exc()
