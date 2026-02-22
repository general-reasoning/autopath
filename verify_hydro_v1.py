
import torch
from autopath.models.hydro import Hydro

def test_model_vit():
    print("Testing Hydro Model_ViT (ViT)...")
    
    # 1. Setup config
    cfg = Hydro.CONFIG(
        model='vit',
        latent_dim=1536,
        image_size=256,
        vit_patch_size=16,
        vit_n_layers=2, # Use fewer layers for faster verification
        vit_hidden_channels=128,
        vit_n_heads=4,
        vit_dim_feedforward=512
    )
    
    # 2. Instantiate model directly to avoid Datablock metadata overhead
    model = Hydro.Model_ViT(
        latent_dim=1536,
        image_size=256,
        hidden_channels=128,
        patch_size=16,
        n_layers=2,
        n_heads=4,
        dim_feedforward=512
    )
    model.train()
    
    # 3. Dummy input
    batch_size = 2
    latent = torch.randn(batch_size, 1536)
    
    # 4. Forward pass
    print(f"Input shape: {latent.shape}")
    output = model(latent)
    print(f"Output shape: {output.shape}")
    
    # 5. Assertions
    assert output.shape == (batch_size, 3, 256, 256), f"Expected shape (2, 3, 256, 256), got {output.shape}"
    assert output.min() >= 0.0 and output.max() <= 255.0, f"Output values out of range [0, 255]: {output.min()}, {output.max()}"
    
    # 6. Backward pass (gradient check)
    target = torch.randn(batch_size, 3, 256, 256) * 255.0
    loss = model.loss(latent, target)
    print(f"Loss: {loss.item()}")
    loss.backward()
    
    # Check if gradients exist in projection weights
    assert model.proj.weight.grad is not None, "Gradients did not propagate to projection layer"
    print("Gradients propagated successfully.")
    
    print("Hydro Model_ViT verification PASSED!")

def test_model_cnn_lpips():
    print("\nTesting Hydro Model_CNN (CNN) with LPIPS loss...")
    spec = dict(
        model='cnn',
        latent_dim=1536,
        image_size=256,
        cnn_initial_size=8,
        cnn_hidden_channels=32,
        loss_type="lpips"
    )
    hydro = Hydro(spec=spec)
    model = hydro.model()
    model.train()
    
    z = torch.randn(2, 1536)
    target = torch.randn(2, 3, 256, 256) * 127.5 + 127.5
    target = target.clamp(0, 255)
    
    loss = model.loss(z, target)
    print(f"LPIPS Loss value: {loss.item():.6f}")
    loss.backward()
    print("Backward pass successful.")

def test_model_vit_lpips():
    print("\nTesting Hydro Model_ViT (ViT) with LPIPS loss...")
    spec = dict(
        model='vit',
        latent_dim=1536,
        image_size=256,
        vit_patch_size=16,
        vit_n_layers=2,
        vit_hidden_channels=128,
        vit_n_heads=4,
        vit_dim_feedforward=512,
        loss_type="lpips"
    )
    hydro = Hydro(spec=spec)
    model = hydro.model()
    model.train()
    
    z = torch.randn(2, 1536)
    target = torch.randn(2, 3, 256, 256) * 127.5 + 127.5
    target = target.clamp(0, 255)
    
    loss = model.loss(z, target)
    print(f"LPIPS Loss value: {loss.item():.6f}")
    loss.backward()
    print("Backward pass successful.")

if __name__ == "__main__":
    test_model_vit()
    test_model_cnn_lpips()
    test_model_vit_lpips()
