import torch
from autopath.models.hydro import Hydro

def test_hydro_improvements():
    latent_dim = 1536
    image_size = 256
    batch_size = 2
    
    latent = torch.randn(batch_size, latent_dim)
    
    # 1. Test Baseline (Default)
    print("Testing Baseline...")
    hydro_baseline = Hydro(spec={'latent_dim': latent_dim, 'image_size': image_size})
    model_baseline = hydro_baseline.model()
    out_baseline = model_baseline(latent)
    assert out_baseline.shape == (batch_size, 3, image_size, image_size)
    print("Baseline OK.")

    # 2. Test Residual Upsampling
    print("Testing Residual Upsampling...")
    hydro_res = Hydro(spec={
        'latent_dim': latent_dim, 
        'image_size': image_size, 
        'use_residual_upsampling': True
    })
    model_res = hydro_res.model()
    out_res = model_res(latent)
    assert out_res.shape == (batch_size, 3, image_size, image_size)
    print("Residual Upsampling OK.")

    # 3. Test Attention Gates
    print("Testing Attention Gates...")
    hydro_att = Hydro(spec={
        'latent_dim': latent_dim, 
        'image_size': image_size, 
        'use_attention_gates': True
    })
    model_att = hydro_att.model()
    out_att = model_att(latent)
    assert out_att.shape == (batch_size, 3, image_size, image_size)
    print("Attention Gates OK.")

    # 4. Test Feature Modulation
    print("Testing Feature Modulation...")
    hydro_mod = Hydro(spec={
        'latent_dim': latent_dim, 
        'image_size': image_size, 
        'use_feature_modulation': True
    })
    model_mod = hydro_mod.model()
    out_mod = model_mod(latent)
    assert out_mod.shape == (batch_size, 3, image_size, image_size)
    print("Feature Modulation OK.")

    # 5. Test All Combined
    print("Testing All Combined...")
    hydro_all = Hydro(spec={
        'latent_dim': latent_dim, 
        'image_size': image_size, 
        'use_residual_upsampling': True,
        'use_attention_gates': True,
        'use_feature_modulation': True
    })
    model_all = hydro_all.model()
    out_all = model_all(latent)
    assert out_all.shape == (batch_size, 3, image_size, image_size)
    print("All Combined OK.")

if __name__ == "__main__":
    test_hydro_improvements()
