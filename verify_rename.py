import torch
from autopath.models.hydro import Hydro, HydroLightning, HydroStill

def verify_rename():
    latent_dim = 1536
    image_size = 64
    
    # Test propagation with new name
    hydro = Hydro(spec={'latent_dim': latent_dim, 'image_size': image_size})
    lightning = HydroLightning(spec={
        'hydro': hydro,
        'cnn_use_spatial_attention_gates': True
    })
    
    # Trigger propagation
    _ = lightning.lightning_module
    assert hydro.cfg.cnn_use_spatial_attention_gates is True
    
    # Check Model_CNN instantiation
    model = lightning.lightning_module.decoder
    assert isinstance(model, Hydro.Model_CNN)
    assert model.use_spatial_attention_gates is True
    assert hasattr(model, 'attention_gates')
    print("Rename verification OK.")

if __name__ == "__main__":
    verify_rename()
