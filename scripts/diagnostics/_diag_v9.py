"""Quick diagnostic for V9 full-scale initial loss values."""
import torch
import torch.nn.functional as F
from ncalm.models.v9.model import AENCAv9Model
from ncalm.profiles import PROFILES

cfg = PROFILES['pile_250m_v9']
model = AENCAv9Model(cfg)
print(f'd_model={model.config.d_model}, d_concept={model.config.d_concept}, vocab={model.config.vocab_size}')
print(f'dilations={model._dilations}, RF={model.config.receptive_field}')
print(f'total params={sum(p.numel() for p in model.parameters()):,}')

# Test forward at init (CPU fp32)
batch = torch.randint(0, model.config.vocab_size, (2, 64))
loss, info = model.forward_train(batch)
print(f'CE={info["ce"]:.4f}, perm={info["perm"]:.6f}, spectral={info["spectral"]:.4f}')
print(f'total_loss={loss.item():.4f}')
print(f'grad flows: {loss.requires_grad}')
loss.backward()
grad_ok = all(p.grad is not None and not torch.isnan(p.grad).any() for p in model.parameters() if p.requires_grad)
print(f'All gradients valid (no NaN): {grad_ok}')

with torch.no_grad():
    logits = model.get_logits(batch)
    print(f'logits min={logits.min():.3f} max={logits.max():.3f} std={logits.std():.3f}')

# Test with AMP (simulating GPU behavior)
if torch.cuda.is_available():
    model2 = AENCAv9Model(cfg).cuda()
    batch2 = batch.cuda()
    model2.zero_grad()
    with torch.cuda.amp.autocast():
        loss_amp, info_amp = model2.forward_train(batch2)
    print(f'[AMP] CE={info_amp["ce"]:.4f}, perm={info_amp["perm"]:.6f}, spectral={info_amp["spectral"]:.4f}')
    print(f'[AMP] total_loss={loss_amp.item():.4f}')
    scaler = torch.cuda.amp.GradScaler()
    scaler.scale(loss_amp).backward()
    scaler.unscale_(torch.optim.AdamW(model2.parameters(), lr=1e-4))
    grad_ok_amp = all(p.grad is not None and not torch.isnan(p.grad).any() for p in model2.parameters() if p.requires_grad)
    print(f'[AMP] All gradients valid (no NaN): {grad_ok_amp}')
