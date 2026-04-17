# 🔥 PyTorch Cheatsheet
### Essential PyTorch Patterns for AI Engineers — AI Magic Mastery × CodeBeez

---

## Installation
```bash
# CUDA (check your CUDA version first: nvidia-smi)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio

# Verify
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## Tensors

```python
import torch

# ── Creation ──────────────────────────────────────────────
torch.tensor([1, 2, 3])                  # From Python list
torch.zeros(3, 4)                        # 3×4 zeros
torch.ones(3, 4)                         # 3×4 ones
torch.eye(3)                             # 3×3 identity
torch.rand(3, 4)                         # Uniform [0, 1)
torch.randn(3, 4)                        # Normal(0, 1)
torch.arange(0, 10, 2)                   # [0, 2, 4, 6, 8]
torch.linspace(0, 1, 5)                  # [0, .25, .5, .75, 1]
torch.full((3, 4), fill_value=7)         # 3×4 filled with 7

# ── Device ────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(3, 4).to(device)
x = torch.randn(3, 4, device=device)    # Direct creation on GPU

# ── Dtype ─────────────────────────────────────────────────
x = torch.randn(3, 4, dtype=torch.float16)  # Half precision
x = x.float()     # → float32
x = x.long()      # → int64
x = x.bool()      # → boolean

# ── Shape Operations ──────────────────────────────────────
x = torch.randn(2, 3, 4)
x.shape               # torch.Size([2, 3, 4])
x.numel()             # 24 (total elements)
x.reshape(6, 4)       # Reshape (shares memory if possible)
x.view(6, 4)          # Reshape (must be contiguous)
x.permute(2, 0, 1)    # Reorder dimensions → [4, 2, 3]
x.transpose(0, 1)     # Swap two dimensions → [3, 2, 4]
x.squeeze(0)          # Remove dim of size 1
x.unsqueeze(0)        # Add dim of size 1 → [1, 2, 3, 4]
x.flatten()           # Flatten to 1D
x.contiguous()        # Make contiguous in memory

# ── Indexing & Slicing ────────────────────────────────────
x[0]                  # First element
x[:, 1]              # All rows, column 1
x[0:2, :, -1]        # Fancy slicing
x[x > 0]             # Boolean indexing
torch.where(x > 0, x, torch.zeros_like(x))  # Conditional

# ── Math Operations ───────────────────────────────────────
a + b                 # Element-wise add (same as torch.add)
a * b                 # Element-wise multiply
a @ b                 # Matrix multiply (same as torch.mm for 2D)
torch.matmul(a, b)    # Batched matrix multiply
a.T                   # Transpose
torch.sum(x)          # Sum all elements
torch.sum(x, dim=0)   # Sum along dimension 0
torch.mean(x)
torch.std(x)
torch.max(x)
torch.argmax(x)       # Index of max value
torch.topk(x, k=3)   # Top-k values and indices
torch.cat([a, b], dim=0)    # Concatenate along dim
torch.stack([a, b], dim=0)  # Stack (adds new dim)
```

---

## Autograd & Gradients

```python
# ── Requires Grad ─────────────────────────────────────────
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2 + 3 * x + 1
y.sum().backward()   # Compute gradients
print(x.grad)        # dy/dx = 2x + 3 → [7., 9.]

# ── No Grad Context ───────────────────────────────────────
with torch.no_grad():          # No gradient computation
    output = model(input)

@torch.no_grad()               # As decorator
def predict(x):
    return model(x)

# ── Detach ────────────────────────────────────────────────
z = x.detach()                 # New tensor, no grad history

# ── Gradient Accumulation ─────────────────────────────────
optimizer.zero_grad()
for i, (x, y) in enumerate(loader):
    loss = criterion(model(x), y) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Building Models

```python
import torch.nn as nn

# ── Simple MLP ────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_features, hidden, out_features, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_features)
        )
    
    def forward(self, x):
        return self.net(x)

# ── CNN ───────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

# ── Transformer Block ─────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.drop(attn_out))
        # Feed-forward with residual
        x = self.norm2(x + self.drop(self.ff(x)))
        return x

# ── Model Inspection ──────────────────────────────────────
model = MLP(784, 256, 10)
total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params:,} | Trainable: {trainable:,}")
```

---

## Training Loop

```python
# ── Standard Training Loop ────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        if scaler:  # Mixed precision
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = model(x)
                loss = criterion(output, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        total_loss += criterion(output, y).item()
        correct += (output.argmax(1) == y).sum().item()
        total += y.size(0)
    
    return total_loss / len(loader), correct / total

# ── Full Training Script ───────────────────────────────────
model = MyModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()  # For mixed precision

best_val_loss = float('inf')
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
    
    print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.3f}")
```

---

## Optimizers & Schedulers

```python
# Optimizers
torch.optim.Adam(model.parameters(), lr=1e-3)
torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)  # Adam + weight decay (preferred)
torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
torch.optim.RMSprop(model.parameters(), lr=1e-3)

# Schedulers
torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # Most popular
torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(loader), epochs=epochs)
torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)  # Adaptive
torch.optim.lr_scheduler.LinearWarmup + CosineDecay  # Standard for transformers
```

---

## Save & Load

```python
# ── Save / Load Model ─────────────────────────────────────
# Save weights only (recommended)
torch.save(model.state_dict(), 'model.pt')
model.load_state_dict(torch.load('model.pt', map_location=device))

# Save full checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pt')

# Load full checkpoint
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

---

## Common Loss Functions

```python
nn.CrossEntropyLoss()           # Classification (includes softmax)
nn.BCEWithLogitsLoss()          # Binary classification (includes sigmoid)
nn.MSELoss()                    # Regression
nn.L1Loss()                     # Regression (robust to outliers)
nn.HuberLoss()                  # Combination of MSE + L1
nn.NLLLoss()                    # Negative log likelihood
nn.CTCLoss()                    # Sequence alignment (speech, OCR)
nn.KLDivLoss()                  # Distribution matching (VAEs)
```

---

## Performance Tips

```python
# ── Speed Up Training ─────────────────────────────────────
# 1. Move data to GPU ASAP
x = x.to(device, non_blocking=True)

# 2. Use mixed precision (2x speed, half memory)
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(x)

# 3. Compile model (PyTorch 2.0+, 1.5–3x speedup)
model = torch.compile(model)

# 4. Enable TF32 for matrix ops (A100+)
torch.backends.cuda.matmul.allow_tf32 = True

# 5. Pin memory in DataLoader
DataLoader(dataset, pin_memory=True, num_workers=4)

# 6. Gradient checkpointing (memory ↓, speed ↓ slightly)
from torch.utils.checkpoint import checkpoint
output = checkpoint(layer, x)

# 7. Profile your code
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    model(x)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

*AI Magic Mastery by CodeBeez × Abid Redwan | [aimagicmastery.codebeez.ai](https://aimagicmastery.codebeez.ai)*
