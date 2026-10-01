"""Analyze V9 FLOP budget for different step counts and 300B scaling."""
import math

d, dc, V = 2048, 9255, 50304
print("=== V9 Current (3 steps) ===")
flops_per_step = 3 * 2 * d * dc
flops_embed = 2 * d * V
flops_3 = 3 * flops_per_step + flops_embed
print(f"FLOPs/tok (3 steps): {flops_3/1e6:.1f}M")
print(f"Kernel params: {3*d*dc/1e6:.1f}M, Embed: {V*d/1e6:.1f}M")

print("\n=== V9 with 6 steps (SAME params, just more iterations) ===")
flops_6 = 6 * flops_per_step + flops_embed
print(f"FLOPs/tok: {flops_6/1e6:.1f}M")
print(f"Expected tok/s: {320/flops_6*1e6*49779:.0f}")

print("\n=== V9 with 12 steps ===")
flops_12 = 12 * flops_per_step + flops_embed
print(f"FLOPs/tok: {flops_12/1e6:.1f}M")
print(f"Expected tok/s: {320/flops_12*1e6*49779:.0f}")

# KEY INSIGHT: with shared weights, more steps = MORE FLOPs but SAME params
# This is the CA advantage: you can trade off throughput for quality
# without changing model size

print("\n=== Scaling to 300B ===")
V300 = 128256
for n_steps in [6, 12, 24]:
    # 3*d*dc + V*d = 300e9
    # Let dc = 4*d: 12*d^2 + V*d = 300e9
    a, b, c = 12, V300, -300_000_000_000
    d_est = int((-b + math.sqrt(b*b - 4*a*c)) / (2*a))
    dc_est = (300_000_000_000 - V300 * d_est) // (3 * d_est)
    flops = n_steps * 3 * 2 * d_est * dc_est + 2 * d_est * V300
    embed_frac = V300 * d_est / 300e9 * 100
    print(f"n_steps={n_steps}: d={d_est:,}, dc={dc_est:,}, "
          f"FLOPs/tok={flops/1e9:.1f}B, embed={embed_frac:.1f}%")
