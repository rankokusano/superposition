"""
visualize_perspective.py

Replication of Noguchi et al. (2022) Fig. 5:
  "Can Process-2 reconstruct A-2's visual perspective?"

Paper procedure:
  1. Train a Visual Decoder to reconstruct A-1's view from f1_v (Process-1's
     visual feature).  This is a standard auto-encoder reconstruction task.
  2. Apply the SAME decoder to f2_v (Process-2's visual feature).
  3. If the decoded image resembles what A-2 actually sees → perspective-taking.

Layout (4 rows × N_SAMPLES columns):
  Row 1: A-1's actual vision  (input to the network)
  Row 2: Decoded from f1_v   (sanity check: should ≈ Row 1)
  Row 3: Decoded from f2_v   (Process-2's "imagined" view: should ≈ Row 4)
  Row 4: A-2's actual vision  (captured from simulation)

Run:
    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run python3 visualize_perspective.py

Output:
    step2/perspective_map.png
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(_HERE, '../../'))
MY_RESEARCH = os.path.abspath(os.path.join(_HERE, '../'))
for p in [_HERE, MY_RESEARCH, PROJ_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation import creator
from simulation.util import load_config
from config_util import load_config as load_exp_config
from rl_agent_sac import ActorLSTM, CriticLSTM
from agent2_green import GreenFollowerAgent
from model_q import SuperpositionNetworkWithQ
from model.modules import VisionDecoderModule   # reuse same decoder structure

# =========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH    = os.path.join(_HERE, 'model_q.pth')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
CRITIC_PATH        = os.path.join(MY_RESEARCH, 'rl_model_v6_critic.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')
SAVE_PATH          = os.path.join(_HERE, 'perspective_map.png')

DECODER_TRAIN_EPISODES = 20    # episodes to collect data for decoder training
DECODER_TRAIN_STEPS    = 60    # steps per episode
DECODER_EPOCHS         = 30    # auto-encoder training epochs
DECODER_LR             = 1e-3

N_SAMPLES = 5    # number of timesteps to visualize

Q_SCALE = 1.0

# =========================================================================
print(f"Device: {DEVICE}")

model_config = load_exp_config(IIZUKA_CONFIG_PATH)
net = SuperpositionNetworkWithQ(model_config).to(DEVICE)

if not os.path.exists(MODEL_SAVE_PATH):
    print("[WARNING] model_q.pth not found.")
    ckpt_iizuka = torch.load(
        os.path.join(PROJ_ROOT, 'data/result/exp1_l1/0/model/00200.pth'),
        map_location=DEVICE)
    net.load_iizuka_weights(ckpt_iizuka, DEVICE)
else:
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    net.load_state_dict(ckpt['net_state_dict'])
    Q_SCALE = ckpt.get('q_scale', 1.0)
    print(f"Loaded: episodes={ckpt.get('episodes','?')}, "
          f"final_loss={ckpt.get('final_loss', float('nan')):.5f}")

net.eval()
for p in net.parameters():
    p.requires_grad = False

critic = CriticLSTM().to(DEVICE)
critic.load_state_dict(torch.load(CRITIC_PATH, map_location=DEVICE))
critic.eval()

actor = ActorLSTM().to(DEVICE)
actor.load_state_dict(torch.load(ACTOR_PATH, map_location=DEVICE))
actor.eval()

env_config = load_config(ENV_CONFIG_PATH)
env = creator.create_environment(env_config.environment)
env.init()
env.off_display()

green_follower = GreenFollowerAgent()

# =========================================================================
# Step 1: Collect A-1 visual frames for decoder training
# =========================================================================
print(f"\nCollecting frames for decoder training "
      f"({DECODER_TRAIN_EPISODES} ep × {DECODER_TRAIN_STEPS} steps)...")

train_frames = []   # A-1's actual visual frames (BCHW tensors)

for ep in range(DECODER_TRAIN_EPISODES):
    env.reset()
    v_raw, _, _, sp_0, op_0 = env.step()
    self_pos  = sp_0.copy()
    other_pos = op_0.copy()
    actor_hidden = None

    for step in range(DECODER_TRAIN_STEPS):
        v_t = torch.FloatTensor(
            v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
        train_frames.append(v_t.squeeze(0).cpu())

        with torch.no_grad():
            action, _, actor_hidden = actor.sample(v_t, actor_hidden)
        a_np = action.squeeze(0).cpu().numpy()
        om_np = green_follower.get_action(other_pos)

        new_self  = np.clip(self_pos  + a_np,  -9.5, 9.5)
        new_other = np.clip(other_pos + om_np, -9.5, 9.5)
        env.self_agent.p  = new_self
        env.other_agent.p = new_other
        v_next_raw, _, _, _, _ = env.step()

        v_raw     = v_next_raw
        self_pos  = new_self
        other_pos = new_other

print(f"  Collected {len(train_frames)} frames.")

# =========================================================================
# Step 2: Train Visual Decoder (f1_v → A-1's current view)
# Paper: "Visual Decoder has the same structure as Visual Predictor"
# =========================================================================
convolved_shape = net.self_vision_encoder_module.get_convolved_shape()
decoder = VisionDecoderModule(
    model_config.vision_decoder_module, convolved_shape
).to(DEVICE)

decoder_optim = optim.Adam(decoder.parameters(), lr=DECODER_LR)

BATCH = 32
frame_tensor = torch.stack(train_frames)   # (N, 3, 16, 64)
N = len(frame_tensor)

print(f"\nTraining Visual Decoder ({DECODER_EPOCHS} epochs, batch={BATCH})...")
for epoch in range(DECODER_EPOCHS):
    idx = torch.randperm(N)
    epoch_loss = 0.0
    n_batches  = 0
    for start in range(0, N, BATCH):
        batch = frame_tensor[idx[start:start+BATCH]].to(DEVICE)
        with torch.no_grad():
            f1_v = net.self_vision_encoder_module(batch)   # (B, 64)
        recon = decoder(f1_v)                              # (B, 3, 16, 64)
        loss = F.l1_loss(recon, batch)
        decoder_optim.zero_grad()
        loss.backward()
        decoder_optim.step()
        epoch_loss += loss.item()
        n_batches  += 1
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/{DECODER_EPOCHS}  loss={epoch_loss/n_batches:.5f}")

decoder.eval()

# =========================================================================
# Step 3: Run one episode and collect samples for visualization
# =========================================================================
print(f"\nRunning visualization episode (sampling {N_SAMPLES} steps)...")

env.reset()
net.superposition_module.init_state(1)
v_raw, _, _, sp_0, op_0 = env.step()
self_pos  = sp_0.copy()
other_pos = op_0.copy()
actor_hidden  = None
critic_hidden = None

# Warm up for 10 steps before sampling
WARMUP = 10
for _ in range(WARMUP):
    v_t = torch.FloatTensor(v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        action, _, actor_hidden = actor.sample(v_t, actor_hidden)
        q_raw, critic_hidden   = critic(v_t, action, critic_hidden)
        om_np = green_follower.get_action(other_pos)
        om_t  = torch.FloatTensor(om_np).unsqueeze(0).to(DEVICE)
        net({'self_vision': v_t, 'self_motion': action, 'other_motion': om_t},
            q_raw / Q_SCALE, 0.0, 0.0)
        net.superposition_module.detach_state()
    a_np = action.squeeze(0).cpu().numpy()
    new_self  = np.clip(self_pos  + a_np,  -9.5, 9.5)
    new_other = np.clip(other_pos + om_np, -9.5, 9.5)
    env.self_agent.p  = new_self
    env.other_agent.p = new_other
    v_raw, _, _, _, _ = env.step()
    self_pos  = new_self
    other_pos = new_other

# Collect N_SAMPLES visualization frames
samples = []  # list of (v_a1, dec_f1, dec_f2, v_a2, pos_a1, pos_a2)

SAMPLE_INTERVAL = 6   # steps between samples
total_steps = N_SAMPLES * SAMPLE_INTERVAL

for step in range(total_steps):
    v_t = torch.FloatTensor(v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        action, _, actor_hidden = actor.sample(v_t, actor_hidden)
        q_raw, critic_hidden   = critic(v_t, action, critic_hidden)
        om_np = green_follower.get_action(other_pos)
        om_t  = torch.FloatTensor(om_np).unsqueeze(0).to(DEVICE)

        _, h1, h2 = net(
            {'self_vision': v_t, 'self_motion': action, 'other_motion': om_t},
            q_raw / Q_SCALE, 0.0, 0.0
        )
        net.superposition_module.detach_state()

        # Decode both visual features with the trained decoder
        f1_v = net.self_vision_encoder_module(v_t)    # Process-1 feature
        f2_v = net.other_vision_encoder_module(v_t)   # Process-2 feature
        dec_f1 = decoder(f1_v)   # should ≈ A-1's current view
        dec_f2 = decoder(f2_v)   # should ≈ A-2's current view (perspective-taking)

        # A-2's actual view from simulation
        v_a2_raw = env.capture_at_pos(other_pos, self_pos)   # A-2's position, sees A-1
        v_a2 = torch.FloatTensor(
            v_a2_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

    if step % SAMPLE_INTERVAL == 0:
        samples.append({
            'v_a1':    v_t.squeeze(0).cpu(),
            'dec_f1':  dec_f1.squeeze(0).cpu(),
            'dec_f2':  dec_f2.squeeze(0).cpu(),
            'v_a2':    v_a2.squeeze(0).cpu(),
            'pos_a1':  self_pos.copy(),
            'pos_a2':  other_pos.copy(),
        })

    a_np = action.squeeze(0).cpu().numpy()
    new_self  = np.clip(self_pos  + a_np,  -9.5, 9.5)
    new_other = np.clip(other_pos + om_np, -9.5, 9.5)
    env.self_agent.p  = new_self
    env.other_agent.p = new_other
    v_raw, _, _, _, _ = env.step()
    self_pos  = new_self
    other_pos = new_other

# =========================================================================
# Step 4: Visualize (4 rows × N_SAMPLES columns)
# =========================================================================
def to_img(t):
    """(3,16,64) tensor → (16,64,3) numpy uint8."""
    return (t.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)

fig, axes = plt.subplots(4, N_SAMPLES, figsize=(N_SAMPLES * 3.2, 4 * 2.2))
fig.suptitle(
    'Perspective-taking check  (Noguchi et al. 2022, Fig. 5 analogy)\n'
    'Row 1: A-1\'s actual view  |  Row 2: Decoded from f¹_v (sanity)\n'
    'Row 3: Decoded from f²_v (Process-2\'s view)  |  Row 4: A-2\'s actual view',
    fontsize=11
)

row_labels = [
    "A-1's actual view\n(network input)",
    "Decoded f¹_v\n(Process-1, sanity check)",
    "Decoded f²_v\n(Process-2, should ≈ Row 4)",
    "A-2's actual view\n(ground truth)",
]

for row, label in enumerate(row_labels):
    axes[row, 0].set_ylabel(label, fontsize=8, rotation=90,
                             va='center', labelpad=4)

for col, s in enumerate(samples):
    imgs = [s['v_a1'], s['dec_f1'], s['dec_f2'], s['v_a2']]
    p1, p2 = s['pos_a1'], s['pos_a2']

    for row, img_t in enumerate(imgs):
        ax = axes[row, col]
        ax.imshow(to_img(img_t))
        ax.axis('off')
        if row == 0:
            ax.set_title(
                f"A-1=({p1[0]:.1f},{p1[1]:.1f})\n"
                f"A-2=({p2[0]:.1f},{p2[1]:.1f})",
                fontsize=7
            )

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\nSaved: {SAVE_PATH}")

# =========================================================================
# Step 5: Pixel-level error summary
# =========================================================================
print("\n=== Pixel error summary ===")
print("  (MAE between decoded and reference images, lower is better)")

err_f1_vs_a1 = []   # f1_v decoded vs A-1's view (should be small)
err_f2_vs_a2 = []   # f2_v decoded vs A-2's view (key result)
err_f1_vs_a2 = []   # f1_v decoded vs A-2's view (baseline)
err_f2_vs_a1 = []   # f2_v decoded vs A-1's view (baseline)

for s in samples:
    def mae(a, b):
        return float((a.float() - b.float()).abs().mean())

    err_f1_vs_a1.append(mae(s['dec_f1'], s['v_a1']))
    err_f2_vs_a2.append(mae(s['dec_f2'], s['v_a2']))
    err_f1_vs_a2.append(mae(s['dec_f1'], s['v_a2']))
    err_f2_vs_a1.append(mae(s['dec_f2'], s['v_a1']))

print(f"  dec(f1_v) vs A-1 view  [self]:    {np.mean(err_f1_vs_a1):.4f}  ← should be small")
print(f"  dec(f2_v) vs A-2 view  [key]:     {np.mean(err_f2_vs_a2):.4f}  ← key result")
print(f"  dec(f1_v) vs A-2 view  [baseline]:{np.mean(err_f1_vs_a2):.4f}")
print(f"  dec(f2_v) vs A-1 view  [baseline]:{np.mean(err_f2_vs_a1):.4f}")
print()
print("  Perspective-taking holds if: dec(f2_v) vs A-2 < dec(f1_v) vs A-2")
if np.mean(err_f2_vs_a2) < np.mean(err_f1_vs_a2):
    print("  => PASS: Process-2 decodes closer to A-2's view than Process-1 does")
else:
    print("  => FAIL: no clear perspective-taking signal")
