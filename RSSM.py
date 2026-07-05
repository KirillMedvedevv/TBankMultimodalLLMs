"""
RSSM — the conductor. Holds the five world-model components and wires them:

  h_t = rnn(h, s, a)                  (deterministic memory)
  prior  = transition.prior(h)        (blind guess of s)
  post   = encoder.posterior(emb, h)  (s with the frame)
  recon  = decoder(h, s)              (reconstruct the frame)
  reward = reward(h, s)               (predict MiniGrid reward)

forward() runs training rollouts. observe_step()/imagine_step() are the two
primitives the planner needs later.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

from NN import Encoder, Transition, RNN, Decoder, RewardModel

torch.set_num_threads(6)


class RSSM(nn.Module):
    def __init__(self, stoch=30, deter=200, action_dim=7, embed=1024, hidden=200):
        super().__init__()
        self.stoch, self.deter, self.action_dim = stoch, deter, action_dim

        self.encoder = Encoder(stoch_dim=stoch, deter_dim=deter, embed_dim=embed, hidden=hidden)

        self.transition = Transition(stoch_dim=stoch, deter_dim=deter, hidden=hidden)

        self.rnn = RNN(stoch_dim=stoch, action_dim=action_dim, deter_dim=deter, embed_dim=hidden)

        self.decoder = Decoder(feat_dim=deter + stoch)
        
        self.reward = RewardModel(stoch_dim=stoch, deter_dim=deter, hidden=hidden)

    def initial(self, batch_size, device):
        h = torch.zeros(batch_size, self.deter, device=device)
        s = torch.zeros(batch_size, self.stoch, device=device)
        return h, s

    # ---- training rollout ------------------------------------------------
    def forward(self, images, actions):
        """images (T,B,3,64,64), actions (T,B,action_dim) one-hot."""
        T, B = images.shape[:2]
        dev = images.device

        embeds = self.encoder.encode(
            images.reshape(T * B, *images.shape[2:])).reshape(T, B, -1)

        h, s = self.initial(B, dev)
        prev_a = torch.zeros(B, self.action_dim, device=dev)

        pri_m, pri_s, post_m, post_s, hs, ss = [], [], [], [], [], []
        for t in range(T):
            h = self.rnn(h, s, prev_a)
            prior = self.transition.prior(h)
            post = self.encoder.posterior(embeds[t], h)
            s = post.rsample()

            pri_m.append(prior.mean);  pri_s.append(prior.stddev)
            post_m.append(post.mean);  post_s.append(post.stddev)
            hs.append(h); ss.append(s)
            prev_a = actions[t]

        hs, ss = torch.stack(hs), torch.stack(ss)
        feats = torch.cat([hs, ss], dim=-1)                  # (T,B,deter+stoch)

        prior = Normal(torch.stack(pri_m), torch.stack(pri_s))
        post = Normal(torch.stack(post_m), torch.stack(post_s))
        recon = self.decoder(feats.reshape(T * B, -1)).reshape(T, B, 3, 64, 64)
        reward = self.reward(hs, ss)                         # (T,B)

        return dict(recon=recon, reward=reward, prior=prior, post=post)

    def loss_old(self, images, actions, rewards, free_nats=1.0):
        out = self(images, actions)
        recon_loss = ((out["recon"] - images) ** 2).sum(dim=[-1, -2, -3]).mean()
        reward_loss = ((out["reward"] - rewards) ** 2).mean()
        kl = kl_divergence(out["post"], out["prior"]).sum(dim=-1)
        kl = torch.clamp(kl, min=free_nats).mean()
        total = recon_loss + reward_loss + kl
        return total, dict(recon=recon_loss.item(), reward=reward_loss.item(),
                           kl=kl.item())

    def loss(self, images, actions, rewards, free_nats=3.0, kl_balance=0.8, fg_weight=10.0):
        out = self(images, actions)

        # --- реконструкция, движущиеся пиксели (агент) с большим весом ---
        err = (out["recon"] - images) ** 2
        if fg_weight > 0:
            with torch.no_grad():  # под no_grad — ТОЛЬКО вес
                bg = images.median(dim=0, keepdim=True).values
                w = 1.0 + fg_weight * ((images - bg).abs().sum(2, keepdim=True) > 0.1).float()
            err = w * err  # умножение — СНАРУЖИ, градиент течёт
        recon_loss = err.sum(dim=[-1, -2, -3]).mean()

        reward_loss = ((out["reward"] - rewards) ** 2).mean()

        # --- KL-балансировка: учим prior под posterior (нужно воображению) ---
        post, prior = out["post"], out["prior"]
        post_sg = Normal(post.mean.detach(), post.stddev.detach())
        prior_sg = Normal(prior.mean.detach(), prior.stddev.detach())
        kl_prior = kl_divergence(post_sg, prior).sum(-1).mean()
        kl_post = kl_divergence(post, prior_sg).sum(-1).clamp(min=free_nats).mean()
        kl = kl_balance * kl_prior + (1.0 - kl_balance) * kl_post

        total = recon_loss + reward_loss + kl
        return total, dict(recon=recon_loss.item(), reward=reward_loss.item(), kl=kl_prior.item())

    # ---- primitives for planning ----------------------------------------
    def observe_step(self, h, s, prev_a, image):
        """One filtering step with a REAL frame (uses posterior)."""
        h = self.rnn(h, s, prev_a)
        embed = self.encoder.encode(image)
        s = self.encoder.posterior(embed, h).rsample()
        return h, s

    def imagine_step(self, h, s, a):
        """One imagination step with NO frame (uses prior). For planning."""
        h = self.rnn(h, s, a)
        s = self.transition.prior(h).rsample()
        return h, s


if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    from minigrid.wrappers import RGBImgObsWrapper
    from env_tasks import MultiObjEnv

    ACTION_DIM, T = 7, 15

    def prep(frame):
        img = Image.fromarray(frame).resize((64, 64))
        return torch.from_numpy(np.array(img)).float().div(255).sub(0.5).permute(2, 0, 1)

    def onehot(a):
        v = torch.zeros(ACTION_DIM); v[a] = 1.0; return v

    env = RGBImgObsWrapper(MultiObjEnv(render_mode="rgb_array", highlight=False))
    obs, _ = env.reset(seed=0)
    frames, acts, rews = [], [], []
    for t in range(T):
        frames.append(prep(obs["image"]))
        a = env.action_space.sample()
        obs, r, term, trunc, _ = env.step(a)
        acts.append(onehot(a)); rews.append(float(r))
        if term or trunc:
            obs, _ = env.reset(seed=0)

    images = torch.stack(frames).unsqueeze(1)
    actions = torch.stack(acts).unsqueeze(1)
    rewards = torch.tensor(rews).unsqueeze(1)

    model = RSSM(action_dim=ACTION_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("overfitting one batch (loss should drop):")
    for step in range(81):
        loss, parts = model.loss(images, actions, rewards)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20 == 0:
            print(f"  step {step:3d} | total {loss.item():8.1f} | "
                  f"recon {parts['recon']:8.1f} | kl {parts['kl']:5.2f}")

    h, s = model.initial(1, images.device)
    h, s = model.observe_step(h, s, actions[0], images[0])
    h, s = model.imagine_step(h, s, actions[0])
    print("\nobserve_step / imagine_step OK -> h", tuple(h.shape), "s", tuple(s.shape))
