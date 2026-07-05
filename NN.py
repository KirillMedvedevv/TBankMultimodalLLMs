import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(6)

class Encoder(nn.Module):
    def __init__(self, stoch_dim = 30, deter_dim=200, hidden = 200, embed_dim=1024) :
        super(Encoder, self).__init__()

        self.stoch_dim = stoch_dim
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.deter_dim = deter_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2), nn.ReLU(),
        )

        self.post_net = nn.Sequential(
            nn.Linear(self.deter_dim + self.embed_dim, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, 2 * self.stoch_dim),
        )

    def encode(self, image):
        # Изображение (B,3,64,64) -> вектор embedding (B, 1024)
        x = self.cnn(image)
        return x.reshape(x.size(0), -1)

    def posterior(self, embed, h):
        # Слияния изображения и состояние h - получаем нормально
        """embedding + memory h -> Normal distribution over s."""
        params = self.post_net(torch.cat([h, embed], dim=-1))
        mean, std = params.chunk(2, dim=-1)
        std = F.softplus(std) + 0.1            # positive, no collapse
        return torch.distributions.Normal(mean, std)

    def infer(self, image, h):
        """Convenience: image + h -> distribution over s (encode then posterior).
        Handy at planning time when you have a single real frame."""
        return self.posterior(self.encode(image), h)

    # forward = просто изображение в вектор
    def forward(self, image):
        return self.encode(image)


class Decoder(nn.Module):
    def __init__(self, feat_dim = 230):
        super(Decoder, self).__init__()

        self.feat_dim = feat_dim

        self.fc = nn.Linear(self.feat_dim, 1024)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, feat_dim) = concat(h, s)
        x = self.fc(feat)  # (B, 1024)
        x = x.reshape(x.shape[0], 1024, 1, 1)
        return self.net(x)


class Transition(nn.Module):
    # St prior пытается угадать
    def __init__(self, stoch_dim, deter_dim, hidden = 200):
        super(Transition, self).__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.hidden = hidden

        self.prior_net = nn.Sequential(
            nn.Linear(self.deter_dim, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, 2*self.stoch_dim),
        )

    def prior(self, h):
        """memory h -> Normal distribution over s (no image)."""
        params = self.prior_net(h)
        mean,std = params.chunk(2, dim=-1)
        std = F.softplus(std) + 0.1
        return torch.distributions.Normal(mean, std)

    def forward(self, h):
        return self.prior(h)


class RNN(nn.Module):
    def __init__(self, stoch_dim=30, action_dim=7, deter_dim=200, embed_dim=200):
        super().__init__()
        self.deter_dim = deter_dim
        # проекция [s, a] в нелинейный признак — вход для GRU
        self.proj = nn.Sequential(
            nn.Linear(stoch_dim + action_dim, embed_dim), nn.ReLU(),
        )
        self.cell = nn.GRUCell(embed_dim, deter_dim)  # (input_size, hidden_size)

    def forward(self, prev_h, prev_s, prev_a):
        x = self.proj(torch.cat([prev_s, prev_a], dim=-1))  # (B, embed_dim)
        return self.cell(x, prev_h)                          # (B, deter_dim) = h_t


class RewardModel(nn.Module):
    def __init__(self, stoch_dim=30, deter_dim=200, hidden=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h, s):
        x = torch.cat([h, s], dim=-1)
        return self.net(x).squeeze(-1)  # (B,) scalar reward