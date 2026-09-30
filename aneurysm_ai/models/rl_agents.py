import torch
import torch.nn as nn


class StateEmbedding(nn.Module):

    def __init__(
        self,
        state_dim=5,
        hidden_dim=128,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, state):
        return self.network(state)


class ActorCritic(nn.Module):

    def __init__(
        self,
        state_dim=5,
        action_dim=9,
        hidden_dim=128,
    ):
        super().__init__()

        self.embedding = StateEmbedding(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        )

        self.actor = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),

            nn.Linear(
                hidden_dim,
                action_dim,
            ),
        )

        self.critic = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),

            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(self, state):

        features = self.embedding(
            state
        )

        logits = self.actor(
            features
        )

        value = self.critic(
            features
        ).squeeze(-1)

        return logits, value

    def distribution(self, state):

        logits, value = self.forward(
            state
        )

        distribution = torch.distributions.Categorical(
            logits=logits
        )

        return distribution, value