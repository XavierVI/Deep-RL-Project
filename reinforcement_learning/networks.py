import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------- #
#                     Value Network Definitions                   #
# --------------------------------------------------------------- #
class ValueNet(nn.Module):
    def __init__(self, s_dim, a_dim, hidden, act):
        super().__init__()

        act_fn = getattr(F, act)
        self.layers = nn.ModuleList()
        prev = s_dim

        for h in hidden:
            self.layers.append(nn.Linear(prev, h))
            prev = h

        self.out = nn.Linear(prev, a_dim)
        self.act_fn = act_fn

    def forward(self, x):
        for l in self.layers:
            x = self.act_fn(l(x))

        return self.out(x)


# --------------------------------------------------------------- #
#                     Policy Network Definitions                  #
# --------------------------------------------------------------- #
class PolicyNet(nn.Module):
    """Policy network for discrete action spaces."""

    def __init__(self, s_dim, a_dim, hidden, act):
        super().__init__()

        # this dynamically gets the activation function from torch.nn.functional
        act_fn = getattr(F, act)
        print(f"Using activation function: {act}")
        self.layers = nn.ModuleList()
        prev = s_dim

        for h in hidden:
            self.layers.append(nn.Linear(prev, h))
            prev = h

        self.out = nn.Linear(prev, a_dim)
        self.act_fn = act_fn

    def forward(self, x):
        for l in self.layers:
            x = self.act_fn(l(x))

        x = self.out(x)

        # return logits and action probabilities
        return x, F.softmax(x, dim=-1)


class ContinuousPolicyNet(nn.Module):
    """Policy network for continuous action spaces using Gaussian policy."""

    def __init__(self, s_dim, a_dim, hidden, act):
        super().__init__()

        act_fn = getattr(F, act)
        print(f"Using activation function: {act}")
        self.layers = nn.ModuleList()
        prev = s_dim

        for h in hidden:
            self.layers.append(nn.Linear(prev, h))
            prev = h

        # Output mean and log_std for Gaussian policy
        self.mean = nn.Linear(prev, a_dim)
        self.log_std = nn.Linear(prev, a_dim)
        self.act_fn = act_fn

    def forward(self, x):
        for l in self.layers:
            x = self.act_fn(l(x))

        mean = torch.tanh(self.mean(x))  # Bound mean to [-1, 1]
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # Prevent numerical instability

        return mean, log_std


# --------------------------------------------------------------- #
#              Small Transformer for Enemy Localization            #
# --------------------------------------------------------------- #
class TransformerEnemyLocalizerHead(nn.Module):
    """
    Lightweight transformer-based module for classifying enemy positions
    and generating action-relevant features from spatial enemy data.
    
    Uses multi-head attention to identify which enemies are most salient
    for action selection in Doom.
    """
    
    def __init__(self, 
                 feature_dim=64,
                 num_heads=4,
                 num_layers=2,
                 max_enemies=10,
                 dropout=0.1):
        """
        Args:
            feature_dim: Dimension of features per enemy (d_model for transformer)
            num_heads: Number of attention heads
            max_enemies: Maximum number of enemies to track simultaneously
            num_layers: Number of transformer encoder layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.max_enemies = max_enemies
        
        # Learnable positional embeddings for enemy positions
        self.pos_embedding = nn.Parameter(torch.randn(1, max_enemies, feature_dim))
        
        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(feature_dim)
        )
        
        # Classification heads
        self.threat_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 3)  # Classify threat level: [not_threat, medium, high]
        )
        
        self.position_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 4)  # Classify position: [left, center, right, overhead]
        )
        
    def forward(self, enemy_features):
        """
        Args:
            enemy_features: Tensor of shape (batch_size, num_enemies, feature_dim)
                          where enemy_features[i, j] contains features for j-th enemy
                          
        Returns:
            threat_logits: (batch_size, num_enemies, 3) - threat level per enemy
            position_logits: (batch_size, num_enemies, 4) - position class per enemy
            attention_weights: (batch_size, num_enemies) - learned importance weights
        """
        batch_size = enemy_features.shape[0]
        
        # Ensure input has correct dimensions
        if len(enemy_features.shape) == 2:
            # If (batch_size, feature_dim), treat as single enemy
            enemy_features = enemy_features.unsqueeze(1)
        
        # Add positional embeddings
        x = enemy_features + self.pos_embedding[:, :enemy_features.shape[1], :]
        
        # Apply transformer encoder
        transformer_out = self.transformer_encoder(x)
        
        # Classification heads
        threat_logits = self.threat_classifier(transformer_out)
        position_logits = self.position_classifier(transformer_out)
        
        # Compute attention-based importance weights (mean across heads)
        attention_weights = torch.mean(torch.abs(transformer_out), dim=-1)
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        return threat_logits, position_logits, attention_weights


class TransformerPolicyNet(nn.Module):
    """
    Policy network that integrates enemy localization via transformer.
    Processes enemy spatial information to inform action selection.
    """
    
    def __init__(self, 
                 state_dim,
                 action_dim,
                 enemy_feature_dim=64,
                 num_transformer_heads=4,
                 num_transformer_layers=2,
                 hidden_layers=[128, 64],
                 act='relu',
                 max_enemies=10):
        """
        Args:
            state_dim: Total state dimension from environment
            action_dim: Number of actions
            enemy_feature_dim: Dimension for enemy features in transformer
            num_transformer_heads: Number of attention heads
            num_transformer_layers: Number of transformer layers
            hidden_layers: Hidden layer sizes for MLP after transformer
            act: Activation function name
            max_enemies: Max enemies to track
        """
        super().__init__()
        
        act_fn = getattr(F, act)
        self.act_fn = act_fn
        
        # Assume state can be split: [main_features, enemy_data]
        # You'll need to split this appropriately in forward pass
        self.state_dim = state_dim
        
        # Extract enemy features from state
        self.enemy_embedding = nn.Linear(state_dim, max_enemies * enemy_feature_dim)
        
        # Transformer localization head
        self.enemy_localizer = TransformerEnemyLocalizerHead(
            feature_dim=enemy_feature_dim,
            num_heads=num_transformer_heads,
            num_layers=num_transformer_layers,
            max_enemies=max_enemies,
            dropout=0.1
        )
        
        # MLP after transformer to produce action logits
        # Input: concatenated transformer output + threat/position info
        transformer_out_dim = enemy_feature_dim * max_enemies + 7 * max_enemies  # 3 threat + 4 position
        
        self.layers = nn.ModuleList()
        prev_dim = transformer_out_dim
        
        for h in hidden_layers:
            self.layers.append(nn.Linear(prev_dim, h))
            prev_dim = h
        
        self.out = nn.Linear(prev_dim, action_dim)
    
    def forward(self, state):
        """
        Args:
            state: (batch_size, state_dim) or (state_dim,)
            
        Returns:
            action_logits: Logits for each action
            action_probs: Probability distribution over actions
        """
        # Handle batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = state.shape[0]
        
        # Extract and reshape enemy features for transformer
        enemy_features_flat = self.enemy_embedding(state)
        enemy_features = enemy_features_flat.view(
            batch_size, 
            self.enemy_localizer.max_enemies, 
            self.enemy_localizer.feature_dim
        )
        
        # Apply transformer enemy localization
        threat_logits, position_logits, attention_weights = self.enemy_localizer(enemy_features)
        
        # Flatten and concatenate all transformer outputs
        threat_probs = F.softmax(threat_logits, dim=-1)  # (batch, max_enemies, 3)
        pos_probs = F.softmax(position_logits, dim=-1)   # (batch, max_enemies, 4)
        
        x = torch.cat([
            enemy_features,
            threat_probs,
            pos_probs,
            attention_weights.unsqueeze(-1)
        ], dim=-1)  # (batch, max_enemies, feature_dim + 3 + 4 + 1)
        
        x = x.view(batch_size, -1)  # Flatten: (batch, max_enemies * (feature_dim + 8))
        
        # MLP to produce actions
        for layer in self.layers:
            x = self.act_fn(layer(x))
        
        logits = self.out(x)
        
        if squeeze_output:
            logits = logits.squeeze(0)
        
        # Return logits and action probabilities
        return logits, F.softmax(logits, dim=-1)



