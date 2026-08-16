import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianEncoding(nn.Module):
    def __init__(self, sigma: float, input_size: int = 2, encoded_size: int = 256):
        super().__init__()
        self.sigma = sigma
        self.input_size = input_size
        self.encoded_size = encoded_size
        self.register_buffer(
            "W", torch.randn(input_size, encoded_size) / sigma
        )

    def forward(self, x):
        x_proj = 2 * torch.pi * x @ self.W
        
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class GeoCLIPLocationEncoder(nn.Module):
    def __init__(self, d_embed: int, sigma_list: list =[1.0, 8.0, 16.0]):
        super(GeoCLIPLocationEncoder, self).__init__()
        self.d_embed = d_embed
        self.encoders = nn.ModuleList([
            nn.Sequential(
                GaussianEncoding(sigma=s, input_size=2, encoded_size=d_embed),
                nn.Linear(d_embed*2, d_embed),
                nn.ReLU(),
                nn.Linear(d_embed, d_embed)
            )
            for s in sigma_list
        ])

    def forward(self, x):
        features = torch.zeros(x.shape[0], x.shape[1], self.d_embed, device=x.device)
        for encoder in self.encoders:
            x_enc = encoder(x)
            features += x_enc
        
        return features

class TheoryLocationEncoder(nn.Module):
    """
    Implementation of https://arxiv.org/pdf/2003.00824#page=10.60
    """
    def __init__(
        self, 
        embedding_dim: int, 
        lambda_min: float, 
        lambda_max: float,
        num_scales: int = 64,
        dropout_rate: float = 0.3
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.g = lambda_max / lambda_min
        self.S = num_scales

        scales = torch.arange(num_scales).reshape(1, num_scales)
        self.register_buffer('scales', scales, persistent=False)

        a1 = torch.tensor([1, 0])
        a2 = torch.tensor([-1/2, torch.sqrt(torch.tensor(3))/2])
        a3 = torch.tensor([-1/2, -torch.sqrt(torch.tensor(3))/2])
        a = torch.stack([a1, a2, a3])  # (3, 2)
        self.register_buffer('a', a, persistent=False)

        self.location_embedding = nn.Sequential(
            nn.Linear(num_scales * 6, num_scales),
            nn.ReLU(),
            nn.Linear(num_scales, embedding_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:  (..., 2)
        Returns:  (..., embedding_dim)
        """
        nominator = (x @ self.a.T).unsqueeze(-1)  # (..., 3, 1)
        denominator = self.lambda_min * \
            torch.pow(self.g, self.scales / (self.S - 1))
        fraction = nominator / denominator  # (..., 3, num_scales)
        fraction = fraction.reshape(
            *fraction.shape[:-2], -1)  # (..., 3*num_scales)
        PE_sj = torch.concat([torch.cos(fraction), torch.sin(
            fraction)], axis=-1)  # (..., 6*num_scales)
        
        return self.location_embedding(PE_sj)  # (..., embedding_dim)

class Poly2Vec(nn.Module):
    """
    Implementation of Poly2Vec for point encoding.
    """
    def __init__(
        self,
        embedding_dim: int, 
        f_min: float,
        f_max: float,
        n_freqs: int,
        hidden_dim: int = 128,
        dropout_rate: float = 0.3,
        device: str = 'cuda'
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device

        # create meshgrid for sampling frequencies
        self.U, self.V, self.fourier_dim = self.create_gmf_meshgrid(n_freqs, f_min, f_max)
        self.U = self.U[None, :, :].to(device)
        self.V = self.V[None, :, :].to(device)
        
        self.location_embedding = nn.Sequential(
            nn.Linear(2 * self.fourier_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout_rate),
        )
        
        self.nn = nn.Sequential(
            nn.Linear(self.fourier_dim, 2*self.fourier_dim),
            nn.ReLU(),
            nn.Linear(self.fourier_dim*2, self.fourier_dim),
            nn.Dropout(dropout_rate)
        )
        
        self.param_phase = nn.Sequential(
            nn.Linear(self.fourier_dim, self.fourier_dim),
            self.nn
        )
        self.param_magnitude = nn.Sequential(
            nn.Linear(self.fourier_dim, self.fourier_dim),
            self.nn
        )
    
    def create_gmf_meshgrid(self, n_freqs, f_min, f_max):
        """Create a geometric meshgrid for frequency sampling."""
        g = (f_max / f_min)**(1/(n_freqs - 1))
        positive_wu = torch.tensor([f_min * g**u for u in range(n_freqs)], dtype=torch.float32)

        if (2 * n_freqs + 1) % 2 == 1:
            Wx = torch.cat((-torch.flip(positive_wu, dims=[0]), torch.tensor([0]), positive_wu))
        else:
            Wx = torch.cat((-torch.flip(positive_wu[:-1], dims=[0]), torch.tensor([0]), positive_wu))

        if n_freqs % 2 == 1:
            Wy = torch.cat((torch.tensor([0]), positive_wu))
        else:
            Wy = positive_wu

        U, V = torch.meshgrid(Wx, Wy, indexing='ij')
        
        return U, V, Wx.shape[0] * Wy.shape[0]

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p: (B, T, 2) - input coordinates, batched sequences
        Returns:
            output: (B, T, embedding_dim)
        """
        B, T, _ = p.shape
        p_x = p[:, :, 0].unsqueeze(-1).unsqueeze(-1) # (B, T, 1, 1)
        p_y = p[:, :, 1].unsqueeze(-1).unsqueeze(-1)

        # Compute complex Fourier encoding
        loc_enc = torch.exp(-2j * torch.pi * (self.U * p_x + self.V * p_y)).reshape(B, T, -1)

        # Apply magnitude and phase encoders
        mag = self.param_magnitude(torch.abs(loc_enc))
        phase = self.param_phase(torch.angle(loc_enc))

        # Concatenate and flatten
        loc_enc = torch.cat((mag, phase), dim=-1)
        loc_enc = loc_enc.reshape(B, T, -1)

        return self.location_embedding(loc_enc) # (B, T, embedding_dim)

def get_location_encoder(loc_encoder_type: str, dim_embed: int, args: str) -> nn.Module:
    """
    Factory function to get the appropriate location encoder.
    
    Args:
        loc_encoder_type (str): Type of location encoder to use.
        dim_embed (int): Dimension of the location embedding.
        args (dict): Additional arguments for the encoder.
        
    Returns:
        nn.Module: An instance of the specified location encoder.
    """
    if loc_encoder_type == "geoclip":
        return GeoCLIPLocationEncoder(d_embed=dim_embed, 
                                      sigma_list=args.gaussian_sigmas)
    
    elif loc_encoder_type == "theory":
        return TheoryLocationEncoder(embedding_dim=dim_embed, 
                                     lambda_min=args.lambda_min, 
                                     lambda_max=args.lambda_max,
                                     dropout_rate=args.dropout,
                                     num_scales=args.num_scales)
    
    elif loc_encoder_type == "poly2vec":
        return Poly2Vec(embedding_dim=dim_embed, 
                        f_min=args.f_min, 
                        f_max=args.f_max, 
                        n_freqs=args.n_freqs,
                        dropout_rate=args.dropout,
                        device=args.device)
    else:
        raise ValueError(f"Unknown location encoder type: {loc_encoder_type}")