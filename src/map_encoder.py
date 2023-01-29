import os
import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .map_files import Transformer, PositionalEncoding
  

class MapEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Connv layers init
        self.get_feature_map = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1), #67 - 3 + 1 = 65
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=3), #(65 - 3)/2 + 1 = 32
            nn.Conv2d(in_channels=16, out_channels=6, kernel_size=1, stride=1), #32 - 1 + 1 = 32
        )
        # SEQUENTIAL: 1x3x67x67 -> 1x6x32x32
        
        # 425 - 5 + 1, (421 - 4)/3 + 1, 140
        
        # Vision Transformer init
        self.VTrans_encoder = ViT(
            image_size = (140, 140), #?
            patch_size = (14, 14), #?
            dim = 32, #?
            depth = 6, #?
            heads = 16, #?
            mlp_dim = 2048, #?
            pool = 'mean', #?
            channels = 6,
            dim_head = 64, #?
            dropout = 0.1, #?
            emb_dropout = 0.1 #?
        )
        
        # def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):

    def forward(self):
        x = self.get_three_channels().cuda().float() # Cast from double (not accepted) to float
        
        # Get the feature map uing conv layers on map's 3 channels
        x = self.get_feature_map(x)
        # Input feature map to VisTransf and get the output features
        x = self.VTrans_encoder(x)
        return x
    
    def get_three_channels(self):
        
        array_dir = 'output/eth/map_arrays'
        dist_path = os.path.join(array_dir, 'dist_array.npy')
        driv_path = os.path.join(array_dir, 'driv_array.npy')
        
        if not os.path.exists(array_dir):
            os.makedirs(array_dir)
            
        if os.path.exists(dist_path) and os.path.exists(driv_path):
            os.makedirs(array_dir, exist_ok=True)
            dist = np.load(dist_path)
            drivable = np.load(driv_path)
        else:
            dist, drivable = self.get_map_from_csv(dist_path, driv_path)
        
        rows, cols = drivable.shape[0], drivable.shape[1]
        indices = torch.arange(0, rows*cols)
        indices = torch.reshape(indices, (1, 1, rows, cols))
        
        dist = torch.reshape(torch.from_numpy(dist), (1, 1, dist.shape[0], dist.shape[1]))
        drivable = torch.reshape(torch.from_numpy(drivable), (1, 1, rows, cols))
        
        out = torch.cat((drivable, dist, indices), 1)
        
        return out
    
    def get_map_from_csv(self, dist_dir, driv_dir):
        data_dir = 'data/map/drivable_map_d1.csv'
        
        idx_x, idx_y, idx_lon, idx_lat, idx_driv = 2, 3, 4, 5, 6
        '''
        with open(data_dir, 'r') as data_obj:
            data_read = csv.reader(data_obj)
            data = list(data_read)
        '''
        data = np.loadtxt(data_dir, delimiter=',', skiprows=1)
        cols = int((data[-1,idx_x]) - (data[0,idx_x] ) + 1)
        rows = int((data[-1,idx_y]) - (data[0,idx_y] ) + 1)
        
        drivable = np.reshape(data[:,idx_driv], (rows, cols))
        
        lon = np.reshape(data[:,idx_lon], (1, rows, cols))
        lat = np.reshape(data[:,idx_lat], (1, rows, cols))
        
        center_idx = np.array([np.ceil(rows/2)-1, np.ceil(cols/2)-1]).astype(int)
        
        lon = (lon - lon[0, center_idx[0], center_idx[1]]) * pow(10, 5)
        lat = (lat - lat[0, center_idx[0], center_idx[1]]) * pow(10, 5)
        
        coord = np.concatenate((lon, lat), 0)
        
        dist = np.linalg.norm(coord, axis=0)
        
        np.save(dist_dir, dist)
        np.save(driv_dir, drivable)
        
        return dist, drivable
    
    
class ContextTransformer(nn.Module):
    def __init__(self, dim_model=32, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, dropout_p=0.1):
        super().__init__()
        
        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # Positional encoding layer init
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout_p=dropout_p, max_len=5000)
        
        # Transformer init
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None):
        # Src size must be (src sequence length, batch_size)
        # Tgt size must be (tgt sequence length, batch_size)
        
        # we permute to obtain size (batch_size, sequence length, dim_model),
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # we permute to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask)

        return out
    

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3,
                 dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = image_size
        patch_height, patch_width = patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )
        
        image_dim = channels * image_height * image_width
        
        self.to_global_embedding = nn.Linear(image_dim, dim)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, img):
        
        # Get patches from feature map
        x = self.to_patch_embedding(img)
        
        # Get global embedding from feature map
        img = torch.flatten(img, start_dim=1)
        g_emb = self.to_global_embedding(img)
        g_emb = torch.unsqueeze(g_emb, dim=1)
        
        b, n, _ = x.shape

        # Concatenate global and local embeddings
        x = torch.cat((g_emb, x), dim=1)
        
        # Add positional embedding and dropout
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Use concat features in the transformer
        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # Get the final output features of the Vision Transformer
        x = self.to_latent(x)
        return x