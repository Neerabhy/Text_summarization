def attention(query, key, value, mask = None, dropout = None):
    d_key = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_key)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model%num_heads ==0, #"Dimension of model should be divisible by number of heads"
        self.dropout = nn.Dropout(p = dropout)
        self.num_heads = num_heads
        self.d_key = d_model//num_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
    
    def forward(self, query, key, value, mask =None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            lin(x).view(nbatches, -1, self.num_heads, self.d_key).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))]
      
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout)
​
        x = (x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_key))
        del query
        del key
        del value
        return self.linears[-1](x)
