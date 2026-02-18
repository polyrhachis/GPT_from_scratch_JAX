import jax
from jax import numpy as jnp, random as jr
import equinox as eqx


class Linear(eqx.Module):

    w : jnp.ndarray
    b : jnp.ndarray

    def __init__(self, key, dim_in, dim_out):
        key = jr.PRNGKey(key)
        self.w = jr.normal(key,(dim_in, dim_out))/40
        self.b = jnp.zeros((dim_out,))

    def __call__(self, x):
        return x@self.w + self.b
    


class FFN(eqx.Module):
    linear1 : Linear
    linear2 : Linear
    activation : object = eqx.field(static=True)

    def __init__(self, key, dim_in, dim):
        self.linear1 = Linear(0, dim_in, dim)
        self.linear2 = Linear(1, dim, dim_in)
        self.activation = jax.nn.relu

    def __call__(self, x):
        return self.linear2(self.activation(self.linear1(x)))




class SelfAttention(eqx.Module):
    Qw : jnp.ndarray
    Kw : jnp.ndarray
    Vw : jnp.ndarray

    def __init__(self, key, dim):
        key = jr.PRNGKey(key)
        key, key2, key3 = jr.split(key, 3)
        self.Qw = jr.normal(key, (dim, dim))/40
        self.Kw = jr.normal(key2, (dim, dim))/40
        self.Vw = jr.normal(key3, (dim, dim))/40
    
    def __call__(self, x):
        Q, K, V = self.get_QKV(x)
        d = x.shape[-1]
        mask = jnp.tril(jnp.ones((x.shape[-2], x.shape[-2])))
        mask = jnp.where(mask == 0, -jnp.inf, 0.0)
        out = (Q@jnp.swapaxes(K, -1, -2))/jnp.sqrt(d)
        out += mask
        out = jax.nn.softmax(out, axis=-1)

        return out@V
    
    def get_QKV(self, x):
        return x@self.Qw, x@self.Kw, x@self.Vw


class MultiHeadAttention(eqx.Module):
    Qw : jnp.ndarray
    Kw : jnp.ndarray
    Vw : jnp.ndarray
    Wo : jnp.ndarray  
    heads : int = eqx.field(static=True)

    def __init__(self, key : int, heads, dim):  

        key = jr.PRNGKey(key)
        self.heads = heads
        key, key2, key3, key4 = jr.split(key, 4)
        self.Qw = jr.normal(key, (dim, dim))/40
        self.Kw = jr.normal(key2, (dim, dim))/40
        self.Vw = jr.normal(key3, (dim, dim))/40
        self.Wo = jr.normal(key4, (dim, dim))/40

    def __call__(self, x):
        Q, K, V = self.get_QKV(x)
        Q, K, V = self.split_and_reshape(Q, K, V, x)
        logits = Q@jnp.swapaxes(K, axis1=-1, axis2=-2)/jnp.sqrt(x.shape[-1]/self.heads)


        seq_len = x.shape[1]
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        mask = jnp.where(mask == 0, -jnp.inf, 0.0)
        logits = logits + mask
        scores = jax.nn.softmax(logits, axis=-1)@V
        scores = scores.swapaxes(axis1=-2, axis2=-3)
        scores = scores.reshape(-1, x.shape[1], x.shape[-1])
        



        return scores@ self.Wo
    
    def get_QKV(self, x):
        return x@self.Qw, x@self.Kw, x@self.Vw
    
    def split_and_reshape(self, Q, K, V, x):
        Q = jnp.swapaxes(Q.reshape(x.shape[0], x.shape[1], self.heads, -1), axis1=-2, axis2=-3)
        K = jnp.swapaxes(K.reshape(x.shape[0], x.shape[1], self.heads, -1), axis1=-2, axis2=-3)
        V = jnp.swapaxes(V.reshape(x.shape[0], x.shape[1], self.heads, -1), axis1=-2, axis2=-3)
        return Q, K, V



class Sequential(eqx.Module):
    layers : tuple
    
    def __init__(self, *layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class LayerNorm(eqx.Module):
    gamma : jnp.ndarray
    beta : jnp.ndarray

    def __init__(self, dim):
        self.gamma = jnp.ones((dim))
        self.beta = jnp.zeros((dim))

    def __call__(self, x):
        num = x - jnp.mean(x, axis=-1, keepdims=True)
        den = jnp.sqrt(jnp.var(x, axis=-1, keepdims=True) + 1e-5)

        return self.gamma*( num/den) + self.beta
    

class Embedding(eqx.Module):
    emb : jnp.ndarray

    def __init__(self, key, vocab_size, dim):
        key = jr.PRNGKey(key)
        self.emb = jr.normal(key, (vocab_size, dim))/40

    def __call__(self, idxs):
        return self.emb[idxs]
    

class TransformerBlock(eqx.Module):
    attn: SelfAttention
    ffn: FFN
    ln1: LayerNorm
    ln2: LayerNorm

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(eqx.Module):
    sequential : Sequential
    w_emb : Embedding
    pos_emb : Embedding

    def __init__(self, key, sequential, w_emb, pos_emb):
        self.sequential = sequential
        self.w_emb = w_emb
        self.pos_emb = pos_emb

    def __call__(self, x):
        seq_len = x.shape[-1]
        pos = jnp.arange(seq_len)
        
        x = self.w_emb(x) + self.pos_emb(pos)  
        return self.sequential(x)
          
    




with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
stoi = {ch : i for i, ch in enumerate(chars)}
itos = {i : ch for i, ch in enumerate(chars)}

encode = lambda string: [stoi[i] for i in string]
decode = lambda num: ''.join([itos[i] for i in num])

x = jnp.array(encode(text))
dataset_len = len(x)
block_size = 32
batch_size = 1280

class DataLoader:
    def __init__(self, data, batch_size):
        self.data = jnp.array(data, dtype=jnp.int32)
        self.batch_size = batch_size
        self.dataset_len = len(data)
    def __call__(self, key):
        key, subkey = jr.split(key)
        ix = jr.randint(
            subkey,
            minval=0,
            maxval=self.dataset_len - block_size,
            shape=(self.batch_size,)
        )

        offsets = jnp.arange(block_size)

        x_batch = self.data[ix[:, None] + offsets[None, :]]
        y_batch = self.data[ix[:, None] + offsets[None, :] + 1]

        yield x_batch, y_batch, key




class RMSprop(eqx.Module):
    memory : jnp.ndarray
    gamma : float
    lr : float
    def __init__(self, model, gamma, lr):
        self.memory = jax.tree.map(jnp.zeros_like, model)
        self.gamma = gamma
        self.lr = lr

    def __call__(self, model, grad):
        model = jax.tree.map(lambda model, memory, grad:
                            model - ((self.lr/(jnp.sqrt(memory) + 1e-6))* grad) ,
                             model, self.memory, grad) 
        return model

    def update(self, grad):
        memory = jax.tree.map(lambda memory, grad:
                                   self.gamma*(memory) + (1 - self.gamma) * grad**2,
                                   self.memory, grad)
        return eqx.tree_at(lambda opt: opt.memory, self, memory)


def loss(preds, y):
    num_classes = preds.shape[-1]
    y_one_hot = jax.nn.one_hot(y, num_classes)
    
    
    log_p = jax.nn.log_softmax(preds, axis=-1)
    
    return -jnp.mean(jnp.sum(y_one_hot * log_p, axis=-1))


model = GPT(
    key=0,
    sequential=Sequential(
        TransformerBlock(MultiHeadAttention(0, 8, 128), FFN(0, 128, 512), LayerNorm(128), LayerNorm(128)),
        TransformerBlock(MultiHeadAttention(1, 8, 128), FFN(1, 128, 512), LayerNorm(128), LayerNorm(128)),
        TransformerBlock(MultiHeadAttention(2, 8, 128), FFN(2, 128, 512), LayerNorm(128), LayerNorm(128)),
        TransformerBlock(MultiHeadAttention(3, 8, 128), FFN(3, 128, 512), LayerNorm(128), LayerNorm(128)),
        TransformerBlock(MultiHeadAttention(4, 8, 128), FFN(4, 128, 512), LayerNorm(128), LayerNorm(128)),
        TransformerBlock(MultiHeadAttention(5, 8, 128), FFN(5, 128, 512), LayerNorm(128), LayerNorm(128)),
        TransformerBlock(MultiHeadAttention(6, 8, 128), FFN(6, 128, 512), LayerNorm(128), LayerNorm(128)),
        Linear(0, 128, 65)
    ), w_emb=Embedding(0, vocab_size=65, dim=128), pos_emb=Embedding(1, vocab_size=block_size, dim=128)
)

loader = DataLoader(data=x, batch_size=batch_size)




def train_loop(steps, model, loss):
    optim = RMSprop(model=model, gamma=0.99, lr=0.001)
    
    def get_pred(model, x, y):
        preds = model(x)
        return loss(preds, y)
    
    
    @jax.jit
    def train_step(model, x, y, optim):
        loss, grad = jax.value_and_grad(get_pred)(model, x, y)
        optim = optim.update(grad)
        model = optim(model, grad)
        return loss, model, optim 
    
    
    key = jr.PRNGKey(0)
    for step in range(steps):
        key, subkey = jr.split(key)
        x, y, key = next(loader(subkey))
        loss, model, optim = train_step(model, x, y, optim)
        if step % 10 == 0:
            print(loss)
    
    return model
    

model = train_loop(1000, model, loss)


def generate(model, start_string, max_new_tokens):


    string = jnp.array(encode(start_string)).reshape(1, -1)
    for i in range(max_new_tokens):

        token = model(string)
        string = jnp.concatenate((string, jnp.argmax(token[:, -1, :], axis=-1).reshape(1, -1)), axis=-1)
        print(decode([string[0][-1].tolist()]), end="")





generate(model, "the:", 50)


