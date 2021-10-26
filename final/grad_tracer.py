import types
import torch


class TransformerTracer:
    def __init__(self, model):
        self.n = 0
        self.model = model
        self.accumulator = {
            'encoder': torch.zeros(
                len(self.model.encoder.encoders),
                self.model.encoder.encoders[0].self_attn.h
            )
        }

    def dump(self, path):
        torch.save({
            'accumulator': self.accumulator,
            'n': self.n
        }, path)

    def update_accumulator(self):
        with torch.no_grad():
            # encoder
            grad_etas = self.get_grad_etas_enc().cpu().float()
            self.accumulator['encoder'].data += grad_etas.abs().sum(0).data
            bsz = grad_etas.shape[0]

            # accumulate number of samples
            self.n += bsz

    def get_grad_etas_enc(self):
        """
        Return:
            (bsz, n_layer, n_head)
        """
        etas = []
        for layer in self.model.encoder.encoders:
            module = layer.self_attn.linear_out
            attn = module.attn_outputs
            _, seq_len, head_dim = attn.shape
            n_head = layer.self_attn.h
            grad = attn.grad.reshape(-1, n_head, seq_len, head_dim // n_head)
            attn = attn.reshape(-1, n_head, seq_len, head_dim // n_head)
            eta = calc_grad_eta(attn, grad)  # eta.shape == [bsz, n_head]
            etas.append(eta)

        etas = torch.stack(etas, 1)
        return etas

    def preload_attn_func(self):
        for layer in self.model.encoder.encoders:
            module = layer.self_attn.linear_out
            module.orig_forward = module.forward
            module.forward = types.MethodType(
                TransformerTracer.save_and_forward, module
            )

    @staticmethod
    def save_and_forward(self, attn):
        attn.retain_grad()
        self.attn_outputs = attn
        return self.orig_forward(attn)


def calc_grad_eta(attn, grad):
    """
    Args:
        attn: attn.shape == (bsz, n_head, seq_len, dim_head)
        grad: grad.shape == (bsz, n_head, seq_len, dim_head)
    Return:
        eta: FloatTensor. shape == (bsz, n_head)
    """
    bsz, n_head, seq_len, dim_head = attn.shape
    grad = grad.data.reshape(bsz, n_head, -1, 1)
    attn = attn.data.reshape(bsz, n_head, 1, -1)
    grad_eta = (attn @ grad).reshape(bsz, n_head)
    return grad_eta
