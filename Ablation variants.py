
class AblatableDHESAdapter(DHESAdapter):
    """Component flags default to True, giving the complete adapter."""
    use_rotation = use_mixing = use_silu_gate = True
    use_temp_gate = use_context_gate = True

    def forward(self, x):
        base_out = self.base_layer(x)
        Ax = self.A(x)

        if self.use_rotation:                                    # (R)
            cos_t, sin_t = torch.cos(self.theta), torch.sin(self.theta)
            x_even, x_odd = Ax[..., 0::2], Ax[..., 1::2]
            rot = torch.stack([cos_t * x_even - sin_t * x_odd,
                               sin_t * x_even + cos_t * x_odd],
                              dim=-1).flatten(-2)
        else:
            rot = Ax

        normed = _rms_norm_fn(rot, self.norm_scale, self._norm_eps)
        combined = normed + normed @ self.Phi if self.use_mixing else normed   # (M)

        gate_seq = F.silu(combined) if self.use_silu_gate else combined        # (G)

        if self.use_context_gate:                                              # (G+T)
            gate_tau = (torch.exp(self.log_gate_tau).clamp(0.05, 5.0)
                        if self.use_temp_gate
                        else torch.ones_like(self.log_gate_tau))               # (T)
            gate_logit = self.W_gate * Ax.mean(dim=1, keepdim=True)
            gated = torch.sigmoid(gate_logit / gate_tau) * gate_seq
        else:
            gated = gate_seq

        return base_out + self.B(gated) * self.lora_scale