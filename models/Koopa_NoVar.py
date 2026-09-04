from models.Koopa import Model as KoopaModel


class Model(KoopaModel):
    """Koopa ablation: disentangle time-variant branch but do not add it to final prediction."""

    def forecast(self, x_enc):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - mean_enc
        std_enc = (x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
        x = x / std_enc

        residual, forecast = x, None
        for i in range(self.num_blocks):
            time_var_input, time_inv_input = self.disentanglement(residual)
            time_inv_output = self.time_inv_kps[i](time_inv_input)
            time_var_backcast, _time_var_output = self.time_var_kps[i](time_var_input)
            residual = residual - time_var_backcast
            forecast = time_inv_output if forecast is None else forecast + time_inv_output

        return forecast * std_enc + mean_enc
