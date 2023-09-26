from pytorch_lightning import LightningModule

from models import Encoder, Decoder, WaveDiscriminator, STFTDiscriminator
from vector_quantize_pytorch import ResidualVQ
import torch
import torch.optim as optim
from utils.loss import*

class Soundstream(LightningModule):
    def __init__(self,
                 channel=32,
                 RVQ_dimension=1,
                 num_qunatizers=8,
                 codebook_size=1024,
                 num_downsampling = 3,
                 downsampling_Factor = 2,
                 n_fft = 1024,
                 hop_length = 256,
                 lambda_adv = 1,
                 lambda_feat = 100,
                 lambda_rec = 1,
                 sample_rate=24000,
                 **kwargs):
        super().__init__()

        self.automatic_optimization = False

        self.encoder = Encoder(C=channel,D=RVQ_dimension)
        self.quantizer = ResidualVQ(num_quantizers=num_qunatizers, dim=RVQ_dimension, codebook_size=codebook_size, kmeans_init=True, kmeans_iters=100, threshold_ema_dead_code=2)
        self.decoder = Decoder(C=channel,D=RVQ_dimension)

        self.wav_disc = WaveDiscriminator(num_D=num_downsampling, downsampling_factor=downsampling_Factor)
        self.stft_disc = STFTDiscriminator(C=channel, F_bins=n_fft//2)

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.criterion_g = lambda x, G_x, features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_stft, lengths_wave: lambda_adv*adversarial_g_loss(features_stft_disc_G_x, features_wave_disc_G_x, lengths_stft, lengths_wave) + lambda_feat*feature_loss(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft) + lambda_rec*spectral_reconstruction_loss(x, G_x, sr=sample_rate)
        self.criterion_d = adversarial_d_loss

        self.validation_step_outputs_g = []
        self.validation_step_outputs_d = []

    def forward(self, x):
        encoder_output = self.encoder(x)
        encoder_output = encoder_output.permute(0,2,1)
        quantized, _, _ = self.quantizer(encoder_output)
        quantized = quantized.permute(0,2,1)
        decoder_output = self.decoder(quantized)
        return decoder_output

    def training_step(self, batch, batch_idx):
        loss_g, loss_d = self.common_step(batch)
        opt_g, opt_d = self.optimizers['optimizer']
        sch_g, sch_d = self.optimizers['lr_scheduler']

        opt_g.zero_grad()
        self.manual_backward(loss_g)
        opt_g.step()
        sch_g.step()

        opt_d.zero_grad()
        self.manual_backward(loss_d)
        opt_d.step()
        sch_d.step()

        self.log_dict({"g_loss": loss_g, "d_loss": loss_d}, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        loss_g, loss_d = self.common_step(batch)
        self.validation_step_outputs_g.append(loss_g)
        self.validation_step_outputs_d.append(loss_d)

    def on_validation_epoch_end(self) -> None:
        outputs_g = self.validation_step_outputs_g
        avg_loss_g = sum(outputs_g) / len(outputs_g)
        self.log('val_loss_g', avg_loss_g, prog_bar=True, sync_dist=True)
        outputs_d = self.validation_step_outputs_d
        avg_loss_d = sum(outputs_d) / len(outputs_d)
        self.log('val_loss_d', avg_loss_d, prog_bar=True, sync_dist=True)
        self.validation_step_outputs_g.clear()
        self.validation_step_outputs_d.clear()

    def common_step(self, batch):
        audio, audio_len = batch
        if len(audio.shape)==2:
            audio = audio.unsqueeze(1)
        G_audio = self(audio)
        s_audio = torch.stft(audio.squeeze(), n_fft=self.n_fft, hop_length=self.hop_length,
                         window=torch.hann_window(window_length=self.n_fft, device=audio.device), return_complex=False).permute(0,3,1,2)
        s_audio_len = 1 + torch.div(audio_len, self.hop_length, rounding_mode="floor")
        s_G_audio = torch.stft(G_audio.squeeze(), n_fft=self.n_fft, hop_length=self.hop_length,
                           window=torch.hann_window(window_length=self.n_fft, device=G_audio.device), return_complex=False).permute(0,3,1,2)

        stft_len = self.stft_disc.features_lengths(s_audio_len)
        wave_len = self.wav_disc.features_lengths(audio_len)

        features_stft_disc_audio = self.stft_disc(s_audio)
        features_wave_disc_audio = self.wav_disc(audio)

        features_stft_disc_G_audio = self.stft_disc(s_G_audio)
        features_wave_disc_G_audio = self.wav_disc(G_audio)

        loss_g = self.criterion_g(audio, G_audio, features_stft_disc_audio, features_wave_disc_audio,
                                  features_stft_disc_G_audio,
                                  features_wave_disc_G_audio, stft_len, wave_len)

        # backward 2번 할 때 필수
        features_stft_disc_audio = self.stft_disc(s_audio)
        features_wave_disc_audio = self.wav_disc(audio)

        features_stft_disc_G_audio_det = self.stft_disc(s_G_audio.detach())
        features_wave_disc_G_audio_det = self.wav_disc(G_audio.detach())

        loss_d = self.criterion_d(features_stft_disc_audio, features_wave_disc_audio, features_stft_disc_G_audio_det,
                                  features_wave_disc_G_audio_det,
                                  stft_len, wave_len)

        return loss_g, loss_d

    def configure_optimizers(self):
        optimizer_g = optim.Adam(list(self.encoder.parameters()) + list(self.quantizer.parameters()) + list(self.decoder.parameters()), lr=1e-4, betas=(0.5, 0.9))
        optimizer_d = optim.Adam(list(self.wav_disc.parameters()) + list(self.stft_disc.parameters()), lr=1e-4, betas=(0.5, 0.9))
        scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=500)
        scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=500)
        sch_config_g = {
            "scheduler": scheduler_g,
            "interval": "step",
            "name": "Learning_rate"
            }
        sch_config_d = {
            "scheduler": scheduler_d,
            "interval": "step",
            "name": "Learning_rate"
        }
        self.optimizers = {
            "optimizer": [optimizer_g, optimizer_d],
            "lr_scheduler": [scheduler_g, scheduler_d],
        }


