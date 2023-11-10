import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from hw_ss.base import BaseTrainer
from hw_ss.base.base_text_encoder import BaseTextEncoder
from hw_ss.logger.utils import plot_spectrogram_to_buf
from hw_ss.metric.pesq import PESQMetric
import pyloudnorm as pyln
from hw_ss.metric.si_sdr_metric import SISDRMetric
from hw_ss.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            lr_scheduler_name=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.skip_oom = skip_oom
        self.lr_scheduler_name = lr_scheduler_name
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = 100

        self.train_metrics = MetricTracker(
            "loss", "grad norm", *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", *[m.name for m in self.metrics], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["ref_audios", "mix_audios",
                                "target_audios", "ref_audios_length",
                                "mix_audios_length", "target_audios_length",
                                "target_ids"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)

        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                if self.lr_scheduler_name!="ReduceLROnPlateau":
                    lr_epoch = self.lr_scheduler.get_last_lr()[0]
                else:
                    lr_epoch = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar(
                   "learning rate", lr_epoch
                )
                self._log_predictions(**batch)
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics
        
        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            if  (self.lr_scheduler is not None) and (self.lr_scheduler_name=="ReduceLROnPlateau"):
                if part == "val":
                    self.lr_scheduler.step(val_log["loss"])
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(**batch)
        if type(outputs) is dict:
            batch.update(outputs)
        else:
            batch["logits"] = outputs

        batch["loss"] = self.criterion(**batch)
        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                if self.lr_scheduler_name!="ReduceLROnPlateau":
                    self.lr_scheduler.step()

        metrics.update("loss", batch["loss"].item())
        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_predictions(**batch)

        # add histogram of model parameters to the tensorboard
        #for name, p in self.model.named_parameters():
        #    self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    
    def _normalise_aud(self, aud, sr=16000):
        meter = pyln.Meter(sr)
        aud = aud.squeeze().detach().cpu().numpy()
        loud = meter.integrated_loudness(aud)
        return pyln.normalize.loudness(aud, loud, -20.0)

    def _log_predictions(
            self,
            short_decode,
            ref_audios,
            target_audios,
            mix_audios,
            target_audios_length,
            audio_names,
            examples_to_log=5,
            *args,
            **kwargs,
    ):
        # TODO: implement logging of beam search results
        if self.writer is None:
            return
        tuples = list(zip(short_decode, ref_audios, target_audios, target_audios_length, audio_names, mix_audios))
        shuffle(tuples)
        rows = {}
        pesq_metric = PESQMetric()
        si_sdr_metric = SISDRMetric()
        for ans, ref, target, length, name, mix in tuples[:examples_to_log]:
            ans = ans.unsqueeze(0)
            ref = ref.unsqueeze(0)
            target = target.unsqueeze(0)
            mix = mix.unsqueeze(0)
            length = torch.tensor([length], dtype=int)
            pesq = pesq_metric(ans, target, length)
            si_sdr = si_sdr_metric(ans, target, length)

            rows[name] = {
                "target_audio" : self.writer.wandb.Audio(target.squeeze().cpu().numpy(), sample_rate=16000),
                "ref_audio" : self.writer.wandb.Audio(ref.squeeze().cpu().numpy(), sample_rate=16000),
                "mix_audio" : self.writer.wandb.Audio(mix.squeeze().cpu().numpy(), sample_rate=16000),
                "pesq": pesq,
                "si_sdr": si_sdr,
                "prediction": self.writer.wandb.Audio(self._normalise_aud(ans), sample_rate=16000)
            }
        self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
