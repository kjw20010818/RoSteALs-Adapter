import os
import sys
from omegaconf import OmegaConf
import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
try:
    from pytorch_lightning.utilities.distributed import rank_zero_only
except ImportError:
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
import time 


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def __init__(self):
        super().__init__()
        self.start_time = time.time()

    def _get_gpu(self, trainer):
        """Compatible way to get GPU device across PL versions."""
        if hasattr(trainer, 'root_gpu'):
            return trainer.root_gpu
        if torch.cuda.is_available():
            return torch.cuda.current_device()
        return None

    def on_train_epoch_start(self, trainer, pl_module):
        gpu = self._get_gpu(trainer)
        if gpu is not None:
            torch.cuda.reset_peak_memory_stats(gpu)
            torch.cuda.synchronize(gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        gpu = self._get_gpu(trainer)
        epoch_time = (time.time() - self.start_time) / 3600
        if gpu is not None:
            torch.cuda.synchronize(gpu)
            max_memory = torch.cuda.max_memory_allocated(gpu) / 2 ** 20
        else:
            max_memory = 0.0
        try:
            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} hours")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except Exception:
            pass


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, fixed_input=False):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.fixed_input = fixed_input

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, fixed_input=self.fixed_input, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)
            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")


class PrintStepCallback(Callback):
    """nohup/pipe 환경에서도 stdout으로 학습 진행상황을 출력하는 콜백."""

    def __init__(self, print_every: int = 50):
        super().__init__()
        self.print_every = print_every
        self._t0 = None
        self._step0 = 0

    def on_train_start(self, trainer, pl_module):
        self._t0 = time.time()
        self._step0 = trainer.global_step
        print(f"[Train] 시작: max_steps={trainer.max_steps}, "
              f"dataset={trainer.num_training_batches} batches/epoch", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        step = trainer.global_step
        if step % self.print_every != 0:
            return
        # 경과 시간 및 속도 계산
        elapsed = time.time() - (self._t0 or time.time())
        steps_done = max(step - self._step0, 1)
        sps = steps_done / elapsed  # steps per second
        remaining = (trainer.max_steps - step) / sps if (sps > 0 and trainer.max_steps > 0) else -1
        eta = f"{remaining/3600:.1f}h" if remaining > 0 else "?"

        # 최신 logged_metrics에서 손실 값 추출
        metrics = trainer.logged_metrics
        loss_str = ""
        for key in ['train/loss', 'train/bit_loss', 'train/img_loss']:
            if key in metrics:
                loss_str += f"  {key.split('/')[-1]}={metrics[key]:.4f}"

        noise_on = getattr(pl_module, 'noise', None)
        noise_info = ""
        if noise_on is not None and hasattr(noise_on, 'is_activated'):
            noise_info = f"  noise={'ON' if noise_on.is_activated() else 'OFF'}"

        print(
            f"[step {step:>7}/{trainer.max_steps}]  epoch={trainer.current_epoch}"
            f"  {sps:.2f}it/s  ETA={eta}{loss_str}{noise_info}",
            flush=True
        )

    def on_train_epoch_end(self, trainer, pl_module):
        print(f"[Epoch {trainer.current_epoch} 완료]  global_step={trainer.global_step}", flush=True)
