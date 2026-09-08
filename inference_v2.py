"""
SmallAE 어댑터 + VQ 오차 보정 파이프라인용 inference.

원본 inference.py 와의 차이:
    원본  stego = G(z + eps)                       (VQ 재구성 오차가 그대로 남음)
    본안  stego = x_orig + (adapter(G(z+eps)) - G(z))
          → 워터마크 신호만 원본에 얹어 VQ 재구성 오차를 제거

사용법:
    python inference_v2.py \
        --config models/VQ4_small_ae_v2.yaml \
        --base_weight models/RoSteALS/epoch=000017-step=000449999.ckpt \
        --adapter_weight /hdd/watermark/model/small_ae_v2_correct/checkpoints/epoch=000001-step=000035000.ckpt \
        --secret Secrets --cover examples/monalisa.jpg --output examples/monalisa_v2.png
"""
import argparse, warnings
import numpy as np
import torch
import lpips
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms

from cldm.post_g_adapter import ControlAEPostG
from cldm.compat import safe_torch_load
from tools.ecc import ECC
from tools.sifid import SIFID
from tools.eval_metrics import compute_psnr, compute_ssim, compute_mse, compute_lpips, compute_sifid

warnings.filterwarnings('ignore')


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ControlAEPostG(**OmegaConf.load(args.config).model.params).to(device).eval()
    for path in (args.base_weight, args.adapter_weight):
        if path:
            sd = safe_torch_load(path, map_location='cpu')
            model.load_state_dict(sd.get('state_dict', sd), strict=False)
    for p in model.parameters():
        p.requires_grad_(False)

    tform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    cover_org = Image.open(args.cover).convert('RGB')
    w, h      = cover_org.size
    cover     = tform(cover_org).unsqueeze(0).to(device)

    ecc    = ECC()
    secret = torch.from_numpy(ecc.encode_text([args.secret])).to(device).float()

    with torch.no_grad():
        z   = model.encode_first_stage(cover)
        Gz  = model.decode_first_stage(z).clamp(-1, 1)
        out = model(z, cover, secret)
        W   = (out[0] if isinstance(out, (list, tuple)) else out).clamp(-1, 1)

        # VQ 오차 보정: 순수 워터마크 신호만 추출
        res = W - Gz if args.vq_correction else W - cover

        # B0(최고주파) 대역은 잔차 에너지의 44%를 차지하면서 강건성 기여는 0.6%p 뿐.
        # 제거하면 PSNR 이 약 1.9dB 오른다.
        if args.drop_b0:
            from cldm.post_g_adapter import _gaussian_blur
            res = _gaussian_blur(res, 1.0)

        # 원본 해상도로 잔차를 올려 적용
        res_full = torch.nn.functional.interpolate(res, (h, w), mode='bilinear', align_corners=False)
        res_full = res_full.permute(0, 2, 3, 1).cpu().numpy()[0]
        stego_u8 = np.clip(res_full + np.array(cover_org) / 127.5 - 1., -1, 1) * 127.5 + 127.5
        stego_u8 = stego_u8.astype(np.uint8)

        cov_np = np.array(cover_org)[None, ...]
        stg_np = stego_u8[None, ...]
        print(f'Quality metrics at resolution: {h}x{w} (HxW)')
        print(f'MSE:   {compute_mse(cov_np, stg_np)}')
        print(f'PSNR:  {compute_psnr(cov_np, stg_np)}')
        print(f'SSIM:  {compute_ssim(cov_np, stg_np)}')

        to_t = lambda a: torch.from_numpy(a / 127.5 - 1.).permute(0, 3, 1, 2).float().to(device)
        print(f'LPIPS: {compute_lpips(to_t(cov_np), to_t(stg_np), lpips.LPIPS(net="alex").to(device))}')
        print(f'SIFID: {compute_sifid(to_t(cov_np), to_t(stg_np), SIFID())}')

        # 저장된 파일(원본 해상도 uint8)에서 되읽어 실제 추출 성능을 확인
        stego_256  = tform(Image.fromarray(stego_u8)).unsqueeze(0).to(device)
        pred       = (model.decoder(stego_256) > 0).cpu().numpy()
        print('\nExtracting secret...')
        print(f'Bit acc: {np.mean(pred == secret.cpu().numpy())}')
        print(f'Recovered secret: {ecc.decode_text(pred)[0]}')

    Image.fromarray(stego_u8).save(args.output)
    print(f'Stego saved to {args.output}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', default='models/VQ4_small_ae_v2.yaml')
    p.add_argument('--base_weight',    default='models/RoSteALS/epoch=000017-step=000449999.ckpt')
    p.add_argument('--adapter_weight',
                   default='/hdd/watermark/model/small_ae_v2_correct/checkpoints/epoch=000001-step=000035000.ckpt')
    p.add_argument('--image_size', type=int, default=256)
    p.add_argument('--secret',  default='Secrets', help='secret message, 7 characters max')
    p.add_argument('--cover',   default='examples/monalisa.jpg')
    p.add_argument('-o', '--output', default='examples/monalisa_v2.png')
    p.add_argument('--vq_correction', action='store_true', default=True)
    p.add_argument('--no_vq_correction', dest='vq_correction', action='store_false')
    p.add_argument('--drop_b0', action='store_true', help='최고주파 대역 제거 (PSNR +1.9dB, IC -0.6%p)')
    main(p.parse_args())
