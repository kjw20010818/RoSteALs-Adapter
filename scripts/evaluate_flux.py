"""
evaluate_flux.py — FLUX.1 VAE 백본 + SmallAE 어댑터 강건성 평가
=================================================================
측정 변형:
  flux_baseline  : FLUX epoch12, 어댑터 없음  W = G(z + eps)
  flux_v2        : FLUX + SmallAE 어댑터      W = H(G(z + eps))
  flux_v2_vq     : VQ 보정 적용              W = x + (H(G(z+eps)) - G(z))
  flux_v2_vq_b0  : VQ 보정 + B0 제거         W = x + blur_σ1(H(G(z+eps)) - G(z))

  (참고) vq_baseline   : VQ-GAN 원본 RoSteALS
  (참고) vq_v2_vq_b0   : VQ-GAN + SmallAE + VQ보정 + B0 (기존 최선)

Usage:
  python scripts/evaluate_flux.py \
    --flux_backbone_ckpt /hdd/watermark/model/rosteals_flux/checkpoints/epoch=000012-step=000150000.ckpt \
    --flux_adapter_ckpt  /hdd/watermark/model/flux_small_ae_v2/checkpoints/epoch=000002-step=000035000.ckpt \
    --vq_adapter_ckpt    /hdd/watermark/model/small_ae_v2_correct/checkpoints/epoch=000001-step=000035000.ckpt \
    --clic      /hdd/watermark/dataset/clic \
    --metfaces  /hdd/watermark/dataset/metfaces \
    --out       results/eval_flux_adapter.json
"""
import argparse, os, sys, glob, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cldm.post_g_adapter import ControlAEPostG
from imagenet_c import corrupt, corruption_dict

SECRET_LEN = 100
SEED       = 42
SEVERITIES = [3, 5]

IC_NAMES  = list(corruption_dict.keys())
TRAIN_IDS = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
IC_LABELS = [IC_NAMES[i] for i in TRAIN_IDS]

IMG_TF = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

FLUX_VARIANTS = ['flux_baseline', 'flux_v2', 'flux_v2_vq', 'flux_v2_vq_b0']
VQ_VARIANTS   = ['vq_baseline', 'vq_v2_vq_b0']
ALL_VARIANTS  = FLUX_VARIANTS + VQ_VARIANTS


def to_uint8(t):
    return torch.clamp((t + 1) * 127.5, 0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def gaussian_blur(x, sigma=1.0):
    k = int(2 * round(3 * sigma) + 1)
    ax = torch.arange(k, device=x.device) - k // 2
    g  = torch.exp(-ax.float() ** 2 / (2 * sigma ** 2))
    g  = (g / g.sum()).to(x.dtype)
    C  = x.shape[1]
    x  = F.conv2d(F.pad(x, (k//2, k//2, 0, 0), mode='reflect'),
                  g.view(1,1,1,k).expand(C,1,1,k), groups=C)
    return F.conv2d(F.pad(x, (0, 0, k//2, k//2), mode='reflect'),
                    g.view(1,1,k,1).expand(C,1,k,1), groups=C)


def ic_corrupt(img_np, cid, sev):
    h, w = img_np.shape[:2]
    bgr  = img_np[:, :, ::-1].copy()
    small = np.array(Image.fromarray(bgr).resize((224, 224), Image.BILINEAR))
    out   = corrupt(small, severity=sev, corruption_number=cid)
    return np.array(Image.fromarray(out[:, :, ::-1]).resize((w, h), Image.BILINEAR))


def load_flux_model(backbone_ckpt, adapter_ckpt, device):
    cfg   = OmegaConf.load('models/FLUX_small_ae_v2.yaml')
    # pretrain_ckpt는 YAML에 이미 있지만 인자로 덮어씀
    cfg.model.params.pretrain_ckpt = backbone_ckpt
    model = ControlAEPostG(**cfg.model.params).to(device).eval()
    # 어댑터 가중치 추가 로드
    sd = torch.load(adapter_ckpt, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd)
    adp_sd = {k: v for k, v in sd.items() if k.startswith('adapter.')}
    model.load_state_dict(adp_sd, strict=False)
    for p in model.parameters():
        p.requires_grad_(False)
    print(f'[FLUX] 백본={backbone_ckpt.split("/")[-1]}, 어댑터={adapter_ckpt.split("/")[-1]}')
    return model


def load_vq_model(vq_adapter_ckpt, device):
    cfg   = OmegaConf.load('models/VQ4_small_ae_v2.yaml')
    model = ControlAEPostG(**cfg.model.params).to(device).eval()
    sd = torch.load('models/RoSteALS/epoch=000017-step=000449999.ckpt',
                    map_location='cpu', weights_only=False)
    model.load_state_dict(sd.get('state_dict', sd), strict=False)
    sd = torch.load(vq_adapter_ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(sd.get('state_dict', sd), strict=False)
    for p in model.parameters():
        p.requires_grad_(False)
    print(f'[VQ] 어댑터={vq_adapter_ckpt.split("/")[-1]}')
    return model


@torch.no_grad()
def embed_flux(flux_model, img_t, secret, device):
    x = img_t.to(device)
    s = secret.to(device)

    z   = flux_model.encode_first_stage(x)
    Gz  = flux_model.decode_first_stage(z).clamp(-1, 1)

    eps, _ = flux_model.control(x, s)
    W_base = flux_model.decode_first_stage(z + eps).clamp(-1, 1)  # flux_baseline

    out, _, _ = flux_model(z, x, s)
    W_v2 = out.clamp(-1, 1)                                        # flux_v2

    wm      = W_v2 - Gz
    W_vq    = (x + wm).clamp(-1, 1)                                # flux_v2_vq
    W_b0    = (x + gaussian_blur(wm, 1.0)).clamp(-1, 1)            # flux_v2_vq_b0

    return {
        'flux_baseline': to_uint8(W_base[0]),
        'flux_v2':       to_uint8(W_v2[0]),
        'flux_v2_vq':    to_uint8(W_vq[0]),
        'flux_v2_vq_b0': to_uint8(W_b0[0]),
    }


@torch.no_grad()
def embed_vq(vq_model, img_t, secret, device):
    x = img_t.to(device)
    s = secret.to(device)

    z   = vq_model.encode_first_stage(x)
    Gz  = vq_model.decode_first_stage(z).clamp(-1, 1)

    eps, _ = vq_model.control(x, s)
    W_base = vq_model.decode_first_stage(z + eps).clamp(-1, 1)

    out, _, _ = vq_model(z, x, s)
    W_v2 = out.clamp(-1, 1)
    wm   = W_v2 - Gz
    W_b0 = (x + gaussian_blur(wm, 1.0)).clamp(-1, 1)

    return {
        'vq_baseline':  to_uint8(W_base[0]),
        'vq_v2_vq_b0':  to_uint8(W_b0[0]),
    }


def eval_dataset(flux_model, vq_model, paths, secret, device, desc='eval'):
    secret_np = secret.cpu().numpy()[0]

    results = {v: {'psnr': [], 'clean': []} for v in ALL_VARIANTS}
    for v in ALL_VARIANTS:
        for name in IC_LABELS:
            for sev in SEVERITIES:
                results[v][f'ic_{name}_s{sev}'] = []

    img_tf_back = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    for p in tqdm(paths, desc=desc, ncols=80):
        img_pil = Image.open(p).convert('RGB')
        img_t   = IMG_TF(img_pil).unsqueeze(0)
        orig_np = to_uint8(img_t[0])

        stegos = {}
        stegos.update(embed_flux(flux_model, img_t, secret, device))
        stegos.update(embed_vq(vq_model,   img_t, secret, device))

        for v, stego_np in stegos.items():
            mse = np.mean((orig_np.astype(np.float32) - stego_np.astype(np.float32)) ** 2)
            results[v]['psnr'].append(10 * np.log10(255**2 / (mse + 1e-10)))

            inp = img_tf_back(Image.fromarray(stego_np)).unsqueeze(0).to(device)
            # FLUX 변형은 FLUX decoder로, VQ 변형은 VQ decoder로 판독
            dec_model = flux_model if v.startswith('flux') else vq_model
            with torch.no_grad():
                pred = (dec_model.decoder(inp) > 0).cpu().numpy()[0]
            results[v]['clean'].append(float(np.mean(pred == secret_np) * 100))

            for cid, name in zip(TRAIN_IDS, IC_LABELS):
                for sev in SEVERITIES:
                    atk = ic_corrupt(stego_np, cid, sev)
                    inp = img_tf_back(Image.fromarray(atk)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        pred = (dec_model.decoder(inp) > 0).cpu().numpy()[0]
                    results[v][f'ic_{name}_s{sev}'].append(
                        float(np.mean(pred == secret_np) * 100))

    out = {}
    for v in ALL_VARIANTS:
        r = results[v]
        agg = {
            'psnr':  float(np.mean(r['psnr'])),
            'clean': float(np.mean(r['clean'])),
        }
        ic_vals = []
        for name in IC_LABELS:
            s3  = float(np.mean(r[f'ic_{name}_s3']))
            s5  = float(np.mean(r[f'ic_{name}_s5']))
            avg = (s3 + s5) / 2
            agg[f'ic_{name}_s3']  = s3
            agg[f'ic_{name}_s5']  = s5
            agg[f'ic_{name}_avg'] = avg
            ic_vals.append(avg)
        agg['ic_avg'] = float(np.mean(ic_vals))
        out[v] = agg
    return out


def print_table(res, ds_name):
    print(f'\n[{ds_name}]')
    print(f'{"변형":18} {"PSNR":>7} {"clean":>7} {"IC avg":>7}  vs vq_baseline  vs flux_baseline')
    print('-' * 75)
    ref_vq   = res['vq_baseline']['ic_avg']
    ref_flux = res['flux_baseline']['ic_avg']
    for v in ALL_VARIANTS:
        r = res[v]
        dvq   = r['ic_avg'] - ref_vq
        dflux = r['ic_avg'] - ref_flux
        print(f'  {v:16} {r["psnr"]:7.2f} {r["clean"]:7.2f} {r["ic_avg"]:7.2f} '
              f'  {dvq:+6.2f}pp        {dflux:+6.2f}pp')

    # 취약 공격 상세
    print(f'\n[{ds_name}] 취약 공격 상세 (IC_avg < 90)')
    weak_names = [n for n in IC_LABELS
                  if any(res[v][f'ic_{n}_avg'] < 90 for v in ALL_VARIANTS)]
    if weak_names:
        header = f'{"공격":22}'
        for v in ALL_VARIANTS:
            header += f' {v[:10]:>12}'
        print(header)
        for name in weak_names:
            row = f'  {name:20}'
            for v in ALL_VARIANTS:
                val = res[v][f'ic_{name}_avg']
                diff = val - res['vq_baseline'][f'ic_{name}_avg']
                row += f'  {val:5.1f}({diff:+4.1f})'
            print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flux_backbone_ckpt',
        default='/hdd/watermark/model/rosteals_flux/checkpoints/epoch=000012-step=000150000.ckpt')
    parser.add_argument('--flux_adapter_ckpt',
        default='/hdd/watermark/model/flux_small_ae_v2/checkpoints/epoch=000002-step=000035000.ckpt')
    parser.add_argument('--vq_adapter_ckpt',
        default='/hdd/watermark/model/small_ae_v2_correct/checkpoints/epoch=000001-step=000035000.ckpt')
    parser.add_argument('--clic',     default='/hdd/watermark/dataset/clic')
    parser.add_argument('--metfaces', default='/hdd/watermark/dataset/metfaces')
    parser.add_argument('--device',   default='cuda:0')
    parser.add_argument('--out',      default='results/eval_flux_adapter.json')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    secret = torch.randint(0, 2, (1, SECRET_LEN)).float()

    print('모델 로드 중...')
    flux_model = load_flux_model(args.flux_backbone_ckpt, args.flux_adapter_ckpt, args.device)
    vq_model   = load_vq_model(args.vq_adapter_ckpt, args.device)

    all_results = {}
    for ds_name, ds_dir in [('CLIC', args.clic), ('MetFaces', args.metfaces)]:
        if not os.path.isdir(ds_dir):
            print(f'[SKIP] {ds_name}: {ds_dir}')
            continue
        paths = sorted(glob.glob(f'{ds_dir}/*.png') + glob.glob(f'{ds_dir}/*.jpg'))
        print(f'\n[{ds_name}] {len(paths)}장', flush=True)
        np.random.seed(SEED)
        res = eval_dataset(flux_model, vq_model, paths, secret, args.device, desc=ds_name)
        all_results[ds_name] = res
        print_table(res, ds_name)

    json.dump(all_results, open(args.out, 'w'), indent=2)
    print(f'\n[저장] {args.out}')


if __name__ == '__main__':
    main()
