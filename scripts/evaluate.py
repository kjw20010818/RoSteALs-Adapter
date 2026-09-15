"""
evaluate.py — SmallAE Adapter 강건성 평가 스크립트
----------------------------------------------------
논문 평가 기준: ImageNet-C 15종 corruption × sev {3, 5} + Clean
bit accuracy 및 PSNR을 데이터셋별로 측정합니다.

Usage:
  python scripts/evaluate.py \
    --config  models/VQ4_small_ae_v2.yaml \
    --ckpt    /path/to/checkpoint.ckpt \
    --clic    /path/to/clic/images/ \
    --metfaces /path/to/metfaces/images/ \
    --out     results/eval.json
"""

import argparse, os, sys, glob, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
from tqdm import tqdm
from ldm.util import instantiate_from_config
from cldm.compat import safe_torch_load
from imagenet_c import corrupt, corruption_dict

IMG_TF = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# ImageNet-C 15종 (논문 기준 — fog 등 일부 제외)
IC_NAMES  = list(corruption_dict.keys())
TRAIN_IDS = [0,1,2,3,8,9,10,11,12,13,14,15,16,17,18]
IC_LABELS = [IC_NAMES[i] for i in TRAIN_IDS]


def load_model(cfg, ckpt, device):
    model = instantiate_from_config(OmegaConf.load(cfg).model)
    sd = safe_torch_load(ckpt, map_location="cpu")
    model.load_state_dict(sd.get("state_dict", sd), strict=False)
    return model.eval().to(device)


@torch.no_grad()
def embed(model, img_t, secret, device):
    imgs = img_t.unsqueeze(0).to(device)
    sec  = secret.unsqueeze(0).to(device)
    z    = model.encode_first_stage(imgs)
    out, _, _ = model(z, imgs, sec)
    return out[0].cpu()


def to_np(tensor):
    return ((tensor.clamp(-1,1)+1)/2*255).clamp(0,255).byte().permute(1,2,0).numpy()


def ic_corrupt(img_np, cid, sev):
    """BGR numpy → apply corruption → BGR numpy"""
    rgb = img_np[:,:,::-1].copy()
    small = np.array(Image.fromarray(rgb).resize((224,224), Image.BILINEAR))
    try:
        corrupted = corrupt(small, severity=sev, corruption_number=cid)
    except Exception:
        return img_np
    return np.array(Image.fromarray(corrupted[:,:,::-1]).resize(
        (img_np.shape[1], img_np.shape[0]), Image.BILINEAR))


def bit_acc(logits, secret_1d, device):
    pred = (logits > 0).float()
    return (pred == secret_1d.unsqueeze(0).to(device)).float().mean().item() * 100


def eval_dataset(model, paths, secret, device, desc="eval"):
    results = {"psnr": [], "clean": []}
    for name in IC_LABELS:
        results[f"ic_{name}_s3"] = []
        results[f"ic_{name}_s5"] = []

    for p in tqdm(paths, desc=desc):
        img_t = IMG_TF(Image.open(p).convert("RGB"))
        out_t = embed(model, img_t, secret, device)

        # PSNR
        a = (img_t.clamp(-1,1)+1)/2
        b = (out_t.clamp(-1,1)+1)/2
        mse = ((a-b)**2).mean().item()
        results["psnr"].append(-10*np.log10(mse+1e-10))

        out_np = to_np(out_t)

        # Clean
        with torch.no_grad():
            inp = IMG_TF(Image.fromarray(out_np)).unsqueeze(0).to(device)
            logits = model.decoder(inp)
            results["clean"].append(bit_acc(logits, secret, device))

        # IC attacks
        for sev in [3, 5]:
            for i, name in zip(TRAIN_IDS, IC_LABELS):
                atk = ic_corrupt(out_np, i, sev)
                with torch.no_grad():
                    inp = IMG_TF(Image.fromarray(atk)).unsqueeze(0).to(device)
                    logits = model.decoder(inp)
                    results[f"ic_{name}_s{sev}"].append(bit_acc(logits, secret, device))

    return {k: float(np.mean(v)) for k, v in results.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    required=True)
    p.add_argument("--ckpt",      required=True)
    p.add_argument("--clic",      default=None, help="CLIC 이미지 폴더")
    p.add_argument("--metfaces",  default=None, help="MetFaces 이미지 폴더")
    p.add_argument("--n_images",  type=int, default=None, help="데이터셋당 최대 이미지 수")
    p.add_argument("--device",    default="cuda:0")
    p.add_argument("--out",       default="results/eval.json")
    args = p.parse_args()

    torch.manual_seed(42)
    secret = torch.randint(0, 2, (100,)).float()

    model = load_model(args.config, args.ckpt, args.device)
    output = {}

    datasets = {}
    if args.clic:
        paths = sorted(glob.glob(f"{args.clic}/*.png") + glob.glob(f"{args.clic}/*.jpg"))
        if args.n_images:
            paths = paths[:args.n_images]
        datasets["clic"] = paths
    if args.metfaces:
        paths = sorted(glob.glob(f"{args.metfaces}/*.png") + glob.glob(f"{args.metfaces}/*.jpg"))
        if args.n_images:
            paths = paths[:args.n_images]
        datasets["metfaces"] = paths

    if not datasets:
        print("--clic 또는 --metfaces 중 하나는 필요합니다.")
        return

    for ds_name, paths in datasets.items():
        print(f"\n=== {ds_name.upper()} ({len(paths)} 이미지) ===")
        res = eval_dataset(model, paths, secret, args.device, desc=ds_name)
        output[ds_name] = res
        print(f"PSNR: {res['psnr']:.2f} dB  |  Clean acc: {res['clean']:.2f}%")
        s3_avg = np.mean([res[f"ic_{n}_s3"] for n in IC_LABELS])
        s5_avg = np.mean([res[f"ic_{n}_s5"] for n in IC_LABELS])
        print(f"IC-C sev=3 평균: {s3_avg:.2f}%  |  sev=5 평균: {s5_avg:.2f}%")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n결과 저장: {args.out}")


if __name__ == "__main__":
    main()
