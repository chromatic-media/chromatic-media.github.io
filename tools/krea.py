#!/usr/bin/env python3
"""Krea-2-Turbo text-to-image on NORMANDY's ComfyUI (10.0.0.153:8000), driven from VM 100.
Recipe from lab-ai-stack.md / the Sithis castle work: 8 steps, cfg 1, euler/simple, negatives zeroed.
Usage: krea.py jobs.json outdir      jobs = [{"name","prompt","seed","w","h", "lora"?, "lora_strength"?}, ...]
Submits every job, waits for all, downloads the PNGs into outdir as <name>.png."""
import json, sys, time, urllib.request, urllib.parse, pathlib
HOST = "http://10.0.0.153:8000"

def graph(j):
    g = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "krea2_turbo_fp8_scaled.safetensors", "weight_dtype": "default"}},
         "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b_fp8_scaled.safetensors", "type": "krea2", "device": "default"}},
         "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
         "4": {"class_type": "CLIPTextEncode", "inputs": {"text": j["prompt"], "clip": ["2", 0]}},
         "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
         "6": {"class_type": "EmptyLatentImage", "inputs": {"width": j.get("w", 1344), "height": j.get("h", 768), "batch_size": 1}},
         "9": {"class_type": "KSampler", "inputs": {"seed": j["seed"], "steps": j.get("steps", 8), "cfg": 1.0, "sampler_name": "euler",
               "scheduler": "simple", "denoise": 1.0, "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
         "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
         "11": {"class_type": "SaveImage", "inputs": {"filename_prefix": "cm_" + j["name"], "images": ["10", 0]}}}
    if j.get("image"):   # img2img: start from an existing picture, keep its layout (denoise < 1)
        g["6"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["13", 0], "vae": ["3", 0]}}
        g["13"] = {"class_type": "LoadImage", "inputs": {"image": j["upload"]}}
        g["9"]["inputs"]["denoise"] = j.get("denoise", 0.5)
    if j.get("lora"):
        g["12"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": j["lora"], "strength_model": j.get("lora_strength", 0.8), "model": ["1", 0]}}
        g["9"]["inputs"]["model"] = ["12", 0]
    return g

def post(g):
    req = urllib.request.Request(HOST + "/prompt", data=json.dumps({"prompt": g}).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=30))
    if r.get("node_errors"):
        raise SystemExit("node errors: " + json.dumps(r["node_errors"])[:600])
    return r["prompt_id"]

def main():
    jobs = json.load(open(sys.argv[1])); out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    import subprocess
    for j in jobs:
        if j.get("image"):
            import hashlib
            j["upload"] = "cm_k_" + hashlib.sha1(str(pathlib.Path(j["image"]).resolve()).encode()).hexdigest()[:10] + "_" + pathlib.Path(j["image"]).name
            subprocess.run(["curl", "-s", "-F", f"image=@{j['image']};filename={j['upload']}", "-F", "overwrite=true", HOST + "/upload/image"],
                           stdout=subprocess.DEVNULL, check=True)
    t0 = time.time(); pending = {post(graph(j)): j for j in jobs}
    print(f"queued {len(pending)} jobs")
    while pending and time.time() - t0 < 1800:
        time.sleep(3)
        for pid in list(pending):
            h = json.load(urllib.request.urlopen(f"{HOST}/history/{pid}", timeout=30))
            if pid not in h:
                continue
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                print("ERROR", pending.pop(pid)["name"]); continue
            if st.get("completed"):
                j = pending.pop(pid)
                for node in h[pid]["outputs"].values():
                    for im in node.get("images", []):
                        q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im.get("subfolder", ""), "type": im.get("type", "output")})
                        (out / f"{j['name']}.png").write_bytes(urllib.request.urlopen(f"{HOST}/view?{q}", timeout=60).read())
                print(f"done {j['name']}  ({time.time() - t0:.0f} s)")
    if pending:
        print("TIMED OUT:", [j["name"] for j in pending.values()])

if __name__ == "__main__":
    main()
