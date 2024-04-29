from pathlib import Path
from time import sleep

from huggingface_hub import snapshot_download


repo_name = "BramVanroy/fietje-2b-sft"
ckpt = "534"
output_path = f"/home/ampere/vanroy/fietje-test/fietje-2b-sft/fietje-2b-sft-ckpt-{ckpt}"
allow_patterns = [
    f"checkpoint-{ckpt}/*",
]
ignore_patterns = [
    f"checkpoint-{ckpt}/global_step*",
    "*latest",
    "*.pth",
    "*.pt",
    "*.bin",
    "*.py",
    "*trainer_state.json",
]
pdout = Path(output_path).resolve()
must_retry = not pdout.exists() or len(list(pdout.iterdir())) == 0
while must_retry:
    output_path = snapshot_download(
        repo_id=repo_name,
        local_dir=output_path,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )

    must_retry = not pdout.exists() or len(list(pdout.iterdir())) == 0
    if must_retry:
        print(f"Retrying for checkpoint {ckpt} in 60s...")
        sleep(60)

print(f"Downloaded to {output_path}")
