"""Run the full pipeline end to end."""
import subprocess, sys
for step in ["collect_data", "embed_cluster", "train", "ablation", "emerging_topics"]:
    print(f"\n########## {step} ##########", flush=True)
    subprocess.run([sys.executable, f"src/{step}.py"], check=True)
