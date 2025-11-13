import argparse, subprocess, re, csv, sys
from pathlib import Path
import matplotlib.pyplot as plt

PATTERN = re.compile(r'^(grayscale|boxblur\(3x3\)|sobel)\s+CPU:\s+([0-9.]+)\s+ms\s+GPU:\s+([0-9.]+)\s+ms\s+Speedup:\s+([0-9.]+)x')

def run_benchmark(bin_path, image_path, iters):
    out = subprocess.check_output([str(bin_path), str(image_path), str(iters)], stderr=subprocess.STDOUT, text=True)
    return out

def parse_output(text):
    rows = []
    width = height = iters = None
    for line in text.splitlines():
        if line.startswith("Input:"):
            try:
                head, rest = line.split(",", 1)
                wh = head.split(":")[1].strip()
                width, height = map(int, wh.split("x"))
                iters = int(rest.strip().split("=")[1])
            except Exception:
                pass
        m = PATTERN.search(line)
        if m:
            op, cpu_ms, gpu_ms, speedup = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
            rows.append({"op": op, "cpu_ms": cpu_ms, "gpu_ms": gpu_ms, "speedup": speedup})
    return rows, width, height, iters

def write_csv(rows, width, height, iters, csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op","cpu_ms","gpu_ms","speedup","width","height","iters"])
        for r in rows:
            w.writerow([r["op"], r["cpu_ms"], r["gpu_ms"], r["speedup"], width, height, iters])

def plot(rows, png_path, title=None):
    ops = [r["op"] for r in rows]
    cpu = [r["cpu_ms"] for r in rows]
    gpu = [r["gpu_ms"] for r in rows]
    speed = [r["speedup"] for r in rows]

    x = range(len(ops))
    fig = plt.figure(figsize=(7,4.5))
    width = 0.35
    plt.bar([i - width/2 for i in x], cpu, width, label="CPU (ms)")
    plt.bar([i + width/2 for i in x], gpu, width, label="GPU (ms)")

    for i, s in enumerate(speed):
        y = max(cpu[i], gpu[i])
        plt.text(i, y*1.02 if y>0 else 0.01, f"{s:.1f}×", ha="center", va="bottom", fontsize=9)

    plt.xticks(list(x), ops)
    plt.ylabel("Time (ms)")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    fig.savefig(png_path, dpi=160)

def main():
    ap = argparse.ArgumentParser(description="Run CUDA benchmark and plot CPU vs GPU performance.")
    ap.add_argument("--bin", default="bin/benchmark", help="Path to benchmark binary")
    ap.add_argument("--image", required=True, help="Path to input PPM image")
    ap.add_argument("--iters", type=int, default=10, help="Iterations to average")
    ap.add_argument("--csv", default="bench.csv", help="Where to write CSV")
    ap.add_argument("--png", default="bench.png", help="Where to write plot PNG")
    ap.add_argument("--no-run", action="store_true", help="Do not run benchmark; read from STDIN instead")
    args = ap.parse_args()

    if args.no_run:
        text = sys.stdin.read()
        rows, W, H, iters = parse_output(text)
    else:
        text = run_benchmark(Path(args.bin), Path(args.image), args.iters)
        rows, W, H, iters = parse_output(text)

    if not rows:
        print("Could not parse any benchmark rows. Did the binary run correctly?", file=sys.stderr)
        sys.exit(2)

    write_csv(rows, W, H, iters, args.csv)
    title = f"CPU vs GPU (Input {W}x{H}, iters={iters})"
    plot(rows, args.png, title=title)
    print(f"Wrote {args.csv} and {args.png}")

if __name__ == "__main__":
    main()
