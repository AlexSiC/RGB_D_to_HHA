from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import re

def parse_depth_txt(path: Path) -> np.ndarray:
	text = path.read_text(encoding="utf-8", errors="ignore")
	rows: list[float] = []
	for line in text.splitlines():
		s = line.strip()
		if not s:
			continue
		# skip header lines like Width:, Height:, etc.
		if s[0].isalpha():
			continue
		parts = s.split(',')
		if len(parts) >= 3:
			try:
				v = float(parts[2])
			except Exception:
				continue
			rows.append(v)
	return np.array(rows, dtype=np.float64)


def main() -> None:
	ap = argparse.ArgumentParser(description="Analyze depth TXT file (rows: r,c,value_mm)")
	ap.add_argument("--path", required=True, help="Path to depth_data_frame_*.txt")
	args = ap.parse_args()

	p = Path(args.path)
	vals = parse_depth_txt(p)
	print(f"file: {p.name}")
	print(f"points total: {vals.size}")
	nz = vals[vals > 0]
	print(f"nonzero: {nz.size} ({(nz.size / max(1, vals.size)) * 100:.2f}%)")
	if nz.size:
		qs = np.percentile(nz, [0, 1, 5, 25, 50, 75, 95, 99, 100])
		labels = ["min", "p1", "p5", "p25", "median", "p75", "p95", "p99", "max"]
		print("mm stats:")
		for lab, val in zip(labels, qs):
			print(f"  {lab}: {val:.2f} mm")
		print(f"  mean: {nz.mean():.2f} mm, std: {nz.std():.2f} mm")
		lo, hi = float(qs[1]), float(qs[-2])
		if hi <= lo:
			lo = float(nz.min()); hi = float(nz.max())
			if hi == lo:
				hi = lo + 1.0
		hist, edges = np.histogram(nz, bins=15, range=(lo, hi))
		idx = np.argsort(hist)[-3:][::-1]
		print("dominant ranges (center mm, share):")
		for i in idx:
			center = 0.5 * (edges[i] + edges[i + 1])
			share = hist[i] / nz.size
			print(f"  ~{center:.1f} mm  ({share * 100:.2f}%)")
		print("meters:")
		print(f"  median: {qs[4] / 1000:.3f} m, mean: {nz.mean() / 1000:.3f} m")


if __name__ == "__main__":
	main()
