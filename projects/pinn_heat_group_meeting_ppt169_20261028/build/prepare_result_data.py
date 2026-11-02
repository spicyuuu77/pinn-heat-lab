import json
import math
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_saved import build_network, check_run
from train_compare_demo2 import evaluate

manifest = json.loads((ROOT / 'outputs/results.json').read_text(encoding='utf-8'))
run = next(r for r in manifest['runs'] if r['id'] == 'standard_independent')
config = manifest['common_config']
metrics = check_run(run, config)
net = build_network(config['hidden_layers'], config['hidden_width'])
net.load_state_dict(torch.load(ROOT / run['model'], map_location='cpu', weights_only=True))
_, x, predicted, exact = evaluate(net, config['alpha'])
history = torch.load(ROOT / run['history'], map_location='cpu', weights_only=True)

def rounded(values):
    return [round(float(v), 8) for v in values]

result = {
    'run': run['id'],
    'model': run['model'],
    'history': run['history'],
    'metrics': metrics,
    'x': rounded(x),
    'slices': [
        {'t': row / 100, 'predicted': rounded(predicted[row]), 'exact': rounded(exact[row])}
        for row in (0, 10, 30)
    ],
    'steps': list(range(1, len(history['total']) + 1)),
    'log10_total': rounded(math.log10(v) for v in history['total']),
    'loss_processing': 'log10 of each original total loss; no smoothing or downsampling',
}
(PROJECT / 'build/result_chart_data.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
print(f"Verified saved run: relative L2={metrics['relative_L2']:.8f}; 3 time slices; {len(history['total'])} original loss values.")
