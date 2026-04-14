Download/prep HierText:

```bash
bash scripts/download_hiertext.sh
```

Create max-300-line JSONLs:

```bash
bash scripts/filter_hiertext_jsonl_max300.sh
```

Training:

```bash
bash scripts/experiments/train_bbox_2d.sh
bash scripts/experiments/train_poly_mlp.sh
bash scripts/experiments/train_poly_transformer.sh
```

Test dataset generation:

```bash
bash scripts/experiments/test_bbox_2d.sh
bash scripts/experiments/test_poly_mlp.sh
bash scripts/experiments/test_poly_transformer.sh
```
