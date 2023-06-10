### Teeth Tracking

Train the network:
```bash
python train.py
```

Inference with demo data:
```bash
python inference.py --data ./data/demo/images --ckp ./outputs/checkpoint/checkpoint_099.pth
```

Triangulate the results to get 3D coordinates of face landmarks and teeth keypoints:
```bash
python triangulate.py
```

Apply Kalman filter and visualize:
```bash
python vis.py
```
