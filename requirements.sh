pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-geometric
pip install ray[tune]

pip install -r requirements.txt

pip install -U kaleido

TMPDIR=/tmp HOROVOD_WITH_GLOO=1 pip install horovod