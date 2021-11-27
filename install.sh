CUDA=${1:-"cu111"}
TORCH=${2:-"1.9.0"}
echo "CUDA: ${CUDA} / TORCH: ${TORCH}"

if [ $CUDA = "cu111" ]; then
  pip3 install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
elif [ $CUDA = "cu102" ]; then
  pip3 install torch==${TORCH}
else
  echo "No match for CUDA: ${CUDA} / TORCH: ${TORCH}"
  exit
fi

pip3 install torch-scatter==2.0.8 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip3 install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip3 install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip3 install torch-geometric==2.0.1
pip3 install -r requirements.txt
echo "Install completed for CUDA: ${CUDA} / TORCH: ${TORCH}"
