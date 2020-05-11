#!/bin/bash
# init.sh

set -Eeuo pipefail

usage() {
	echo "Usage: $0 (init|clear|install|generate|train)"
	echo "init | create output directories"
	echo "clear | remove output directories"
	echo "setup | setup python packages"
	echo "generate | generate synthetic data files"
	echo "train | train models"
}

if [ $# -ne 1 ]; then
  usage
  exit 1
fi

case $1 in
	"init")
		mkdir -p data
		mkdir -p checkpoints
		mkdir -p plots
		mkdir -p output
		;;

	"clear")
		rm -rf data checkpoints plots output
		;;

	"setup")
		python3 -m venv env
		source env/bin/activate		
		pip install --upgrade pip setuptools wheel
		pip install numpy
		pip install pandas
		pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
		pip install torch-scatter==latest+cpu torch-sparse==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.4.0.html
		pip install torch_geometric
		cp bin/__init__.pyi env/lib/python*/site-packages/torch/optim/__init__.pyi		
		;;

	"generate")
		source env/bin/activate
		chmod +x generate_data.py
		python3 ./generate_data.py
		;;

	"train")
		source env/bin/activate
		chmod +x train.py
		python3 ./train.py
		;;

	*)
		usage
		exit 1
		;;

esac
