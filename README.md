# install lmms-eval

<!-- git clone https://github.com/EvolvingLMMs-Lab/lmms-eval -->

cd lmms-eval

pip install -e .

# install llava

<!-- for llava 1.5

cd lmms-eval

git clone https://github.com/haotian-liu/LLaVA

cd LLaVA

pip install -e . -->

# for llava-next (1.6)

cd lmms-eval

<!-- git clone https://github.com/LLaVA-VL/LLaVA-NeXT -->

cd LLaVA-NeXT

pip install -e .

# for evaluation

requires hugginface-cli login

# evaluation code

cd lmms-eval

./run_eval.sh