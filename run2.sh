#!/bin/bash
set -eux
envs=("Asteroids" "Berzerk" "Bowling" "Boxing" "Breakout" "DemonAttack" "Freeway" "Frostbite" "Hero" "MontezumaRevenge" "MsPacman" "Pitfall" "Pong" "PrivateEye" "Qbert" "Riverraid" "Seaquest" "SpaceInvaders" "Tennis" "Venture" "VideoPinball" "YarsRevenge")
for env in ${envs[@]}
do
	CUDA_VISIBLE_DEVICES=$1 python -m scripts.run_probe --method ib --env-name "${env}NoFrameskip-v4" --wandb-name "${env}-IB" --wandb-proj InfoBottle1-z --beta 0.001
done
