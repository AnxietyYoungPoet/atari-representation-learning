import subprocess

base_cmd = "sbatch"
ss= "exp_scripts/run_gpu.sl"
module = "scripts.run_probe"
args = [base_cmd, ss, module]
args.append("--method random_cnn")
args.append("--num-frame-stack 1")
args.append("--num-processes 1")
envs = ['asteroids', 'berzerk', 'boxing',
        'demon_attack', 'enduro', 'freeway', 'frostbite', 'hero', 
        'ms_pacman', 'pong', 'private_eye', 'qbert', 'riverraid', 
        'seaquest', 'solaris', 'space_invaders', 'venture', 'video_pinball', 
        'yars_revenge','breakout','pitfall','montezuma_revenge'
        ]


suffix = "NoFrameskip-v4"
for i,env in enumerate(envs):
    
    names = env.split("_")
    name = "".join([s.capitalize() for s in names])
    sargs = args + ["--env-name"]
    
    sargs.append(name + suffix) 
    
    
    subprocess.run(sargs)