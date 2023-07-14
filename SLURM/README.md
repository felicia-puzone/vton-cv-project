# Summary
- unet_job_request.sh -> request job for training unet architecture; it relies on unet_train.sh
- unet_train.sh -> activate python env and run python training script of the unet architecture


# SLURM
- srun command doc: https://slurm.schedmd.com/srun.html
- sbatch command doc: https://slurm.schedmd.com/sbatch.html
- (!!!) slurm launcher examples: https://hpc.uni.lu/old/users/docs/slurm_launchers.html
- AImageLab wiki: https://ailb-web.ing.unimore.it/wiki/



# Linux utilities

### Screen command
- screen command tutorial and examples: https://linuxize.com/post/how-to-use-linux-screen/
- `screen: screen -S prova` -> activate a session named "prova"
- `ctrl+a`, then `ctrl+d` -> detach screen session
- `screen -ls` -> list all active sessions
- `screen -r session_to_join_with` -> join the session
