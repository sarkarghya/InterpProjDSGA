I compiled a short layman version of Resouces found [Cluster support](https://github.com/nyu-dl/cluster-support/tree/master/greene) and [Website](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/getting-started?authuser=0)
This will help you set up singualrity (a virtual computer). Note that you can still submit jobs using slurm.

## Setting Up Your Singularity container

1. Copy the empty filesystem image to your scratch directory:
   ```bash
   cp /scratch/work/public/overlay-fs-ext3/overlay-50G-10M.ext3.gz $SCRATCH/
   ```

2. Unzip the archive:
   ```bash
   gunzip -v $SCRATCH/overlay-50G-10M.ext3.gz
   ```

3. Start an interactive session to set up your environment:
   ```bash
   srun --nodes=1 --tasks-per-node=1 --cpus-per-task=1 --mem=32GB --time=1:00:00 --gres=gpu:1 --pty /bin/bash
   ```

4. Launch the Singularity container in read-write mode:
   ```bash
   singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash
   ```

5. Install conda and necessary packages in the overlay:
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash ./Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
   ```

6. Create a helper script for conda activation at `/ext3/env.sh`:
   ```bash
   #!/bin/bash
   source /ext3/miniconda3/etc/profile.d/conda.sh
   export PATH=/ext3/miniconda3/bin:$PATH
   ```

7. Install dependencies if any (example):
   ```bash
   source /ext3/env.sh
   conda activate
   pip install deepseek-ai # or the appropriate package for deepseek
   ```

## Running Inference as a Batch Job

Create a batch script (e.g., `deepseek_job.slurm`) with the following content:

```bash
#!/bin/bash
#SBATCH --job-name=deepseek_inference
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:rtx8000:1  # Specify RTX8000 GPU
#SBATCH --mem=64G
#SBATCH -c 4

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/env.sh
conda activate

# Clone the deepseek repository if needed
# git clone https://github.com/deepseek-ai/deepseek-coder.git
# cd deepseek-coder

# Run your inference script
python your_inference_script.py --model_path /path/to/model --input_data /path/to/data
"
```

Submit the job with:
```bash
sbatch deepseek_job.slurm
```

To monitor your jobs use `squeue --user=your_netid` and check output files for results.
