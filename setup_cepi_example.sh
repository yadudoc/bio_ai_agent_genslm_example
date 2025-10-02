module load frameworks
module load tmux

source /home/yadunand/bio_ai_agent_genslm_example/venv_genslm/bin/activate

export CEPI_MODEL_CACHE_DIR=/home/yadunand/stash_dir/2.5B
export CEPI_CHECKPOINT_DIR=/home/yadunand/stash_dir/checkpoint-2929

export PYTHONPATH=FOO:/home/yadunand/bio_ai_agent_genslm_example/example:$PYTHONPATH
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE

