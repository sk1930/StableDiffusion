These instructions are documented in InstructionsHPG.docx


Login to hypergator using putty:
Hostname: hpg.rc.ufl.edu
port: 22
connection type: ssh
Login to hypergator using Winscp (SFTP):
hostname: hpg.rc.ufl.edu

Web Interfaces to login to Hypergator
•	https://ood.rc.ufl.edu
•	Jupyter Notebook : https://jhub.rc.ufl.edu 

	select university as University of Florida and enter Gatorlink username and password to login. After logging into https://ood.rc.ufl.edu  there will be various options on top menu bar for Files, Jobs, Clusters, Interactive Apps.

Some Useful Commands:

1.	ncdu – to see memory usage of the current working directory.
2.	ps     --  to see running processes
3.	TO run a process in Background enter & in the end of the command.
4.	id   - - shows the user id and group id in hypergator.
5.	showAssoc <username>
to get the QOS group and group name.
6.	showQos <specified_qos>
	to get the Qos group limits.


Environment Setup for StableDiffusion - https://github.com/harrywang/finetune-sd#setup
To run the fine-tuning Stable Diffusion Code do all the steps from step number 1 to number 8.
1.	To create a environment with conda first load conda using
module load conda
2.	To create an environment named StableDif2 with python 3.10.9 
Conda create -n StableDif2 python= 3.10.9
3.	Activate the StableDIf2 Environment using
conda activate StableDif2
4.	Install Pytorch using below commands
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install torch-geometric
5.	Load the git module using
  module load git


6.	Clone the repository
git clone https://github.com/harrywang/finetune-sd.git
7.	cd (change directory) in to the directory containing requirements.txt
8.	Install the python libraries
	pip install -r requirements.txt
9.	I got an error running the accelerate launch train_text_to_image_lora.py 
So I have installed triton to overcome that error. 
pip install triton==2.0.0.dev20221120

HuggingFace and WandB account creation
1.	create an account at https://huggingface.co/settings/tokens
2.	copy the token from User Access Tokens: section :
3.	to login to HuggingFace using your token use the command:
huggingface-cli login 
	It will prompt you to enter the token
paste the token copied from User Access Tokens.
4.	Create an account at  https://wandb.ai/authorize
To create the account you can signin with github.
5.	To login to WandB using your API key use the command :
wandb login
6.	It will prompt you to enter the API key:
You can find the API key at  https://wandb.ai/authorize

Fine-tune using Dreambooth with LoRA
1. Requesting a GPU node in interactive mode 
srun -p gpu --nodes=1 --gpus=a100:1 --time 600 --ntasks=1 --cpus-per-task=4 --mem=32000MB --pty -u bash -i
--time 600 is 600 minutes
--mem= 32000MB  is 32 GB 
2.  load conda and activate the environment
module load conda
conda activate StableDif2
accelerate config default	
	above command is to use the GPU






export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/dreambooth/dog"
export OUTPUT_DIR="./models/dreambooth-lora/dog"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=20 \
  --seed=42 \
  --report_to="wandb" >logfile2 2>&1 &

Returns process id : 130457

in the accelerate launch train_dreambooth command in the last line  >logfile2 2>&1 &

>logfile2 is to write the logs to file named logfile2
2>&1 is to write the error  to standard output.
the last & is to run the process in background.


To test the model:
python generate-lora.py --prompt "A photo of sks dog near lake" --model_path "./models/dreambooth-lora/dog" --output_folder "./outputs" --steps 400







Link to the help document.
https://help.rc.ufl.edu/doc/Getting_Started#Connecting_from_Windows
https://help.rc.ufl.edu/doc/Checking_and_Using_Secondary_Resources

https://help.rc.ufl.edu/doc/Slurm_and_GPU_Use

https://help.rc.ufl.edu/doc/GPU_Access


