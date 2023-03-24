https://help.rc.ufl.edu/doc/Getting_Started#Connecting_from_Windows

Web Interfaces to login to Hypergator
•	https://ood.rc.ufl.edu
•	Jupyter Notebook : https://jhub.rc.ufl.edu 

select university as University of Florida and enter Gatorlink user name and password to login


After logging into https://ood.rc.ufl.edu    there will be various options on top menu bar for Files, Jobs, Clusters, Interactive Apps.


Some Useful Commands:

ncdu – to see memory usage of the current working directory.

ps to see running processes



(StableDif2) [padamatinti.s@c1107a-s35 finetune-sd]$ ps
   PID TTY          TIME CMD
 12661 pts/0    00:00:01 python3.10
 12725 pts/0    00:00:00 python3.10
 12790 pts/0    00:00:00 python3.10
 12855 pts/0    00:00:00 python3.10
 12932 pts/0    00:00:00 ps
105530 pts/0    00:00:01 accelerate
105579 pts/0    00:04:13 python3.10
110697 pts/0    00:00:00 bash


https://help.rc.ufl.edu/doc/Checking_and_Using_Secondary_Resources

$ id   - - shows the user id and group id

$ showAssoc <username>
to get the QOS group and group name.

$ showQos <specified_qos>

	to get the Qos group limits.

[padamatinti.s@login5 finetune-sd]$ showAssoc padamatinti.s
User               Account        Def Acct       Def QOS        QOS
------------------ -------------- -------------- -------------- ----------------------------------------
padamatinti.s      subramanianh   subramanianh   subramanianh   subramanianh,subramanianh-b

[padamatinti.s@login5 finetune-sd]$ showQos subramanianh
Name                 Descr                          GrpTRES                                        GrpCPUs
-------------------- ------------------------------ --------------------------------------------- --------
subramanianh         subramanianh qos               cpu=4,gres/gpu=1,mem=32000M                          4



[padamatinti.s@login6 ~]$ showQos subramanianh-b
Name                 Descr                          GrpTRES                                        GrpCPUs
-------------------- ------------------------------ --------------------------------------------- --------
subramanianh-b       subramanianh burst qos         cpu=36,gres/gpu=0,mem=288000M                       36


Instructions as per https://github.com/harrywang/finetune-sd#setup
To create a environment with required libraries.
module load conda
Conda create -n StableDif2 python= 3.10.9
conda activate StableDif2
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install torch-geometric

module load git

git clone https://github.com/harrywang/finetune-sd.git

pip install -r requirements.txt

pip install triton==2.0.0.dev20221120

create an account at https://huggingface.co/settings/tokens
copy the token from User Access Tokens: section :

•  login to HuggingFace using your token: huggingface-cli login 
paste the token copied from User Access Tokens.
•  login to WandB using your API key: wandb login
You can find your API key in your browser here: https://wandb.ai/authorize
login with github.
Requesting a GPU node in interactive mode:
srun -p gpu --nodes=1 --gpus=a100:1 --time 600 --ntasks=1 --cpus-per-task=4 --mem=32000MB --pty -u bash -i

module load conda
conda activate StableDif2

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"
export OUTPUT_DIR="./models/lora/pokemon"

accelerate config default	
	above command is to use the GPU


accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=4 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=150 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=30 \
  --validation_prompt="Totoro" \
  --seed=42 \
  --report_to=wandb >logfile2 2>&1 &

Returns process id : 130457

in the git clone command in the last line  >logfile2 2>&1 &

>logfile2 is to write the logs to file named logfile2
2>&1 is to write the error  to standard output.
the last & is to run the process in background.


