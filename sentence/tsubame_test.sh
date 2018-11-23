#!/bin/sh
##n current working directory
#$ -cwd
## Resource type F: qty 2
#$ -l f_node=2
## maximum run time
#$ -l h_rt=0:10:00
## output filename
#$ -N sample

. /etc/profile.d/modules.sh
module load python/3.6.5
module load intel
module load cuda
module load openmpi
#実行
source ~/imagin/bin/activate
source ~/.bash_profile
th translate.lua -model model/840B.300d.600rnn_epoch12_25.01.t7 -config config-trans
