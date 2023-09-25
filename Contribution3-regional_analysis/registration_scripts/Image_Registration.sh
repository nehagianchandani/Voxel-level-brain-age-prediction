#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --mem=7GB
#SBATCH --job-name=MNI_template_reg_%j
##SBATCH --mail-type=FAIL,TIME_LIMIT
##SBATCH --mail-user=#TODO: ADD EMAIL
#SBATCH --output=oasis_diseased_MNI_template_reg_%j.out

# Load Modules

echo ">>>>> Load Modules <<<<<<"
module load ants
echo ">>>>> Modules loaded <<<<"


##for step 1 - uncomment this when running commands_to_sbatch_registration
MOVING=$1 #- MNI template of brain
FIXED=$2 # - T1w image to which the MNI template will be registered
OUTPUT=$3

#for step 2 - uncomment this when running commands_to_sbatch_transformation
# MOVING=$1
# FIXED=$2
# TRANSFORM=$3
# OUTPUT=$4

#-t 's': This argument specifies the type of transformation model to be used for registration. In this case, 's' stands for "SyN", which refers to Symmetric Normalization. SyN is a powerful and commonly used algorithm for non-linear image registration in medical imaging.

#-p 'f': This argument specifies the type of gradient step size update schedule for the registration. In this context, 'f' typically stands for "Fixed", which means that the gradient step size remains constant throughout the registration process.

##step 1 - non-linear registration of mni TO T1 (FIXED IS T1W image and MOVING iss MNI TEMPLATE of brain (not atlas))
antsRegistrationSyN.sh -d 3 -f $FIXED -m $MOVING -o $OUTPUT -t 's' -p 'f'

#step 2 - transform brain_pad_mask using .mat file generated in step 1
#antsApplyTransforms -d 3 -i $MOVING -r $FIXED -o $OUTPUT --interpolation NearestNeighbor -t $TRANSFORM


