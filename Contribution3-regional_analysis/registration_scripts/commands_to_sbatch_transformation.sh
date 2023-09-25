#!/bin/bash
# THe purpose of this code is to generate registrations over a loop, it basically generates SBTACH jobs one for every image


FIXED_PATH='XXXX/MNI-maxprob-thr50-1mm.nii.gz' # patht to the MNI atlas with 9 regions

MOVING_PATH='XXXX' 
# Here will be the path to the moving images (directory which has all PAD maps, amd T1w images saved)
# Note don't add the last back slash in the path

shopt -s extglob
echo ">>>> START SENDING COMMANDS AND PRODUCE SBATCH REGISTRATION JOBS <<<<<<"
counter=0  # Initialize the counter

PATTERN_1='orig'

for file in $(find $MOVING_PATH -maxdepth 2 -name "$PATTERN_1*" -type f); do
   counter=$((counter + 1))  # Increment the counter

   folder=$(dirname $file)

   MRI_FILE=${file%%+(/)}

   MRI_FILE=${MRI_FILE##*/}


   sub_info="${MRI_FILE%.nii*}" # KEEP EVERYTHING TO THE L;EFT OF .NII


   sub_name="${sub_info#orig_}"

   transform_suffix="_0GenericAffine.mat"
   transform_name="${sub_name}${transform_suffix}"
   transform_name_complete="${folder}/${transform_name}"

   input_file="${folder}/${MRI_FILE}"



   output_img="${folder}/${sub_name}_MNI_brain.nii.gz"


   #next line is command for step 2

   sbatch ./Image_Registration.sh ${FIXED_PATH} ${input_file} ${transform_name_complete} ${output_img}

   	

done

