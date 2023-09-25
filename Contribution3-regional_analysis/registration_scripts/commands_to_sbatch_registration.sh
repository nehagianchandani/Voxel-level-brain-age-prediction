#!/bin/bash
# THe purpose of this code is to generate registrations over a loop, it basically generates SBTACH jobs one for every image


FIXED_PATH='XXXX/MNI152_T1_1mm_brain.nii'  # Here wil be the path to MNI brain template (1mm voxel size)

MOVING_PATH='XXXX' 
# Here will be the path to the moving images (directory which has all PAD maps, amd T1w images saved)
# Note don't add the last back slash in the path


# If the registration doesn't work properly, you'll have to change this parameter the most common number that I used were: 0.01, 0.2, 0.3, 0.4, 0.5, feel free to change them as you wish.
REGULAR_STEP_GRADIENT=0.4  

shopt -s extglob
echo ">>>> START SENDING COMMANDS AND PRODUCE SBATCH REGISTRATION JOBS <<<<<<"

PATTERN_1='orig'
counter=0

for file in $(find $MOVING_PATH -maxdepth 2 -name "$PATTERN_1*" -type f); do
   folder=$(dirname $file)

   MRI_FILE=${file%%+(/)}
   MRI_FILE=${MRI_FILE##*/}
   MRI_FILE_COMPLETE="${folder}/${MRI_FILE}"


   sub_info="${MRI_FILE%.nii*}"
   sub_info="${sub_info#orig_*}"  # Keep everything to the right of orig
   transform_name="${folder}/${sub_info}_"


   output_img="${folder}/${sub_info}_"



   sbatch ./Image_Registration.sh ${FIXED_PATH} ${file} ${output_img}

   ((counter++))
   echo "$counter"

done

