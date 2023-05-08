declare -a properties=("color"
         "shape"
         "material")

declare -a icl_sizes=("0"
         "4"
         "8")

for icl_size in ${icl_sizes[@]}; do
    for property in ${properties[@]}; do
        /home/tejas/anaconda3/envs/vl/bin/python -W ignore -m evaluation.gpt3_prompting \
            --probe_property ${property} \
            --icl_size ${icl_size}
    done
done