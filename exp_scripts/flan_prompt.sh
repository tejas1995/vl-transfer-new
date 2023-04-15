declare -a flan_sizes=("base"
         "large"
         "xl"
         "xxl")

declare -a properties=("color"
         "shape"
         "material")

declare -a icl_sizes=("4"
         "8"
         "16")

for icl_size in ${icl_sizes[@]}; do
    for flan_size in ${flan_sizes[@]}; do
        for property in ${properties[@]}; do
            /home/tejas/anaconda3/envs/vl/bin/python -W ignore -m evaluation.flan_t5_prompting \
                --flan_size ${flan_size} \
                --probe_property ${property} \
                --icl_size ${icl_size}
        done
    done
done