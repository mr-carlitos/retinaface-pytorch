##### CARLOS CODE FILE ########
from glob import glob
def first_try():
    val_imgs_dir = '/local/scratch/datasets/WiderFace/WIDER_val/images/'
    txt_filename = '/home/user/ckirchdorfer/carlos-workspace/Pytorch_Retinaface/data/widerface/val/wider_val.txt'
    filenames = glob('/local/scratch/datasets/WiderFace/WIDER_val/images/*/*.jpg')

    with open(txt_filename, 'a+', encoding='utf-8') as f:
        for filename in filenames:
            f.write(filename.replace(val_imgs_dir, '')+'\n')



def main_try(input_file, output_file):
    with open(input_file, 'r') as f_in:
        with open(output_file, 'w') as f_out:
            for line in f_in:
                line = line.strip()
                if line.startswith('#'):
                    # Remove the '#' and any leading/trailing whitespace
                    path = line[1:].strip()
                    f_out.write(f"{path}\n")

# Usage
input_file = '/home/user/ckirchdorfer/carlos-workspace/Pytorch_Retinaface/data/widerface/val/label.txt'  # Your input file
output_file = '/home/user/ckirchdorfer/carlos-workspace/Pytorch_Retinaface/data/widerface/val/wider_val.txt'  # Output file with just paths

main_try(input_file, output_file)