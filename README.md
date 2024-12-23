cd model

python main.py --input_file data_for_prediction_example.csv --thres 0.5



#if input is bam file (e.g. the output of bwa/bowtie), use:
python main_csv_and_bam.py --input_file_name your_bam_file.bam --thres 0.5 -b --fa_name your_pirna_fasta_file.fa

