GLOBALLEN="4096"
MAXCTXLEN="3996"
GENLEN="100"

SEED=42
DEVICE="0,1" # the GPU device id
TOPP="0.9" # top-p sampling, set to 0.0 for greedy decoding

TESTFILE="fin|$1"
OUTPUT_PATH="$2"
bash run_group_decode_fileio.sh $SEED $DEVICE $TESTFILE $GLOBALLEN $MAXCTXLEN $GENLEN $TOPP $OUTPUT_PATH
