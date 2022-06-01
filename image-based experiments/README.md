# WSRL

To reproduce the numbers in Table 3, call

 python ball_code.py --workers 6 --batch-size 100 --seed 1 --n_balls [n_balls] --lr 1e-4 [--block_offset] [--combinations]
 
 With [n_balls] \in {2,3,4}. Add the [--block_offset] flag for the second column and [--combinations] for the third column.
