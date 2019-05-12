#!/bin/bash

EPS=0.01
NUM_TEST=1000
OPP_POLICIES="random max exact self"

# Table I and Table II
for as_player in 1 2; do
    for turn in 1 2; do
        [[ $turn -eq 1 ]] && t='first' || t='second'
        printf "Agent trained as Player $as_player (playing as $t turn):\n\n"
        for policy in $OPP_POLICIES; do
            echo "vs. $policy policy:"
            python test.py --eps $EPS --num_test $NUM_TEST \
                           --load_path pretrained/p${as_player}-09999.pth \
                           --player $turn --opp_policy $policy 
            echo
        done
    done
done

# Table III
for turn in 1 2; do
    [[ $turn -eq 1 ]] && t='first' || t='second'
    printf "Player 1 as $t turn:\n\n"
    python test.py --eps $EPS --num_test $NUM_TEST \
                   --load_path pretrained/p1-09999.pth \
                   --pvp --player $turn
    echo
done
