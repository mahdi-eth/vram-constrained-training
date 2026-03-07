@echo off
setlocal

set EXE=.\build\ct_train.exe
set TRAIN=data/tokenized/train.bin
set VAL=data/tokenized/val.bin
set COMMON=--benchmark 1 --seed 42 --eval-interval 500 --checkpoint-interval 0 --sample-interval 0

echo ============================================================
echo  Stage 2: Convergence Runs (2000 steps per config)
echo  Estimated time: ~11.4 hours total — run overnight
echo ============================================================

echo.
echo --- Config A: Baseline ---
%EXE% --train %TRAIN% --val %VAL% --steps 2000 %COMMON% --log experiments/full_A_baseline.csv

echo.
echo --- Config B: Mixed Precision ---
%EXE% --train %TRAIN% --val %VAL% --steps 2000 %COMMON% --mixed-precision 1 --log experiments/full_B_mp.csv

echo.
echo --- Config C: GC All ---
%EXE% --train %TRAIN% --val %VAL% --steps 2000 %COMMON% --grad-checkpoint 1 --grad-checkpoint-layers all --log experiments/full_C_gc_all.csv

echo.
echo --- Config D: GC Half ---
%EXE% --train %TRAIN% --val %VAL% --steps 2000 %COMMON% --grad-checkpoint 1 --grad-checkpoint-layers 0,2,4,6,8 --log experiments/full_D_gc_half.csv

echo.
echo --- Config E: MP + GC All ---
%EXE% --train %TRAIN% --val %VAL% --steps 2000 %COMMON% --mixed-precision 1 --grad-checkpoint 1 --grad-checkpoint-layers all --log experiments/full_E_mp_gc_all.csv

echo.
echo --- Config F: MP + GC Half ---
%EXE% --train %TRAIN% --val %VAL% --steps 2000 %COMMON% --mixed-precision 1 --grad-checkpoint 1 --grad-checkpoint-layers 0,2,4,6,8 --log experiments/full_F_mp_gc_half.csv

echo.
echo ============================================================
echo  Stage 2 Complete. Check experiments/full_*.csv
echo ============================================================
echo.
