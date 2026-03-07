@echo off
setlocal

set EXE=.\build\ct_train.exe
set TRAIN=data/tokenized/train.bin
set VAL=data/tokenized/val.bin
set COMMON=--benchmark 1 --seed 42 --eval-interval 0 --checkpoint-interval 0 --sample-interval 0

echo ============================================================
echo  Stage 1: Quick Profiling (200 steps per config)
echo  Estimated time: ~1.1 hours total
echo ============================================================

echo.
echo --- Config A: Baseline (FP32, no checkpointing) ---
%EXE% --train %TRAIN% --val %VAL% --steps 200 %COMMON% --log experiments/quick_A_baseline.csv

echo.
echo --- Config B: Mixed Precision only ---
%EXE% --train %TRAIN% --val %VAL% --steps 200 %COMMON% --mixed-precision 1 --log experiments/quick_B_mp.csv

echo.
echo --- Config C: Gradient Checkpointing (all layers) ---
%EXE% --train %TRAIN% --val %VAL% --steps 200 %COMMON% --grad-checkpoint 1 --grad-checkpoint-layers all --log experiments/quick_C_gc_all.csv

echo.
echo --- Config D: Gradient Checkpointing (even layers only) ---
%EXE% --train %TRAIN% --val %VAL% --steps 200 %COMMON% --grad-checkpoint 1 --grad-checkpoint-layers 0,2,4,6,8 --log experiments/quick_D_gc_half.csv

echo.
echo --- Config E: Mixed Precision + GC (all layers) ---
%EXE% --train %TRAIN% --val %VAL% --steps 200 %COMMON% --mixed-precision 1 --grad-checkpoint 1 --grad-checkpoint-layers all --log experiments/quick_E_mp_gc_all.csv

echo.
echo --- Config F: Mixed Precision + GC (even layers) ---
%EXE% --train %TRAIN% --val %VAL% --steps 200 %COMMON% --mixed-precision 1 --grad-checkpoint 1 --grad-checkpoint-layers 0,2,4,6,8 --log experiments/quick_F_mp_gc_half.csv

echo.
echo ============================================================
echo  Stage 1 Complete. Check experiments/quick_*.csv
echo ============================================================
echo.
