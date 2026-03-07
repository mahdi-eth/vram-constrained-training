@echo off
set EXE=.\build\ct_train.exe
set TRAIN=data/tokenized/train.bin
set VAL=data/tokenized/val.bin
set COMMON=--benchmark 1 --seed 42 --eval-interval 0 --checkpoint-interval 0 --sample-interval 0

echo ============================================================
echo  OOM Boundary Experiment v2
echo  Strategy A: seq=384, default scratch (300 MB)
echo  Strategy B: seq=256, reduced scratch (195 MB)
echo ============================================================
echo.

echo === Strategy A: seq=384, no GC (expected OOM) ===
%EXE% --train %TRAIN% --val %VAL% --steps 10 %COMMON% --seq 384 --log experiments/oom_A1_seq384_baseline.csv
echo Exit code: %ERRORLEVEL%
echo.

echo === Strategy A: seq=384, GC all (expected success) ===
%EXE% --train %TRAIN% --val %VAL% --steps 50 %COMMON% --seq 384 --grad-checkpoint 1 --log experiments/oom_A2_seq384_gc.csv
echo Exit code: %ERRORLEVEL%
echo.

echo ============================================================
echo  Strategy B: seq=256, scratch=195 MB
echo ============================================================
echo.

echo === Strategy B: seq=256, scratch=195, no GC (expected OOM) ===
%EXE% --train %TRAIN% --val %VAL% --steps 10 %COMMON% --seq 256 --scratch-mb 195 --log experiments/oom_B1_seq256_scratch195_baseline.csv
echo Exit code: %ERRORLEVEL%
echo.

echo === Strategy B: seq=256, scratch=195, GC all (expected success) ===
%EXE% --train %TRAIN% --val %VAL% --steps 50 %COMMON% --seq 256 --scratch-mb 195 --grad-checkpoint 1 --log experiments/oom_B2_seq256_scratch195_gc.csv
echo Exit code: %ERRORLEVEL%
echo.

echo Done. At least one strategy should show: baseline OOM + GC success.
pause
