add_library(signal_shim SHARED signal_shim.c)
target_link_libraries(signal_shim PRIVATE -Wl,-z,now ${CMAKE_DL_LIBS} Threads::Threads m)
set_target_properties(signal_shim PROPERTIES PREFIX "") # signal_shim.so, as documented

add_executable(ptrace_shim ptrace_shim.c)
target_link_libraries(ptrace_shim PRIVATE m)

add_executable(spinners spinners.c)
target_link_libraries(spinners PRIVATE Threads::Threads)

add_executable(lifecycle_target lifecycle_target.c)
target_link_libraries(lifecycle_target PRIVATE Threads::Threads)

# Lifecycle tests: lifecycle_target checks its own invariants (see the modes
# in lifecycle_target.c) while one of the shims perturbs it. Each test is a
# POSIX sh script that starts with these variables set and an empty scratch
# directory $d:
#   $shim    signal_shim.so
#   $tracer  ptrace_shim
#   $target  lifecycle_target
# and these helpers:
#   check_log FILES   pause logs have 7 columns and ordered timestamps
#   field FILE NAME   value of NAME=... in a summary file
# Most tests set spinners=0 / -x 0 because lifecycle_target's workers spin.
set(LIFECYCLE_PRELUDE [=[
shim=$1 tracer=$2 target=$3 d=$4
rm -rf "$d" && mkdir -p "$d" || exit 1
check_log() {
  awk 'NF != 7 || $2 > $3 || $3 > $4 || $4 > $5 { bad++ } END { exit bad > 0 }' "$@"
}
field() {
  sed -E "s/.* $2=([^ ]+).*/\1/" "$1"
}
]=])

function(add_lifecycle_test name script)
  add_test(
    NAME sidecar_${name}
    COMMAND sh -c "${LIFECYCLE_PRELUDE}${script}" sh
            $<TARGET_FILE:signal_shim> $<TARGET_FILE:ptrace_shim> $<TARGET_FILE:lifecycle_target>
            ${CMAKE_CURRENT_BINARY_DIR}/lifecycle/${name}
  )
  set_tests_properties(sidecar_${name} PROPERTIES TIMEOUT 60)
endfunction()

# --- signal_shim ---------------------------------------------------------

add_lifecycle_test(signal_shim_abi [=[
# syscall() return values and errno under futex injection
LD_PRELOAD=$shim SIGNAL_SHIM=futex_prob=1,futex_delay=100 "$target" abi 0 0
]=])

add_lifecycle_test(signal_shim_old_option_names [=[
LD_PRELOAD=$shim SIGNAL_SHIM=futex=1,pause=100 "$target" abi 0 0
]=])

add_lifecycle_test(signal_shim_spin [=[
# spinning workers, many pauses
LD_PRELOAD=$shim SIGNAL_SHIM=rate=2000,spinners=0,warmup=1,futex_prob=0.1,log=$d \
  "$target" spin 8 5 || exit 1
s=$(ls "$d"/summary-*) || exit 1
cat "$s"
check_log "$d"/pauses-*.log && [ "$(field "$s" done)" -gt 0 ] &&
  [ "$(field "$s" stale_signals)" -eq 0 ] && grep -q lifecycle_target "$d"/maps-*-*.log
]=])

add_lifecycle_test(signal_shim_summary_to_stderr [=[
# without log=, the counters go to stderr
LD_PRELOAD=$shim SIGNAL_SHIM=rate=2000,spinners=0,warmup=1 "$target" spin 2 1 2> "$d/err" || exit 1
cat "$d/err"
grep -q "^signal_shim: pid=[0-9]* seed=0x[0-9a-f]* sent=[1-9]" "$d/err"
]=])

add_lifecycle_test(signal_shim_long_hold [=[
# holds longer than the abandon timeout: every logged pause must have held
# >= 1.5 s (t2 - t1), i.e. no slot was reclaimed while its handler slept
LD_PRELOAD=$shim SIGNAL_SHIM=rate=200,spinners=0,warmup=1,hold=1500000,abandon=100,log=$d \
  "$target" spin 8 6 || exit 1
s=$(ls "$d"/summary-*) || exit 1
cat "$s"
short=$(awk '($4 - $3) < 1500000000 { n++ } END { print n + 0 }' "$d"/pauses-*.log)
echo "short holds: $short"
check_log "$d"/pauses-*.log && [ "$short" -eq 0 ] && [ "$(field "$s" done)" -gt 0 ]
]=])

add_lifecycle_test(signal_shim_blocked_signal [=[
# workers block the pause signal for the first half, so requests are
# abandoned and their signals delivered late
LD_PRELOAD=$shim SIGNAL_SHIM=rate=2000,spinners=0,warmup=1,abandon=100,log=$d \
  "$target" block 8 6 || exit 1
s=$(ls "$d"/summary-*) || exit 1
cat "$s"
check_log "$d"/pauses-*.log && [ "$(field "$s" abandoned)" -gt 0 ]
]=])

add_lifecycle_test(signal_shim_handler_replaced [=[
# the program installs its own handler for the pause signal
LD_PRELOAD=$shim SIGNAL_SHIM=rate=2000,spinners=0,warmup=1,log=$d "$target" handler 4 4 || exit 1
s=$(ls "$d"/summary-*) || exit 1
cat "$s"
[ "$(field "$s" handler_replaced)" -eq 1 ]
]=])

foreach(mode fork exec leader-exit)
  add_lifecycle_test(signal_shim_${mode} "
LD_PRELOAD=$shim SIGNAL_SHIM=rate=2000,spinners=0,warmup=1,futex_prob=0.1 \"$target\" ${mode} 4 3
")
endforeach()

add_lifecycle_test(signal_shim_fd_exhaustion [=[
# a running program must not be ended while file descriptors are exhausted
# (lifecycle_target exits with 3; a shim that wrongly concludes all threads
# are gone would exit with 0)
LD_PRELOAD=$shim SIGNAL_SHIM=rate=2000,spinners=0,warmup=1 "$target" fdexhaust 4 6
rc=$?
echo "exit status $rc (expected 3)"
[ $rc -eq 3 ]
]=])

add_lifecycle_test(signal_shim_hold_and_maxhold [=[
# hold and maxhold together are rejected
LD_PRELOAD=$shim SIGNAL_SHIM=rate=10,hold=10,maxhold=100 "$target" abi 0 0 > "$d/out" 2>&1
rc=$?
cat "$d/out"
[ $rc -ne 0 ] && grep -q "mutually exclusive" "$d/out"
]=])

add_lifecycle_test(signal_shim_invalid_config [=[
LD_PRELOAD=$shim SIGNAL_SHIM=rate=abc "$target" abi 0 0 > "$d/out" 2>&1
rc=$?
cat "$d/out"
[ $rc -ne 0 ] && grep -q "invalid rate" "$d/out"
]=])

add_lifecycle_test(signal_shim_seed [=[
# a given seed is reported as is; without one, each process gets its own
# random seed; a malformed seed is rejected
for run in a b c; do
  mkdir "$d/$run" || exit 1
done
LD_PRELOAD=$shim SIGNAL_SHIM=seed=0x1234,log=$d/a "$target" spin 2 1 || exit 1
LD_PRELOAD=$shim SIGNAL_SHIM=log=$d/b "$target" spin 2 1 || exit 1
LD_PRELOAD=$shim SIGNAL_SHIM=log=$d/c "$target" spin 2 1 || exit 1
cat "$d"/*/summary-*
b=$(field "$d"/b/summary-* seed)
c=$(field "$d"/c/summary-* seed)
grep -q "seed=0x0000000000001234 " "$d"/a/summary-* && [ -n "$b" ] && [ "$b" != "$c" ] || exit 1
LD_PRELOAD=$shim SIGNAL_SHIM=seed=12x "$target" abi 0 0 > "$d/out" 2>&1
rc=$?
cat "$d/out"
[ $rc -ne 0 ] && grep -q "invalid seed" "$d/out"
]=])

# --- ptrace_shim ---------------------------------------------------------

add_lifecycle_test(ptrace_shim_spin [=[
# spinning workers, many pauses; warmup ends at once, but the maps snapshot
# must still be of the program, not of the sidecar before the exec
"$tracer" -r 2000 -H 0 -x 0 -w 1 -W 0 -s 1 -l "$d" -- "$target" spin 8 5 || exit 1
s=$(ls "$d"/summary-*) || exit 1
cat "$s"
check_log "$d"/pauses-*.log && grep -q "still_pending=0" "$s" && grep -q "foreign=0" "$s" &&
  grep -q lifecycle_target "$d"/maps-*-*.log
]=])

add_lifecycle_test(ptrace_shim_exec_failure [=[
# a program that cannot be executed is reported, with exit status 127
"$tracer" -- "$d/missing" > "$d/out" 2>&1
rc=$?
cat "$d/out"
[ $rc -eq 127 ] && grep -q "cannot execute" "$d/out"
]=])

add_lifecycle_test(ptrace_shim_invalid_options [=[
# malformed or out-of-range numbers are rejected, not read as 0
for opts in "-x abc" "-x 101" "-r -1" "-H 1ms" "-M 0" "-T 0" "-w 1.5x" "-s 12x" "-s -1"; do
  "$tracer" $opts -- "$target" abi 0 0 > "$d/out" 2>&1
  rc=$?
  echo "$opts: exit status $rc: $(head -n 1 "$d/out")"
  [ $rc -eq 2 ] && grep -q "invalid" "$d/out" || exit 1
done
]=])

add_lifecycle_test(ptrace_shim_no_pauses [=[
# -r 0 runs the program unperturbed
"$tracer" -r 0 -s 0x1234 -l "$d" -- "$target" spin 4 2 || exit 1
s=$(ls "$d"/summary-*) || exit 1
cat "$s"
grep -q "seed=0x0000000000001234 " "$s" && [ "$(field "$s" pauses)" -eq 0 ]
]=])

add_lifecycle_test(ptrace_shim_low_rate [=[
# a low rate does not keep the sidecar waiting after the program exits
start=$(date +%s)
"$tracer" -r 0.01 -w 0 -W 0 -- "$target" spin 2 1 || exit 1
elapsed=$(($(date +%s) - start))
echo "elapsed: $elapsed s"
[ $elapsed -le 3 ]
]=])

add_lifecycle_test(ptrace_shim_hold_and_maxhold [=[
# -H and -M together are rejected
"$tracer" -r 10 -H 10 -M 100 -- "$target" abi 0 0 > "$d/out" 2>&1
rc=$?
cat "$d/out"
[ $rc -ne 0 ] && grep -q "mutually exclusive" "$d/out"
]=])

add_lifecycle_test(ptrace_shim_abi [=[
# syscall() return values and errno while being paused
"$tracer" -r 2000 -H 0 -x 0 -w 0 -W 0 -s 1 -- "$target" abi 0 0
]=])

add_lifecycle_test(ptrace_shim_stop_continue [=[
# job-control stop and continue of the program while it is being paused
"$tracer" -r 1000 -H 1000 -x 0 -w 1 -W 0 -s 1 -- "$target" spin 4 5 &
tpid=$!
sleep 1.5
victim=$(pgrep -P $tpid | head -n 1)
kill -STOP "$victim"
sleep 1
state=$(awk '/^State:/ { print $2 }' /proc/$victim/status)
kill -CONT "$victim"
wait $tpid
rc=$?
echo "exit status $rc, stopped state was '$state'"
[ $rc -eq 0 ] && [ "$state" = T ]
]=])

foreach(mode exec leader-exit)
  add_lifecycle_test(ptrace_shim_${mode} "
\"$tracer\" -r 1000 -M 1000 -x 0 -w 1 -W 0 -s 1 -- \"$target\" ${mode} 4 3
")
endforeach()

add_lifecycle_test(ptrace_shim_exec_while_held [=[
# a thread exits (here: killed by another thread's exec) while the sidecar
# holds it; the sidecar must reap it, or the program can never be reaped and
# the sidecar waits forever (holds are long, so this is near certain)
for run in 1 2 3; do
  mkdir "$d/$run" || exit 1
  "$tracer" -r 2000 -H 2000 -x 0 -w 1 -W 0 -l "$d/$run" -- "$target" exec 4 3 || exit 1
  cat "$d/$run"/summary-*
  grep -q "still_pending=0" "$d/$run"/summary-* || exit 1
done
]=])
set_tests_properties(sidecar_ptrace_shim_exec_while_held PROPERTIES TIMEOUT 30)

add_lifecycle_test(ptrace_shim_killed [=[
# tracer killed: the program is left untraced, not stopped, and runs to
# completion
"$tracer" -r 2000 -H 1000 -x 0 -w 1 -W 0 -s 1 -- "$target" spin 4 4 &
tpid=$!
sleep 1
victim=$(pgrep -P $tpid | head -n 1)
kill -9 $tpid
wait $tpid
sleep 0.5
tracerpid=$(awk '/^TracerPid:/ { print $2 }' /proc/$victim/status)
state=$(awk '/^State:/ { print $2 }' /proc/$victim/status)
i=0
while [ -e /proc/$victim ] && [ $i -lt 100 ]; do
  sleep 0.1
  i=$((i + 1))
done
echo "TracerPid=$tracerpid State=$state"
[ "$tracerpid" = 0 ] && [ "$state" != T ] && [ ! -e /proc/$victim ]
]=])
