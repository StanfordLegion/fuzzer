# Quickstart

Everything is packaged into a single `do_all.sh` script that takes the
name of the machine to run on. E.g.:

```bash
./experiment/do_all.sh sapling
```

See the contents of `do_all.sh` to look at the individual steps.

This will launch a set of jobs into the batch system on the machine,
check `slurm-*.out` for output from the specific runs.

Note: this will download Legion if it is not already present, but will
not update Legion. Any existing Legion will be reconfigured/rebuilt.
