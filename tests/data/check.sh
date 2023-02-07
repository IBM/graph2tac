#!/bin/sh

USAGE="
usage: $(basename $0) [--help] [--all] [--record] [--yes]
\n
\noptional arguments:
\n\t-h,--help \t show this help message and exit
\n\t-a,--all \t run all tests (i.e. do not stop after a test fails)
\n\t-r,--record \t when results don't match the stored fingerprints, store them as the new fingerprints
\n\t-y,--yes \t answer yes to all questions
"

# parse command line arguments

ARGS=$(getopt -a --options hary --long "help,all,record,yes" -- "$@")
eval set -- "$ARGS"

all="false"
record="false"
yes="false"
while true; do
  case "$1" in
    -a|--all)
      all="true"
      shift;;
    -r|--record)
      record="true"
      shift;;
	  -y|--yes)
      yes="true"
      shift;;
    -h|--help)
      echo $USAGE
      exit 0;;
    --)
      break;;
    *)
      printf "Unknown option %s\n" "$1"
      exit 1;;
  esac
done

# TODO (VP): as more tests show up refactor the code as a loop over tests
# or switch to pytest

BASEDIR=$(dirname "$0")

TESTDIR0="mini_stdlib"
TESTDIR1="propchain_large"
TESTDIR2="propchain_small"
TESTDIR3="propositional"

TEST_DIRECTORIES="$TESTDIR0 $TESTDIR1 $TESTDIR2 $TESTDIR3"

summary=""
tests_passed=0
tests_failed=0
for TESTDIR in $TEST_DIRECTORIES
do
  echo "================================================================"
  echo "Running tests in directory $TESTDIR"
  cat "$BASEDIR/$TESTDIR/README.md"

  SPEC_DIRECTORIES=$(find "$BASEDIR/$TESTDIR" -maxdepth 1 -mindepth 1 -type d -name "test*" -print0 | xargs -0)
  for SPECDIR in $SPEC_DIRECTORIES
  do
    echo "Running sub-test on directory $TESTDIR with spec $SPECDIR"

    RESULTFILE="$(mktemp)"
    echo "The fingerprint for this run will be recorded in $RESULTFILE"

    TEMPLOGFILE="$(mktemp)"
    echo "The log will be recorded in $TEMPLOGFILE"

    echo "Running g2t-train"
    export CUDA_VISIBLE_DEVICES=-1 && g2t-train "$BASEDIR/$TESTDIR/dataset" "$SPECDIR/params.yaml" --fingerprint="$RESULTFILE" 2>&1 | tee "$TEMPLOGFILE"

    EXP_RESULTFILE="$SPECDIR/fingerprint.out"
    echo "Expected fingerprint in $EXP_RESULTFILE is:\n"
    EXP_RESULT="$(cat $EXP_RESULTFILE)"
    echo "$EXP_RESULT\n"

    RESULT="$(cat $RESULTFILE)"
    echo "Computed fingerprint in $RESULTFILE is:\n"
    echo "$RESULT\n"

    if cmp -s "$RESULTFILE" "$EXP_RESULTFILE" ; then
	tests_passed=$((tests_passed+1))
	summary="$summary\n$SPECDIR ==> PASSED ($TEMPLOGFILE)"
	echo "CHECK PASSED"
	echo "===================="
    else
	tests_failed=$((tests_failed+1))
	summary="$summary\n$SPECDIR ==> FAILED ($TEMPLOGFILE)"
	echo "CHECK FAILED"
	echo "===================="
	echo "See the training log file in $TEMPLOGFILE"

	if [ $record = true ] ; then
	  if [ $yes = true ] ; then
	    echo "Overwriting fingerprint"
	    cp "$RESULTFILE" "$EXP_RESULTFILE"
	  else
	    while true; do
	      read -p "Do you want to record this fingerprint as the new reference [y/n]? (use --yes to skip this question)" yn
	      case $yn in
		[Yy]* ) echo "Overwriting fingerprint"; cp "$RESULTFILE" "$EXP_RESULTFILE"; break;;
		[Nn]* ) break;;
		* ) echo "Please answer yes or no.";;
	      esac
	    done
	  fi
	fi

	if [ $all = false ] ; then
	  echo "Aborting checks, use --all to run all tests regardless of failures"
	  exit 1
	fi
    fi
  done
done

echo "SUMMARY: passed $tests_passed/$((tests_passed+tests_failed)) tests"
echo $summary
