#!/bin/sh
python pairwise_pref_eval.py test_data/preferences.txt test_data/results.txt > test_data/expected_results.txt.tmp
diff test_data/expected_results.txt* > /dev/null
if [[ $? -eq 0 ]]; then
  echo "Results match!"
  rm test_data/expected_results.txt.tmp
else
  echo "Results don't match! See:"
  diff test_data/expected_results.txt*
fi
