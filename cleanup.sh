#!/usr/bin/env bash
#
# Cleans up the data/ directory of any previous
# training results and log and creates a fresh
# directory with a valid structure.
function clearAndRecreate() {
  rm -frv data/

  # Remove previous compressed results
  rm results.tar.gz

  # Remove nohup.out if it exists
  # This stores the training process' output
  # when its running on Google Cloud
  rm nohup.out

  mkdir data/
  mkdir data/logs
  mkdir data/checkpoints

  touch data/logs/training.log
  touch nohup.out
}

echo "The following files will be deleted:"
find data/
echo "results.tar.gz"
echo "nohup.out"

while true; do
  read -p "Do you wish to continue?" yn
  case $yn in
  [Yy]*)
    clearAndRecreate
    break
    ;;
  [Nn]*) exit ;;
  *) echo "Please answer yes or no." ;;
  esac
done
