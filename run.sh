#!/bin/bash
if [ "$#" -ne 1 ] ; then
  echo "Missing Parameters ..."
  echo "Usage: $0 dataset[small|small_old|small_tree|small_path] " >&2
  exit 1
fi
