#!/bin/bash

file_exec="rep.py"

rm -Rf dtws_*
./$file_exec
rm -Rf __pycache__
