#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

ipython $DIR/setup.ipy
ipython $DIR/dfl_train.ipy -- $@