#!/bin/sh
cd /Users/reiddye/darknet
./darknet classifier predict cfg/imagenet1k.data cfg/darknet19.cfg darknet19.weights data/testSquirrel.jpg
cd
