#!/usr/bin/env bash
set -e

VERSION="${VERSION:-latest}"
IMAGE=climvarchange:$VERSION

docker build -t "$IMAGE" .
