#!/usr/bin/env bash
set -e

VERSION="${VERSION:-latest}"
IMAGE=climvarchange:$VERSION

docker run --rm -v $(pwd):/usr/local/src/ClimVarChange "$IMAGE" "$@"
