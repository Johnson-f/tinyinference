#!/bin/sh
set -eu

model=''
output=''
previous=''
for argument in "$@"; do
  if [ "$previous" = "--model" ]; then model="$argument"; fi
  if [ "$previous" = "--output_file" ]; then output="$argument"; fi
  previous="$argument"
done
[ -n "$model" ]
[ -n "$output" ]

directory=$(dirname "$model")
printf '%s' "$*" > "$directory/captured.args"
cat > "$directory/captured.stdin"
printf 'fake wav bytes' > "$output"
if [ "${model##*/}" = "voice-fail.onnx" ]; then
  printf 'fake piper error' >&2
  exit 7
fi
