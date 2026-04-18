#! /bin/bash

IMAGES_PATH=$1
LABELS_PATH=$2
count=0

flatten_dir() {
    local dir="$1"
    [ -d "$dir" ] || return 1

    (
        cd "$dir"
        find . -mindepth 2 -type f -exec sh -c 'for f do mv -n "$f" .; done' sh {} +
        find . -type d -empty -delete
    )
}

flatten_dir "$IMAGES_PATH"
flatten_dir "$LABELS_PATH"

for image in "$IMAGES_PATH"/* ; do
    base=$(basename "${image}")
    expectedlabel=${base/_ref/_label}
    num=$(printf "%03d" "$count")

    if [ -e "$LABELS_PATH/$expectedlabel" ]; then
        mv "$image" "$IMAGES_PATH/Aneyrysm_${num}_0001.nii.gz"
        mv "$LABELS_PATH/$expectedlabel" "$LABELS_PATH/Aneyrysm_${num}.nii.gz"
    fi
    ((count++))
done

echo "Done"


