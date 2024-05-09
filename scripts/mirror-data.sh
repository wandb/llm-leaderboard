#!/bin/bash
# Mirror public Wandb artifacts to a private server
set -eou pipefail

host=$1 # private server url
dest=$2 # user/project on private server

function login_public(){
  echo "Preparing to mirror below artifacts to $dest:"
  echo
  cat scripts/artifacts.txt
  echo
  echo "First, your local config will be deleted in order to use the public server"
  echo "Preparing to delete ~/.config/wandb/settings"

  echo "To continue push Enter, to exist push CTRL+C"
  read ok

  # Delete any existing config
  rm -f ~/.config/wandb/settings

  wandb login --relogin
}

function download(){
  cat scripts/artifacts.txt | while read artifact; do
    echo
    echo
    echo "Downloading $artifact"
    project=$(basename $dest)
    python scripts/download_artifact.py $artifact $project
  done
  echo
  echo
}


function login_private(){
  echo "Now you need to log in to your local server: $host"

  wandb login --relogin --host="$host"

  echo
  echo
}

function upload(){
  cat scripts/artifacts.txt | while read artifact; do
    # have to remove version part (:v0)
    artifact_name=$(basename $artifact| sed 's/:.*//')
    artifact_dest=$(echo "$dest/$artifact_name")
    # This contains the version part
    local_path="artifacts/$(basename $artifact)"

    echo "Preparing to upload artifact to $artifact_dest"
    python scripts/upload_artifact.py $artifact_dest "$local_path"
  done 
}

function main(){
  login_public
  download
  login_private
  upload

  echo
  echo "All done"
}

main
