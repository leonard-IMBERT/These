#!/bin/bash

TEXTIDOTE_PATH=~/Downloads/textidote.jar-0.9.jar

mkdir -p reports


echo "Generating reports"

for file in chapters/*; do
  chapter_name=$(basename $file)
  if [[ chapters/$chapter_name -nt reports/$chapter_name.html ]] || [[ reports/dico.txt -nt reports/$chapter_name.html ]]; then
    echo "Generating report for $chapter_name"
    java -jar $TEXTIDOTE_PATH --check en \
      --dict reports/dico.txt \
      --output html \
      --ignore sh:c:noin,sh:figref,lt:en:SPACEX \
      chapters/$chapter_name > reports/$chapter_name.html
  else
    echo "$chapter_name is up to date"
  fi

done
