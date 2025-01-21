#!/bin/bash

TEXTIDOTE_PATH=~/Software/textidote.jar

mkdir -p reports


echo $([[ "$1" -eq "-f" ]])
echo "Generating reports"

for file in chapters/*; do
  chapter_name=$(basename $file)
  if [[ chapters/$chapter_name -nt reports/$chapter_name.html ]] || [[ reports/dico.txt -nt reports/$chapter_name.html ]] || [[ "$1" == "-f" ]]; then
    echo "Generating report for $chapter_name"
    lang="en"
    if [[ $chapter_name == "remerciements.tex" ]] || [[ $chapter_name == "resume.tex" ]]; then
      lang="fr"
    fi
    java -jar $TEXTIDOTE_PATH --check $lang \
      --dict reports/dico.txt \
      --output html \
      --ignore sh:seclen,sh:capperiod,sh:c:noin,sh:figref,lt:en:SPACEX \
      chapters/$chapter_name > reports/$chapter_name.html
  else
    echo "$chapter_name is up to date"
  fi

done
